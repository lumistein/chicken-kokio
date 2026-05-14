const crypto = require("crypto");

const GOOGLE_SCOPE = "https://www.googleapis.com/auth/cloud-platform";
const GOOGLE_TIMEOUT_MS = 25_000;
const DEFAULT_PROMPT = [
  "이미지 안에 노란색 글씨로 표시된 쿨타임 숫자만 읽어주세요.",
  "다른 텍스트, 단위, 설명 없이 숫자만 답하세요.",
  "숫자가 보이지 않으면 NONE만 답하세요."
].join(" ");

const tokenCache = new Map();

module.exports = async function handler(req, res) {
  if (req.method === "OPTIONS") {
    res.statusCode = 204;
    return res.end();
  }

  if (req.method !== "POST") {
    return sendJson(res, 405, { error: "POST만 지원합니다." });
  }

  try {
    const body = await readJsonBody(req);
    const provider = ["vertex", "studio", "custom"].includes(body.provider) ? body.provider : "vertex";
    const prompt = String(body.prompt || DEFAULT_PROMPT);
    const imageBase64 = cleanBase64(body.imageBase64);

    if (!imageBase64) {
      return sendJson(res, 400, { error: "imageBase64가 필요합니다." });
    }

    let text;
    if (provider === "studio") {
      text = await callAIStudio(body.studio || {}, imageBase64, prompt);
    } else if (provider === "custom") {
      text = await callCustomModel(body.custom || {}, imageBase64, prompt);
    } else {
      text = await callVertexAI(body.vertex || {}, imageBase64, prompt);
    }

    return sendJson(res, 200, { text });
  } catch (error) {
    const status = Number.isInteger(error.status) ? error.status : 500;
    return sendJson(res, status, {
      error: error.message || "서버 오류가 발생했습니다.",
      source: error.source || "server",
      upstreamStatus: error.upstreamStatus
    });
  }
};

function sendJson(res, status, data) {
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json; charset=utf-8");
  res.end(JSON.stringify(data));
}

async function readJsonBody(req) {
  if (Buffer.isBuffer(req.body)) {
    const raw = req.body.toString("utf8");
    return raw ? JSON.parse(raw) : {};
  }
  if (req.body && typeof req.body === "object") {
    return req.body;
  }
  if (typeof req.body === "string") {
    return JSON.parse(req.body || "{}");
  }

  const chunks = [];
  for await (const chunk of req) {
    chunks.push(Buffer.from(chunk));
  }
  const raw = Buffer.concat(chunks).toString("utf8");
  return raw ? JSON.parse(raw) : {};
}

function cleanBase64(value) {
  if (typeof value !== "string") return "";
  const commaIndex = value.indexOf(",");
  return (commaIndex >= 0 ? value.slice(commaIndex + 1) : value).trim();
}

async function callAIStudio(config, imageBase64, prompt) {
  const apiKey = requireString(config.apiKey, "Google AI Studio API Key가 필요합니다.");
  const modelId = encodeURIComponent(config.modelId || "gemini-2.5-flash");
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelId}:generateContent?key=${encodeURIComponent(apiKey)}`;
  const response = await fetchWithTimeout(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{
        parts: [
          {
            inline_data: {
              mime_type: "image/png",
              data: imageBase64
            }
          },
          { text: prompt }
        ]
      }],
      generationConfig: {
        temperature: 0,
        maxOutputTokens: 8
      }
    })
  }, "Google AI Studio generateContent");

  return extractGeminiText(await parseGoogleResponse(response, "Google AI Studio"));
}

async function callVertexAI(config, imageBase64, prompt) {
  const projectId = requireString(config.projectId, "Vertex AI Project ID가 필요합니다.");
  const clientEmail = requireString(config.clientEmail, "Vertex AI Client Email이 필요합니다.");
  const privateKey = requireString(config.privateKey, "Vertex AI Private Key가 필요합니다.");
  const location = String(config.location || "us-central1").trim();
  const modelId = String(config.modelId || "gemini-2.5-flash").trim();
  const token = await getVertexAccessToken({ clientEmail, privateKey });
  const url = `https://${encodeURIComponent(location)}-aiplatform.googleapis.com/v1/projects/${encodeURIComponent(projectId)}/locations/${encodeURIComponent(location)}/publishers/google/models/${encodeURIComponent(modelId)}:generateContent`;

  const response = await fetchWithTimeout(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "Authorization": `Bearer ${token}`
    },
    body: JSON.stringify({
      contents: [{
        role: "user",
        parts: [
          { text: prompt },
          {
            inlineData: {
              mimeType: "image/png",
              data: imageBase64
            }
          }
        ]
      }],
      generationConfig: {
        temperature: 0,
        maxOutputTokens: 8
      }
    })
  }, "Vertex AI generateContent");

  return extractGeminiText(await parseGoogleResponse(response, "Vertex AI"));
}

async function callCustomModel(config, imageBase64, prompt) {
  const endpointUrl = normalizeChatCompletionsUrl(requireString(config.url, "Custom Endpoint URL이 필요합니다."));
  const modelId = requireString(config.modelId, "Custom Model ID가 필요합니다.");
  const apiKey = String(config.apiKey || "").trim();
  const partOrder = config.partOrder === "promptFirst" ? "promptFirst" : "imageFirst";
  const imagePart = {
    type: "image_url",
    image_url: {
      url: `data:image/png;base64,${imageBase64}`
    }
  };
  const promptPart = {
    type: "text",
    text: prompt
  };
  const content = partOrder === "promptFirst"
    ? [promptPart, imagePart]
    : [imagePart, promptPart];
  const headers = { "Content-Type": "application/json" };
  if (apiKey) {
    headers.Authorization = `Bearer ${apiKey}`;
  }

  const response = await fetchWithTimeout(endpointUrl, {
    method: "POST",
    headers,
    body: JSON.stringify({
      model: modelId,
      messages: [{
        role: "user",
        content
      }],
      temperature: 0,
      max_tokens: 8
    })
  }, "Custom chat/completions");

  return extractCustomText(await parseUpstreamResponse(response, "Custom model", "custom"));
}

function requireString(value, message) {
  const text = String(value || "").trim();
  if (!text) {
    const error = new Error(message);
    error.status = 400;
    throw error;
  }
  return text;
}

function normalizeChatCompletionsUrl(rawUrl) {
  let url;
  try {
    url = new URL(rawUrl);
  } catch (_) {
    const error = new Error("Custom Endpoint URL 형식이 올바르지 않습니다.");
    error.status = 400;
    throw error;
  }
  const pathname = url.pathname.replace(/\/+$/g, "");
  if (pathname.endsWith("/chat/completions")) {
    url.pathname = pathname;
    return url.toString();
  }
  url.pathname = `${pathname}/chat/completions`;
  return url.toString();
}

async function getVertexAccessToken(config) {
  const cacheKey = config.clientEmail;
  const cached = tokenCache.get(cacheKey);
  if (cached && cached.expiresAt - Date.now() > 90_000) {
    return cached.accessToken;
  }

  const jwt = createServiceAccountJwt(config.clientEmail, config.privateKey, GOOGLE_SCOPE);
  const form = new URLSearchParams();
  form.set("grant_type", "urn:ietf:params:oauth:grant-type:jwt-bearer");
  form.set("assertion", jwt);

  const response = await fetchWithTimeout("https://oauth2.googleapis.com/token", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: form
  }, "Google OAuth token");
  const data = await parseGoogleResponse(response, "Google OAuth");

  if (!data.access_token) {
    throw new Error("OAuth access_token을 받지 못했습니다.");
  }

  tokenCache.set(cacheKey, {
    accessToken: data.access_token,
    expiresAt: Date.now() + Math.max(0, (data.expires_in || 3600) - 60) * 1000
  });

  return data.access_token;
}

function createServiceAccountJwt(clientEmail, privateKey, scope) {
  const iat = Math.floor(Date.now() / 1000);
  const exp = iat + 3600;
  const header = base64UrlJson({ alg: "RS256", typ: "JWT" });
  const claims = base64UrlJson({
    iss: clientEmail,
    scope,
    aud: "https://oauth2.googleapis.com/token",
    exp,
    iat
  });
  const signingInput = `${header}.${claims}`;
  const signer = crypto.createSign("RSA-SHA256");
  signer.update(signingInput);
  signer.end();
  const normalizedKey = privateKey.replace(/\\n/g, "\n");
  const signature = base64Url(signer.sign(normalizedKey));
  return `${signingInput}.${signature}`;
}

function base64UrlJson(value) {
  return base64Url(Buffer.from(JSON.stringify(value), "utf8"));
}

function base64Url(buffer) {
  return Buffer.from(buffer)
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/g, "");
}

async function fetchWithTimeout(url, options, label) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), GOOGLE_TIMEOUT_MS);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } catch (error) {
    if (error.name === "AbortError") {
      const timeoutError = new Error(`${label} 요청이 ${Math.round(GOOGLE_TIMEOUT_MS / 1000)}초 안에 끝나지 않았습니다.`);
      timeoutError.status = 504;
      timeoutError.source = "timeout";
      throw timeoutError;
    }
    error.source = error.source || "network";
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

async function parseGoogleResponse(response, label) {
  return parseUpstreamResponse(response, label, "google");
}

async function parseUpstreamResponse(response, label, source) {
  const text = await response.text();
  let data;
  try {
    data = text ? JSON.parse(text) : {};
  } catch (_) {
    const error = new Error(`${label} 응답을 JSON으로 읽지 못했습니다: ${text.slice(0, 180)}`);
    error.status = response.ok ? 500 : response.status;
    error.source = source;
    error.upstreamStatus = response.status;
    throw error;
  }

  if (!response.ok) {
    const error = new Error(extractErrorMessage(data) || response.statusText || `${label} 요청 실패`);
    error.status = response.status;
    error.source = source;
    error.upstreamStatus = response.status;
    throw error;
  }

  return data;
}

function extractErrorMessage(data) {
  if (typeof data?.error === "string") return data.error;
  if (typeof data?.error?.message === "string") return data.error.message;
  if (typeof data?.message === "string") return data.message;
  return "";
}

function extractGeminiText(data) {
  const parts = data?.candidates?.[0]?.content?.parts || [];
  const text = parts.map((part) => part.text || "").join("").trim();
  if (!text) {
    const reason = data?.candidates?.[0]?.finishReason;
    throw new Error(`Gemini 응답에 텍스트가 없습니다${reason ? ` (${reason})` : ""}.`);
  }
  return text;
}

function extractCustomText(data) {
  if (typeof data?.output_text === "string" && data.output_text.trim()) {
    return data.output_text.trim();
  }

  const message = data?.choices?.[0]?.message;
  if (typeof message?.content === "string" && message.content.trim()) {
    return message.content.trim();
  }
  if (Array.isArray(message?.content)) {
    const text = message.content
      .map((part) => part?.text || part?.content || "")
      .join("")
      .trim();
    if (text) return text;
  }

  const legacyText = data?.choices?.[0]?.text;
  if (typeof legacyText === "string" && legacyText.trim()) {
    return legacyText.trim();
  }

  try {
    return extractGeminiText(data);
  } catch (_) {
    throw new Error("Custom 모델 응답에서 텍스트를 찾지 못했습니다.");
  }
}
