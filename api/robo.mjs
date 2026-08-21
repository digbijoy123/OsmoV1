/**
 * ROBO AIOS v2.12 — provider-neutral AI adapter.
 * Vercel serverless function: /api/robo
 *
 * Current provider: Gemini
 * Model: Gemini 3.1 Flash-Lite
 *
 * Supports text chat plus one optional camera image per request.
 * The Gemini API key stays server-side in GEMINI_API_KEY.
 */

const SYSTEM_PROMPT =
  'You are Robo, a warm, concise AI companion. ' +
  'Respond naturally for spoken conversation. ' +
  'Keep answers reasonably short unless the user asks for detail. ' +
  'Do not mention being a language model unless directly asked. ' +
  'When a camera image is attached, use it to answer the user’s question. ' +
  'If the image is unclear or irrelevant, say so briefly instead of inventing details.';

function makeError(message, code, status) {
  const err = new Error(message);
  err.code = code;
  err.status = status;
  return err;
}

function normalizeImage(image) {
  if (!image || typeof image !== 'object') return null;

  const mimeType = String(image.mimeType || '').toLowerCase();
  let data = String(image.data || '').trim();

  if (!mimeType || !mimeType.startsWith('image/')) {
    throw makeError('Invalid vision image MIME type', 'INVALID_IMAGE', 400);
  }

  // Accept either raw base64 or a data URL from the browser.
  const comma = data.indexOf(',');
  if (data.startsWith('data:') && comma >= 0) {
    data = data.slice(comma + 1);
  }

  if (!data) {
    throw makeError('Vision image data is empty', 'INVALID_IMAGE', 400);
  }

  // Keep accidental oversized requests out of the serverless function.
  // The browser currently sends a small JPEG frame.
  if (data.length > 12_000_000) {
    throw makeError('Vision image is too large', 'IMAGE_TOO_LARGE', 413);
  }

  return { mimeType, data };
}

const PROVIDERS = {
  gemini: {
    async generate({ messages, image }) {
      const key = process.env.GEMINI_API_KEY;

      if (!key) {
        throw makeError(
          'GEMINI_API_KEY is not configured',
          'AI_NOT_CONFIGURED',
          503,
        );
      }

      const normalizedImage = normalizeImage(image);

      const contents = messages.map((message, index) => {
        const parts = [
          {
            text: String(message.content ?? ''),
          },
        ];

        // Attach the live camera frame to the latest user message only.
        if (
          normalizedImage &&
          message.role === 'user' &&
          index === messages.length - 1
        ) {
          parts.push({
            inline_data: {
              mime_type: normalizedImage.mimeType,
              data: normalizedImage.data,
            },
          });
        }

        return {
          role: message.role === 'assistant' ? 'model' : 'user',
          parts,
        };
      });

      const model =
        process.env.GEMINI_MODEL || 'gemini-3.1-flash-lite';

      const response = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'x-goog-api-key': key,
          },
          body: JSON.stringify({
            systemInstruction: {
              parts: [{ text: SYSTEM_PROMPT }],
            },
            contents,
            generationConfig: {
              temperature: 0.7,
              maxOutputTokens: 300,
            },
          }),
        },
      );

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        const message =
          data?.error?.message ||
          `Gemini request failed (${response.status})`;

        throw makeError(
          message,
          data?.error?.status ||
            data?.error?.code ||
            'GEMINI_API_ERROR',
          response.status,
        );
      }

      const text =
        data?.candidates?.[0]?.content?.parts
          ?.map((part) => part?.text || '')
          .join('')
          .trim() || '';

      if (!text) {
        throw makeError(
          'Gemini returned no text',
          'AI_EMPTY_RESPONSE',
          502,
        );
      }

      return {
        text,
        provider: 'gemini',
        model,
        vision: Boolean(normalizedImage),
      };
    },
  },
};

function json(res, status, body) {
  return res.status(status).json(body);
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  const providerName = String(
    process.env.AI_PROVIDER || 'gemini',
  ).toLowerCase();

  if (req.method === 'GET') {
    const configured =
      providerName === 'gemini'
        ? Boolean(process.env.GEMINI_API_KEY)
        : false;

    return json(res, 200, {
      ok: true,
      provider: providerName,
      configured,
      model:
        providerName === 'gemini'
          ? process.env.GEMINI_MODEL || 'gemini-3.1-flash-lite'
          : null,
      vision: providerName === 'gemini',
    });
  }

  if (req.method !== 'POST') {
    return json(res, 405, {
      error: 'Method not allowed',
      code: 'METHOD_NOT_ALLOWED',
    });
  }

  try {
    const body =
      typeof req.body === 'string'
        ? JSON.parse(req.body)
        : req.body || {};

    const provider = PROVIDERS[providerName];

    if (!provider) {
      return json(res, 400, {
        error: `Unsupported AI provider: ${providerName}`,
        code: 'UNSUPPORTED_PROVIDER',
      });
    }

    const messages = Array.isArray(body.messages)
      ? body.messages
      : [];

    const cleanMessages = messages
      .filter(
        (m) =>
          m &&
          (m.role === 'user' || m.role === 'assistant'),
      )
      .slice(-12)
      .map((m) => ({
        role: m.role,
        content: String(m.content || '').slice(0, 12000),
      }))
      .filter((m) => m.content.trim());

    if (!cleanMessages.length) {
      return json(res, 400, {
        error: 'No conversation messages supplied',
        code: 'EMPTY_INPUT',
      });
    }

    const result = await provider.generate({
      messages: cleanMessages,
      image: body.image || null,
    });

    return json(res, 200, result);
  } catch (error) {
    const status = Number.isInteger(error?.status)
      ? error.status
      : 500;

    return json(res, status, {
      error: error?.message || 'AI provider error',
      code: error?.code || 'AI_PROVIDER_ERROR',
    });
  }
}
