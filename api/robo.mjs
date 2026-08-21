/**
 * ROBO AIOS v2.12 — provider-neutral AI adapter.
 * Vercel serverless function: /api/robo
 *
 * Current provider: Gemini
 * Model: Gemini 3.1 Flash-Lite
 *
 * The Gemini API key stays server-side in:
 * GEMINI_API_KEY
 */

const PROVIDERS = {
  gemini: {
    async generate({ messages }) {
      const key = process.env.GEMINI_API_KEY;

      if (!key) {
        const err = new Error('GEMINI_API_KEY is not configured');
        err.code = 'AI_NOT_CONFIGURED';
        err.status = 503;
        throw err;
      }

      const contents = messages.map((message) => ({
        role: message.role === 'assistant' ? 'model' : 'user',
        parts: [
          {
            text: String(message.content ?? ''),
          },
        ],
      }));

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
              parts: [
                {
                  text:
                    'You are Robo, a warm, concise AI companion. ' +
                    'Respond naturally for spoken conversation. ' +
                    'Keep answers reasonably short unless the user asks for detail. ' +
                    'Do not mention being a language model unless directly asked.',
                },
              ],
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

        const err = new Error(message);
        err.code =
          data?.error?.status ||
          data?.error?.code ||
          'GEMINI_API_ERROR';
        err.status = response.status;

        throw err;
      }

      const text =
        data?.candidates?.[0]?.content?.parts
          ?.map((part) => part?.text || '')
          .join('')
          .trim() || '';

      if (!text) {
        const err = new Error('Gemini returned no text');
        err.code = 'AI_EMPTY_RESPONSE';
        err.status = 502;
        throw err;
      }

      return {
        text,
        provider: 'gemini',
        model,
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
    });

    return json(res, 200, result);
  } catch (error) {
    const status =
      Number.isInteger(error?.status)
        ? error.status
        : 500;

    return json(res, status, {
      error: error?.message || 'AI provider error',
      code: error?.code || 'AI_PROVIDER_ERROR',
    });
  }
}
