/**
 * ROBO AIOS v2.12 — provider-neutral AI adapter.
 * Vercel serverless function: /api/robo
 *
 * Provider selection is controlled by AI_PROVIDER.
 * Current provider: openai.
 * Future providers can implement the same generate() contract.
 */

const PROVIDERS = {
  openai: {
    async generate({ messages, model }) {
      const key = process.env.OPENAI_API_KEY;
      if (!key) {
        const err = new Error('OPENAI_API_KEY is not configured');
        err.code = 'AI_NOT_CONFIGURED';
        err.status = 503;
        throw err;
      }

      const input = messages.map((m) => ({
        role: m.role,
        content: [{ type: 'input_text', text: String(m.content ?? '') }],
      }));

      const response = await fetch('https://api.openai.com/v1/responses', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          Authorization: `Bearer ${key}`,
        },
        body: JSON.stringify({
          model: process.env.OPENAI_MODEL || 'gpt-5.6',
          input,
          instructions:
            'You are Robo, a warm, concise AI companion. Respond naturally for spoken conversation. Keep answers reasonably short unless the user asks for detail.',
          store: false,
        }),
      });

      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        const err = new Error(
          data?.error?.message || `OpenAI request failed (${response.status})`,
        );
        err.code = data?.error?.code || 'OPENAI_API_ERROR';
        err.status = response.status;
        throw err;
      }

      const text = typeof data?.output_text === 'string' ? data.output_text.trim() : '';
      if (!text) {
        const err = new Error('OpenAI returned no text');
        err.code = 'AI_EMPTY_RESPONSE';
        err.status = 502;
        throw err;
      }

      return { text, provider: 'openai', model: process.env.OPENAI_MODEL || 'gpt-5.6' };
    },
  },
};

function json(res, status, body) {
  return res.status(status).json({ ...body });
}

export default async function handler(req, res) {
  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method === 'GET') {
    const providerName = String(process.env.AI_PROVIDER || 'openai').toLowerCase();
    const configured = providerName === 'openai' ? Boolean(process.env.OPENAI_API_KEY) : false;
    return json(res, 200, {
      ok: true,
      provider: providerName,
      configured,
      model: providerName === 'openai' ? (process.env.OPENAI_MODEL || 'gpt-5.6') : null,
    });
  }

  if (req.method !== 'POST') {
    return json(res, 405, { error: 'Method not allowed', code: 'METHOD_NOT_ALLOWED' });
  }

  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body) : (req.body || {});
    const providerName = String(process.env.AI_PROVIDER || 'openai').toLowerCase();
    const provider = PROVIDERS[providerName];

    if (!provider) {
      return json(res, 400, {
        error: `Unsupported AI provider: ${providerName}`,
        code: 'UNSUPPORTED_PROVIDER',
      });
    }

    const messages = Array.isArray(body.messages) ? body.messages : [];
    const cleanMessages = messages
      .filter((m) => m && (m.role === 'user' || m.role === 'assistant'))
      .slice(-12)
      .map((m) => ({ role: m.role, content: String(m.content || '').slice(0, 12000) }))
      .filter((m) => m.content.trim());

    if (!cleanMessages.length) {
      return json(res, 400, { error: 'No conversation messages supplied', code: 'EMPTY_INPUT' });
    }

    const result = await provider.generate({
      messages: cleanMessages,
      });

    return json(res, 200, result);
  } catch (error) {
    const status = Number.isInteger(error?.status) ? error.status : 500;
    return json(res, status, {
      error: error?.message || 'AI provider error',
      code: error?.code || 'AI_PROVIDER_ERROR',
    });
  }
}
