// ROBO AIOS v2.11 — Vercel Node.js Function.
// This endpoint is intentionally optional during development.
// Without OPENAI_API_KEY it returns 503 so the browser can use Robo's local fallback.

module.exports = async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return res.status(503).json({
      error: 'AI backend is not configured yet.'
    });
  }

  try {
    let body = req.body || {};
    if (typeof body === 'string') {
      try { body = JSON.parse(body); }
      catch { return res.status(400).json({ error: 'Invalid JSON body.' }); }
    }

    const incoming = Array.isArray(body.messages) ? body.messages : [];
    const messages = incoming
      .filter(m => m && (m.role === 'user' || m.role === 'assistant') && typeof m.content === 'string')
      .slice(-8)
      .map(m => ({ role: m.role, content: m.content.slice(0, 2000) }));

    if (!messages.length || messages[messages.length - 1].role !== 'user') {
      return res.status(400).json({ error: 'A user message is required.' });
    }

    const model = process.env.OPENAI_MODEL || 'gpt-5.6';
    const input = [
      {
        role: 'developer',
        content: 'You are Robo, a concise, warm AI companion living inside a small floating emoji robot. Reply naturally in 1 to 3 short sentences. Be conversational and useful.'
      },
      ...messages
    ];

    const upstream = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({ model, input, max_output_tokens: 180 })
    });

    const data = await upstream.json().catch(() => ({}));
    if (!upstream.ok) {
      const message = data?.error?.message || `OpenAI returned HTTP ${upstream.status}`;
      console.error('ROBO upstream error:', upstream.status, message);
      return res.status(502).json({ error: message });
    }

    const reply = typeof data.output_text === 'string' ? data.output_text.trim() : '';
    if (!reply) return res.status(502).json({ error: 'AI returned no text.' });

    return res.status(200).json({ reply, model });
  } catch (err) {
    console.error('ROBO AI API error:', err);
    return res.status(500).json({ error: err?.message || 'AI backend request failed.' });
  }
};
