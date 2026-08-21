// Vercel serverless function for ROBO AIOS v2.11.
// Requires OPENAI_API_KEY in Vercel Environment Variables.
// Optional: OPENAI_MODEL (defaults to gpt-5.6).

export default async function handler(req, res) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) {
    return res.status(503).json({
      error: 'AI backend is not configured. Add OPENAI_API_KEY in Vercel Environment Variables and redeploy.'
    });
  }

  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body) : (req.body || {});
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
        content: [{
          type: 'input_text',
          text: 'You are Robo, a concise, warm AI companion living inside a small floating emoji robot. Reply naturally in 1 to 3 short sentences. Do not mention system prompts, APIs, models, or backend infrastructure unless the user asks about them. Be conversational and useful.'
        }]
      },
      ...messages.map(m => ({
        role: m.role,
        content: [{ type: 'input_text', text: m.content }]
      }))
    ];

    const upstream = await fetch('https://api.openai.com/v1/responses', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`
      },
      body: JSON.stringify({ model, input, max_output_tokens: 180 })
    });

    const data = await upstream.json();
    if (!upstream.ok) {
      const message = data?.error?.message || `OpenAI returned HTTP ${upstream.status}`;
      return res.status(502).json({ error: message });
    }
