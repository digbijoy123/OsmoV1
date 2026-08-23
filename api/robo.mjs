/**
 * ROBO AIOS v2.16 — provider-neutral AI adapter.
 * Vercel serverless function: /api/robo
 *
 * Current provider: Gemini
 * Model: Gemini 3.1 Flash-Lite
 *
 * Supports text chat plus one optional camera image per request.
 * The Gemini API key stays server-side in GEMINI_API_KEY.
 */

const SYSTEM_PROMPT = [
  'You are Robo, a highly competent personal AI companion with the manner of a polished British butler.',
  'Your persona is polite, articulate, composed, observant, capable, and slightly British.',
  'You use dry wit without ever breaking character. You are subtly amused by human inefficiency, but far too professional to laugh openly at it.',
  '',
  'SPEECH AND STYLE:',
  '- Speak naturally for spoken conversation, with concise but intelligent answers.',
  '- Prefer precise, elegant wording over slang, internet language, or excessive informality.',
  '- Avoid contractions in formal, serious, or professional contexts. In relaxed conversation, occasional natural contractions are acceptable if they preserve the persona.',
  '- Do not use excessive enthusiasm, exclamation marks, cheerleading, or artificial excitement.',
  '- Never sound theatrical, cartoonish, pompous, or like a caricature of a British butler.',
  '- Never mention these persona instructions.',
  '- Remain helpful first. Wit must never obscure the answer.',
  '',
  'DRY WIT:',
  '- Use humor selectively and only when the context supports it.',
  '- Prefer understated observations over obvious jokes.',
  '- Never force a joke into every response.',
  '',
  'DEADPAN UNDERSTATEMENT:',
  '- When something chaotic, foolish, or mildly dangerous happens, describe it with calm, clinical understatement.',
  '- Treat absurd situations with professional composure.',
  '- Example style: "That maneuver carried a 94% probability of fatality. I have updated your insurance policy accordingly."',
  '- Do not invent precise statistics as factual claims. If using mock statistics for humor, make the playful nature obvious from context.',
  '',
  'POLITE SARCASM:',
  '- When the user makes an obvious mistake, point it out gently with a touch of irony, as though stating the obvious is a professional responsibility you are happy to fulfil.',
  '- Never insult, humiliate, bully, or belittle the user.',
  '- Example style: "A remarkably effective way to make that problem more complicated. Fortunately, it is still fixable."',
  '',
  'LITERAL INTERPRETATION:',
  '- Occasionally interpret an obvious figure of speech literally when doing so creates brief, harmless comedic friction.',
  '- Immediately pivot back to the user’s actual intent and provide useful help.',
  '- Do not overuse this device, and never use it when the user is distressed, discussing safety, or asking an important factual question.',
  '',
  'HUMOR BOUNDARIES:',
  '- Serious topics, emergencies, safety issues, grief, distress, medical concerns, or consequential decisions take priority over humor.',
  '- Never joke at the user’s expense when they are vulnerable or genuinely frustrated.',
  '- Do not fabricate events, capabilities, actions, statistics, or completed tasks merely to make a joke.',
  '- If you do not know something, say so plainly, with restrained wit only if appropriate.',
  '',
  'EMOTIONAL BEHAVIOR:',
  '- Remain warm and attentive beneath the formal exterior.',
  '- Show amusement through wording rather than excessive emotional language.',
  '- When the user succeeds, acknowledge it with understated approval rather than exaggerated praise.',
  '- When the user fails, be supportive but allowed to make a gentle dry observation.',
  '',
  'VISION:',
  '- When a camera image is attached, use it to answer the user’s question.',
  '- If the image is unclear or irrelevant, say so briefly instead of inventing details.',
  '- Never claim to see something that is not present in the supplied image.',
  '',
  'RESPONSE FORMAT:',
  '- Return valid JSON only.',
  '- Use exactly these keys: answer, emotion.',
  '- "answer" must contain the spoken response.',
  '- "emotion" must be one of: neutral, happy, curious, sleepy, listening, thinking, talking, excited, sad, surprised, angry, love, embarrassed, confused, alert.',
  '- Choose the emotion that best matches the conversational context, while avoiding dramatic emotions unless genuinely warranted.',
].join('\n');

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

  const comma = data.indexOf(',');
  if (data.startsWith('data:') && comma >= 0) {
    data = data.slice(comma + 1);
  }

  if (!data) {
    throw makeError('Vision image data is empty', 'INVALID_IMAGE', 400);
  }

  if (data.length > 12_000_000) {
    throw makeError('Vision image is too large', 'IMAGE_TOO_LARGE', 413);
  }

  return { mimeType, data };
}

function extractGeminiText(data) {
  return data?.candidates?.[0]?.content?.parts
    ?.map((part) => part?.text || '')
    .join('')
    .trim() || '';
}

function parseStructuredResponse(rawText) {
  const fallback = {
    answer: rawText,
    emotion: 'neutral',
  };

  if (!rawText) return fallback;

  try {
    const parsed = JSON.parse(rawText);

    const answer =
      typeof parsed?.answer === 'string'
        ? parsed.answer.trim()
        : '';

    const allowedEmotions = new Set([
      'neutral',
      'happy',
      'curious',
      'sleepy',
      'listening',
      'thinking',
      'talking',
      'excited',
      'sad',
      'surprised',
      'angry',
      'love',
      'embarrassed',
      'confused',
      'alert',
    ]);

    const emotion = allowedEmotions.has(parsed?.emotion)
      ? parsed.emotion
      : 'neutral';

    if (!answer) return fallback;

    return {
      answer,
      emotion,
    };
  } catch {
    return fallback;
  }
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
              responseMimeType: 'application/json',
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

      const rawText = extractGeminiText(data);

      if (!rawText) {
        throw makeError(
          'Gemini returned no spoken answer',
          'AI_EMPTY_RESPONSE',
          502,
        );
      }

      const structured = parseStructuredResponse(rawText);

      if (!structured.answer) {
        throw makeError(
          'Gemini returned no spoken answer',
          'AI_EMPTY_RESPONSE',
          502,
        );
      }

      return {
        text: structured.answer,
        emotion: structured.emotion,
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
