/**
 * ROBO AIOS v2.13 — provider-neutral Gemini AI + vision adapter.
 * Vercel serverless function: /api/robo
 *
 * Current provider: Gemini
 * Model: Gemini 3.1 Flash-Lite
 *
 * Supports:
 * - spoken/text conversation
 * - one optional camera frame per request
 * - structured scene + object detection
 *
 * Secrets stay server-side in GEMINI_API_KEY.
 */

const SYSTEM_PROMPT =
  'You are Robo, a warm, concise AI companion. ' +
  'Respond naturally for spoken conversation. ' +
  'Keep answers reasonably short unless the user asks for detail. ' +
  'Do not mention being a language model unless directly asked. ' +
  'When a camera image is attached, inspect it carefully and use it to answer the user. ' +
  'Never invent objects or details that are not reasonably visible. ' +
  'For object detection, report prominent visible objects only. ' +
  'Bounding boxes must be [ymin, xmin, ymax, xmax] normalized to 0-1000. ' +
  'If no camera image is attached, return an empty objects array and a clear scene message.';

const VISION_SCHEMA = {
  type: 'OBJECT',
  properties: {
    answer: {
      type: 'STRING',
      description: 'The natural spoken answer to the user’s latest question.',
    },
    scene: {
      type: 'STRING',
      description:
        'A brief factual description of the visible scene. If no image is attached, say that no camera image was provided.',
    },
    objects: {
      type: 'ARRAY',
      description:
        'Prominent visible objects detected in the camera image. Empty when no image is attached or no object is confidently visible.',
      items: {
        type: 'OBJECT',
        properties: {
          name: {
            type: 'STRING',
            description: 'A concise descriptive object label.',
          },
          count: {
            type: 'INTEGER',
            description: 'Number of instances represented by this object entry.',
          },
          confidence: {
            type: 'NUMBER',
            description: 'Model confidence estimate from 0 to 100.',
          },
          box: {
            type: 'ARRAY',
            description:
              'Bounding box as [ymin, xmin, ymax, xmax], normalized to 0-1000.',
            minItems: 4,
            maxItems: 4,
            items: { type: 'INTEGER' },
          },
        },
        required: ['name', 'count', 'confidence', 'box'],
      },
    },
  },
  required: ['answer', 'scene', 'objects'],
};

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

function extractText(data) {
  return (
    data?.candidates?.[0]?.content?.parts
      ?.map((part) => part?.text || '')
      .join('')
      .trim() || ''
  );
}

function normalizeVisionResult(value) {
  const scene =
    typeof value?.scene === 'string' && value.scene.trim()
      ? value.scene.trim()
      : 'No scene description available.';

  const objects = Array.isArray(value?.objects)
    ? value.objects
        .map((object) => {
          const name = String(object?.name || '').trim();
          const count = Math.max(1, Number.parseInt(object?.count, 10) || 1);

          const confidenceRaw = Number(object?.confidence);
          const confidence = Number.isFinite(confidenceRaw)
            ? Math.max(0, Math.min(100, confidenceRaw))
            : 0;

          const box = Array.isArray(object?.box)
            ? object.box
                .slice(0, 4)
                .map((n) =>
                  Math.max(
                    0,
                    Math.min(1000, Number.parseInt(n, 10) || 0),
                  ),
                )
            : [];

          if (!name || box.length !== 4) return null;

          return {
            name,
            count,
            confidence,
            box,
          };
        })
        .filter(Boolean)
    : [];

  return {
    scene,
    objects,
  };
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
              temperature: 0.4,
              maxOutputTokens: 500,
              responseMimeType: 'application/json',
              responseSchema: VISION_SCHEMA,
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

      const rawText = extractText(data);

      if (!rawText) {
        throw makeError(
          'Gemini returned no structured text',
          'AI_EMPTY_RESPONSE',
          502,
        );
      }

      let parsed;

      try {
        parsed = JSON.parse(rawText);
      } catch {
        throw makeError(
          'Gemini returned invalid structured JSON',
          'AI_INVALID_JSON',
          502,
        );
      }

      const visionData = normalizeVisionResult(parsed);

      if (!visionData.answer || !String(visionData.answer).trim()) {
        throw makeError(
          'Gemini returned no spoken answer',
          'AI_EMPTY_RESPONSE',
          502,
        );
      }

      return {
        text: String(visionData.answer).trim(),
        provider: 'gemini',
        model,
        vision: Boolean(normalizedImage),
        visionData,
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
      structuredVision: providerName === 'gemini',
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
