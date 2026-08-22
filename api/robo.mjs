/**
 * ROBO AIOS v2.13 — provider-neutral AI adapter.
 * Vercel serverless function: /api/robo
 *
 * Current provider: Gemini
 * Model: Gemini 3.1 Flash-Lite
 *
 * Supports text chat plus one optional camera image per request.
 * When an image is supplied, Gemini returns structured ROBO VISION data:
 * scene + detected objects + normalized bounding boxes.
 * The Gemini API key stays server-side in GEMINI_API_KEY.
 */

const SYSTEM_PROMPT =
  'You are Robo, a warm, concise AI companion. ' +
  'Respond naturally for spoken conversation. ' +
  'Keep answers reasonably short unless the user asks for detail. ' +
  'Do not mention being a language model unless directly asked. ' +
  'When a camera image is attached, inspect it carefully and use it to answer the user. ' +
  'Do not invent objects or details that are not visible. ' +
  'When vision data is requested, report only objects that are reasonably visible in the image. ' +
  'Bounding boxes are approximate and must be normalized to 0-1000 as [ymin, xmin, ymax, xmax].';

const VISION_SCHEMA = {
  type: 'object',
  properties: {
    scene: {
      type: 'string',
      description: 'A concise description of the visible scene.',
    },
    objects: {
      type: 'array',
      description: 'Visible objects detected in the image.',
      items: {
        type: 'object',
        properties: {
          name: {
            type: 'string',
            description: 'Short common name of the visible object.',
          },
          count: {
            type: 'integer',
            description: 'Number of instances of this object visible in the image.',
          },
          confidence: {
            type: 'integer',
            description: 'Approximate visual confidence from 0 to 100.',
          },
          box: {
            type: 'array',
            description: 'Approximate normalized bounding box [ymin, xmin, ymax, xmax], each value 0-1000.',
            items: { type: 'integer' },
          },
        },
        required: ['name', 'count', 'confidence', 'box'],
      },
    },
  },
  required: ['scene', 'objects'],
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
  if (data.startsWith('data:') && comma >= 0) data = data.slice(comma + 1);

  if (!data) {
    throw makeError('Vision image data is empty', 'INVALID_IMAGE', 400);
  }

  if (data.length > 12_000_000) {
    throw makeError('Vision image is too large', 'IMAGE_TOO_LARGE', 413);
  }

  return { mimeType, data };
}

function extractText(data) {
  return data?.candidates?.[0]?.content?.parts
    ?.map((part) => part?.text || '')
    .join('')
    .trim() || '';
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
      const visionMode = Boolean(normalizedImage);

      const contents = messages.map((message, index) => {
        const parts = [
          { text: String(message.content ?? '') },
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

      const generationConfig = visionMode
        ? {
            temperature: 0.2,
            maxOutputTokens: 700,
            response_mime_type: 'application/json',
            response_json_schema: VISION_SCHEMA,
          }
        : {
            temperature: 0.7,
            maxOutputTokens: 300,
          };

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
            generationConfig,
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
          'Gemini returned no text',
          'AI_EMPTY_RESPONSE',
          502,
        );
      }

      if (visionMode) {
        let visionData;

        try {
          visionData = JSON.parse(rawText);
        } catch {
          throw makeError(
            'Gemini returned invalid structured vision JSON',
            'VISION_INVALID_JSON',
            502,
          );
        }

        const objects = Array.isArray(visionData.objects)
          ? visionData.objects
              .map((object) => ({
                name: String(object?.name || 'unknown').trim(),
                count: Math.max(
                  1,
                  Number.parseInt(object?.count, 10) || 1,
                ),
                confidence: Math.max(
                  0,
                  Math.min(
                    100,
                    Number.parseInt(object?.confidence, 10) || 0,
                  ),
                ),
                box:
                  Array.isArray(object?.box) &&
                  object.box.length === 4
                    ? object.box.map((value) =>
                        Math.max(
                          0,
                          Math.min(
                            1000,
                            Number.parseInt(value, 10) || 0,
                          ),
                        ),
                      )
                    : [0, 0, 1000, 1000],
              }))
              .filter((object) => object.name)
          : [];

        const scene =
          String(visionData.scene || '').trim() ||
          'No clear scene description.';

        const reply = objects.length
          ? `I can see ${objects
              .map(
                (o) =>
                  `${o.count > 1 ? `${o.count} ` : ''}${o.name}`,
              )
              .join(', ')}.`
          : 'I do not see any clear objects I can identify confidently.';

        return {
          text: reply,
          provider: 'gemini',
          model,
          vision: true,
          visionData: {
            scene,
            objects,
          },
        };
      }

      return {
        text: rawText,
        provider: 'gemini',
        model,
        vision: false,
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
          ? process.env.GEMINI_MODEL ||
            'gemini-3.1-flash-lite'
          : null,
      vision: providerName === 'gemini',
      objectDetection: providerName === 'gemini',
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
    let body;

    try {
      body =
        typeof req.body === 'string'
          ? JSON.parse(req.body)
          : req.body || {};
    } catch {
      return json(res, 400, {
        error: 'Invalid JSON body',
        code: 'INVALID_JSON',
      });
    }

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
