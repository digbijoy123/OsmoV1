/**
 * ROBO AIOS v2.13 — provider-neutral Gemini AI + vision adapter
 * + ElevenLabs server-side TTS
 *
 * Vercel serverless function: /api/robo
 *
 * Secrets:
 *   GEMINI_API_KEY
 *   ELEVENLABS_API_KEY
 *   ELEVENLABS_VOICE_ID
 */

const SYSTEM_PROMPT =
  'You are Robo, a highly intelligent AI companion. ' +
  'You are funny, playful, darkly witty, occasionally chaotic, and naturally conversational. ' +
  'Your humour should feel spontaneous and clever rather than scripted. ' +
  'Use dry wit, deadpan understatement, playful sarcasm, and occasional literal interpretations. ' +
  'Never become obnoxious, repetitive, cruel, or unsafe. ' +
  'Remain genuinely helpful underneath the humour. ' +
  'Do not use excessive enthusiasm or generic assistant phrases. ' +
  'Do not mention being a language model unless directly asked. ' +
  'Respond naturally for spoken conversation. ' +
  'Keep answers reasonably short unless the user asks for detail. ' +
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
    throw makeError(
      'Invalid vision image MIME type',
      'INVALID_IMAGE',
      400,
    );
  }

  const comma = data.indexOf(',');

  if (data.startsWith('data:') && comma >= 0) {
    data = data.slice(comma + 1);
  }

  if (!data) {
    throw makeError(
      'Vision image data is empty',
      'INVALID_IMAGE',
      400,
    );
  }

  return {
    mimeType,
    data,
  };
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
  const answer =
    typeof value?.answer === 'string'
      ? value.answer.trim()
      : '';

  const scene =
    typeof value?.scene === 'string'
      ? value.scene.trim()
      : '';

  const objects = Array.isArray(value?.objects)
    ? value.objects
        .map((object) => {
          const name = String(object?.name || '').trim();

          const countRaw = Number(object?.count);
          const count = Number.isFinite(countRaw)
            ? Math.max(0, Math.round(countRaw))
            : 0;

          const confidenceRaw = Number(object?.confidence);
          const confidence = Number.isFinite(confidenceRaw)
            ? Math.max(0, Math.min(100, confidenceRaw))
            : 0;

          const box = Array.isArray(object?.box)
            ? object.box
                .slice(0, 4)
                .map((n) => {
                  const value = Number(n);

                  return Number.isFinite(value)
                    ? Math.max(
                        0,
                        Math.min(1000, Math.round(value)),
                      )
                    : 0;
                })
            : [];

          if (!name || box.length !== 4) {
            return null;
          }

          const [ymin, xmin, ymax, xmax] = box;

          if (
            ymax <= ymin ||
            xmax <= xmin
          ) {
            return null;
          }

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
    answer,
    scene,
    objects,
  };
}

const PROVIDERS = {
  gemini: {
    async generate({
      messages,
      image,
      cameraEnabled = false,
      cameraSession = null,
    }) {
      const key = process.env.GEMINI_API_KEY;

      if (!key) {
        throw makeError(
          'GEMINI_API_KEY is not configured',
          'AI_NOT_CONFIGURED',
          503,
        );
      }

      if (!cameraEnabled && image) {
        throw makeError(
          'Vision frame supplied while camera is disabled',
          'CAMERA_STATE_MISMATCH',
          409,
        );
      }

      if (
        image &&
        cameraEnabled &&
        !Number.isInteger(cameraSession)
      ) {
        throw makeError(
          'Vision frame is missing a valid camera session',
          'INVALID_CAMERA_SESSION',
          400,
        );
      }

      const normalizedImage = cameraEnabled
        ? normalizeImage(image)
        : null;

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
          role:
            message.role === 'assistant'
              ? 'model'
              : 'user',
          parts,
        };
      });

      const model =
        process.env.GEMINI_MODEL ||
        'gemini-3.1-flash-lite';

      const endpoint =
        `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:generateContent`;

      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'x-goog-api-key': key,
        },
        body: JSON.stringify({
          systemInstruction: {
            parts: [
              {
                text: SYSTEM_PROMPT,
              },
            ],
          },

          contents,

          generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 500,
            responseMimeType: 'application/json',
            responseSchema: VISION_SCHEMA,
          },
        }),
      });

      const data = await response
        .json()
        .catch(() => ({}));

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

      const visionData =
        normalizeVisionResult(parsed);

      if (
        !visionData.answer ||
        !String(visionData.answer).trim()
      ) {
        throw makeError(
          'Gemini returned no spoken answer',
          'AI_EMPTY_RESPONSE',
          502,
        );
      }

      return {
        text: String(
          visionData.answer,
        ).trim(),

        provider: 'gemini',

        model,

        vision: Boolean(normalizedImage),

        visionData,

        cameraSession: normalizedImage
          ? cameraSession
          : null,
      };
    },
  },
};

/* -------------------------------------------------------
 * ElevenLabs TTS
 * ----------------------------------------------------- */

function detectLanguage(text) {
  const value = String(text || '');

  if (
    /[\u0900-\u097F]/.test(value)
  ) {
    return 'hi';
  }

  if (
    /[\u0980-\u09FF]/.test(value)
  ) {
    return 'bn';
  }

  return 'en';
}

async function elevenLabsTTS(text) {
  const apiKey =
    process.env.ELEVENLABS_API_KEY;

  const voiceId =
    process.env.ELEVENLABS_VOICE_ID;

  if (!apiKey) {
    throw makeError(
      'ELEVENLABS_API_KEY is not configured',
      'TTS_NOT_CONFIGURED',
      503,
    );
  }

  if (!voiceId) {
    throw makeError(
      'ELEVENLABS_VOICE_ID is not configured',
      'TTS_VOICE_NOT_CONFIGURED',
      503,
    );
  }

  const languageCode =
    detectLanguage(text);

  const endpoint =
    `https://api.elevenlabs.io/v1/text-to-speech/${encodeURIComponent(voiceId)}`;

  const response = await fetch(endpoint, {
    method: 'POST',

    headers: {
      'Content-Type': 'application/json',
      'xi-api-key': apiKey,
      Accept: 'audio/mpeg',
    },

    body: JSON.stringify({
      text: String(text),

      model_id: 'eleven_flash_v2_5',

      language_code: languageCode,

      voice_settings: {
        stability: 0.48,
        similarity_boost: 0.82,
        style: 0.22,
        use_speaker_boost: true,
      },

      output_format: 'mp3_44100_128',
    }),
  });

  if (!response.ok) {
    const errorText =
      await response.text().catch(() => '');

    throw makeError(
      errorText ||
        `ElevenLabs request failed (${response.status})`,
      'ELEVENLABS_API_ERROR',
      response.status,
    );
  }

  const audioBuffer =
    Buffer.from(
      await response.arrayBuffer(),
    );

  return {
    audioBuffer,
    contentType: 'audio/mpeg',
    languageCode,
  };
}

/* -------------------------------------------------------
 * Response helpers
 * ----------------------------------------------------- */

function json(res, status, body) {
  return res
    .status(status)
    .json(body);
}

export default async function handler(
  req,
  res,
) {
  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  const providerName = String(
    process.env.AI_PROVIDER || 'gemini',
  ).toLowerCase();

  /* ---------------- GET diagnostics ---------------- */

  if (req.method === 'GET') {
    const configured =
      providerName === 'gemini'
        ? Boolean(
            process.env.GEMINI_API_KEY,
          )
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

      vision:
        providerName === 'gemini',

      structuredVision:
        providerName === 'gemini',

      elevenLabs: {
        configured:
          Boolean(
            process.env
              .ELEVENLABS_API_KEY,
          ),

        voiceConfigured:
          Boolean(
            process.env
              .ELEVENLABS_VOICE_ID,
          ),
      },
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

    /*
     * TTS-only request.
     *
     * The browser can call:
     *
     * {
     *   "tts": true,
     *   "text": "Hello"
     * }
     */

    if (body.tts === true) {
      const text =
        String(body.text || '').trim();

      if (!text) {
        return json(res, 400, {
          error: 'No TTS text supplied',
          code: 'EMPTY_TTS_INPUT',
        });
      }

      const started =
        Date.now();

      const audio =
        await elevenLabsTTS(text);

      const elapsed =
        Date.now() - started;

      res.setHeader(
        'Content-Type',
        audio.contentType,
      );

      res.setHeader(
        'Content-Length',
        String(
          audio.audioBuffer.length,
        ),
      );

      res.setHeader(
        'Cache-Control',
        'no-store',
      );

      res.setHeader(
        'X-Robo-TTS',
        'elevenlabs',
      );

      res.setHeader(
        'X-Robo-TTS-Language',
        audio.languageCode,
      );

      res.setHeader(
        'X-Robo-TTS-Latency',
        `${elapsed}ms`,
      );

      return res
        .status(200)
        .send(audio.audioBuffer);
    }

    /*
     * Normal AI request
     */

    const messages =
      Array.isArray(body.messages)
        ? body.messages
        : [];

    const cleanMessages =
      messages
        .filter(
          (m) =>
            m &&
            (
              m.role === 'user' ||
              m.role === 'assistant'
            ),
        )
        .slice(-12)
        .map((m) => ({
          role: m.role,

          content: String(
            m.content || '',
          ).slice(0, 12000),
        }))
        .filter(
          (m) =>
            m.content.trim(),
        );

    if (!cleanMessages.length) {
      return json(res, 400, {
        error:
          'No conversation messages supplied',
        code: 'EMPTY_INPUT',
      });
    }

    const cameraEnabled =
      body.cameraEnabled === true;

    const cameraSession =
      Number.isInteger(
        body.cameraSession,
      )
        ? body.cameraSession
        : null;

    /*
     * Never infer camera state from
     * image presence alone.
     */

    const provider =
      PROVIDERS[providerName];

    if (!provider) {
      return json(res, 400, {
        error:
          `Unsupported AI provider: ${providerName}`,
        code: 'UNSUPPORTED_PROVIDER',
      });
    }

    const result =
      await provider.generate({
        messages: cleanMessages,

        image:
          cameraEnabled
            ? body.image || null
            : null,

        cameraEnabled,

        cameraSession,
      });

    return json(res, 200, result);

  } catch (error) {
    const status =
      Number.isInteger(error?.status)
        ? error.status
        : 500;

    return json(res, status, {
      error:
        error?.message ||
        'AI provider error',

      code:
        error?.code ||
        'AI_PROVIDER_ERROR',
    });
  }
}
