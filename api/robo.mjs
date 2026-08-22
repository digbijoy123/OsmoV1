/**
 * ROBO AIOS v2.14 — Gemini AI + optional structured vision.
 * Vercel serverless function: /api/robo
 *
 * Current provider: Gemini
 * Model: Gemini 3.1 Flash-Lite
 *
 * IMPORTANT:
 * - Normal text/voice requests use a normal text response.
 * - Vision requests use structured JSON.
 * - Camera images are never required for ordinary conversation.
 * - GEMINI_API_KEY stays server-side.
 */

const SYSTEM_PROMPT =
  'You are Robo, a warm, concise AI companion. ' +
  'Respond naturally for spoken conversation. ' +
  'Keep answers reasonably short unless the user asks for detail. ' +
  'Do not mention being a language model unless directly asked. ' +
  'You have two separate abilities: hearing the user through speech recognition and seeing camera images when an image is provided. ' +
  'Never confuse hearing with seeing. ' +
  'If the user asks "can you hear me", answer about hearing. ' +
  'If the user asks whether you can see them, answer about the camera image. ' +
  'Only describe the camera image when one is actually attached. ' +
  'Never invent visual details.';

const VISION_SCHEMA = {
  type: 'OBJECT',
  properties: {
    answer: {
      type: 'STRING',
      description:
        'The natural spoken answer to the user’s latest question.',
    },

    scene: {
      type: 'STRING',
      description:
        'A brief factual description of the visible camera scene.',
    },

    objects: {
      type: 'ARRAY',
      description:
        'Prominent confidently visible objects in the camera image.',
      items: {
        type: 'OBJECT',
        properties: {
          name: {
            type: 'STRING',
            description: 'Concise object name.',
          },

          count: {
            type: 'INTEGER',
            description: 'Number of instances represented.',
          },

          confidence: {
            type: 'NUMBER',
            description: 'Confidence estimate from 0 to 100.',
          },

          box: {
            type: 'ARRAY',
            description:
              'Bounding box [ymin, xmin, ymax, xmax], normalized 0-1000.',
            minItems: 4,
            maxItems: 4,
            items: {
              type: 'INTEGER',
            },
          },
        },

        required: [
          'name',
          'count',
          'confidence',
          'box',
        ],
      },
    },
  },

  required: [
    'answer',
    'scene',
    'objects',
  ],
};


function makeError(message, code, status) {
  const err = new Error(message);
  err.code = code;
  err.status = status;
  return err;
}


function normalizeImage(image) {
  if (!image || typeof image !== 'object') {
    return null;
  }

  const mimeType =
    String(image.mimeType || '').toLowerCase();

  let data =
    String(image.data || '').trim();

  if (!mimeType.startsWith('image/')) {
    throw makeError(
      'Invalid vision image MIME type',
      'INVALID_IMAGE',
      400,
    );
  }

  const comma = data.indexOf(',');

  if (
    data.startsWith('data:') &&
    comma >= 0
  ) {
    data = data.slice(comma + 1);
  }

  if (!data) {
    throw makeError(
      'Vision image data is empty',
      'INVALID_IMAGE',
      400,
    );
  }

  if (data.length > 12_000_000) {
    throw makeError(
      'Vision image is too large',
      'IMAGE_TOO_LARGE',
      413,
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
  const scene =
    typeof value?.scene === 'string' &&
    value.scene.trim()
      ? value.scene.trim()
      : 'No scene description available.';

  const objects =
    Array.isArray(value?.objects)
      ? value.objects
          .map((object) => {

            const name =
              String(
                object?.name || '',
              ).trim();

            if (!name) {
              return null;
            }

            const count =
              Math.max(
                1,
                Number.parseInt(
                  object?.count,
                  10,
                ) || 1,
              );

            const confidenceRaw =
              Number(
                object?.confidence,
              );

            const confidence =
              Number.isFinite(
                confidenceRaw,
              )
                ? Math.max(
                    0,
                    Math.min(
                      100,
                      confidenceRaw,
                    ),
                  )
                : 0;

            const box =
              Array.isArray(object?.box)
                ? object.box
                    .slice(0, 4)
                    .map((n) =>
                      Math.max(
                        0,
                        Math.min(
                          1000,
                          Number.parseInt(
                            n,
                            10,
                          ) || 0,
                        ),
                      ),
                    )
                : [];

            if (box.length !== 4) {
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
    scene,
    objects,
  };
}


const PROVIDERS = {

  gemini: {

    async generate({
      messages,
      image,
    }) {

      const key =
        process.env.GEMINI_API_KEY;

      if (!key) {
        throw makeError(
          'GEMINI_API_KEY is not configured',
          'AI_NOT_CONFIGURED',
          503,
        );
      }

      const normalizedImage =
        normalizeImage(image);

      const isVision =
        Boolean(normalizedImage);


      /*
       * Build Gemini conversation.
       *
       * Only the latest user message receives
       * the camera frame.
       */
      const contents =
        messages.map(
          (message, index) => {

            const parts = [
              {
                text:
                  String(
                    message.content ?? '',
                  ),
              },
            ];

            if (
              isVision &&
              message.role === 'user' &&
              index === messages.length - 1
            ) {

              parts.push({
                inline_data: {
                  mime_type:
                    normalizedImage.mimeType,

                  data:
                    normalizedImage.data,
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
          },
        );


      const model =
        process.env.GEMINI_MODEL ||
        'gemini-3.1-flash-lite';


      /*
       * IMPORTANT:
       *
       * Normal conversation does NOT use
       * responseSchema.
       *
       * Vision requests DO use structured output.
       *
       * This prevents ordinary voice requests
       * from failing with AI_EMPTY_RESPONSE.
       */

      const generationConfig =
        isVision
          ? {
              temperature: 0.4,
              maxOutputTokens: 500,

              responseMimeType:
                'application/json',

              responseSchema:
                VISION_SCHEMA,
            }
          : {
              temperature: 0.7,
              maxOutputTokens: 300,
            };


      const response =
        await fetch(
          `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
          {
            method: 'POST',

            headers: {
              'Content-Type':
                'application/json',

              'x-goog-api-key':
                key,
            },

            body: JSON.stringify({

              systemInstruction: {
                parts: [
                  {
                    text:
                      SYSTEM_PROMPT,
                  },
                ],
              },

              contents,

              generationConfig,

            }),
          },
        );


      const data =
        await response
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


      const rawText =
        extractText(data);


      if (!rawText) {

        throw makeError(
          isVision
            ? 'Gemini returned no structured vision response'
            : 'Gemini returned no text response',

          'AI_EMPTY_RESPONSE',

          502,
        );
      }


      /*
       * NORMAL TEXT RESPONSE
       */
      if (!isVision) {

        return {
          text: rawText,

          provider:
            'gemini',

          model,

          vision:
            false,

          visionData:
            null,
        };
      }


      /*
       * STRUCTURED VISION RESPONSE
       */
      let parsed;

      try {

        parsed =
          JSON.parse(rawText);

      } catch {

        throw makeError(
          'Gemini returned invalid vision JSON',

          'AI_INVALID_JSON',

          502,
        );
      }


      const visionData =
        normalizeVisionResult(
          parsed,
        );


      const text =
        String(
          parsed?.answer || '',
        ).trim();


      if (!text) {

        throw makeError(
          'Gemini returned no spoken answer',

          'AI_EMPTY_RESPONSE',

          502,
        );
      }


      return {

        text,

        provider:
          'gemini',

        model,

        vision:
          true,

        visionData,

      };
    },
  },
};


function json(
  res,
  status,
  body,
) {
  return res
    .status(status)
    .json(body);
}


export default async function handler(
  req,
  res,
) {

  if (req.method === 'OPTIONS') {
    return res
      .status(204)
      .end();
  }


  const providerName =
    String(
      process.env.AI_PROVIDER ||
        'gemini',
    ).toLowerCase();


  /*
   * GET /api/robo
   *
   * Used by the developer diagnostics page.
   */
  if (req.method === 'GET') {

    const configured =
      providerName === 'gemini'
        ? Boolean(
            process.env.GEMINI_API_KEY,
          )
        : false;


    return json(
      res,
      200,
      {
        ok: true,

        provider:
          providerName,

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
      },
    );
  }


  if (req.method !== 'POST') {

    return json(
      res,
      405,
      {
        error:
          'Method not allowed',

        code:
          'METHOD_NOT_ALLOWED',
      },
    );
  }


  try {

    const body =
      typeof req.body === 'string'
        ? JSON.parse(req.body)
        : req.body || {};


    const provider =
      PROVIDERS[providerName];


    if (!provider) {

      return json(
        res,
        400,
        {
          error:
            `Unsupported AI provider: ${providerName}`,

          code:
            'UNSUPPORTED_PROVIDER',
        },
      );
    }


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

        .map(
          (m) => ({
            role:
              m.role,

            content:
              String(
                m.content || '',
              ).slice(
                0,
                12000,
              ),
          }),
        )

        .filter(
          (m) =>
            m.content.trim(),
        );


    if (!cleanMessages.length) {

      return json(
        res,
        400,
        {
          error:
            'No conversation messages supplied',

          code:
            'EMPTY_INPUT',
        },
      );
    }


    /*
     * Vision is activated ONLY when the frontend
     * explicitly sends body.image.
     *
     * If camera is OFF, body.image should be null
     * and this backend performs ordinary text chat.
     */
    const result =
      await provider.generate({
        messages:
          cleanMessages,

        image:
          body.image || null,
      });


    return json(
      res,
      200,
      result,
    );


  } catch (error) {

    const status =
      Number.isInteger(
        error?.status,
      )
        ? error.status
        : 500;


    return json(
      res,
      status,
      {
        error:
          error?.message ||
          'AI provider error',

        code:
          error?.code ||
          'AI_PROVIDER_ERROR',
      },
    );
  }
}
