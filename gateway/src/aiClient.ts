import { AnalyzeResponseSchema } from "./schemas.js";

const AI_SERVICE_URL = process.env.AI_SERVICE_URL ?? "http://localhost:8001";
const TIMEOUT_MS = Number(process.env.AI_TIMEOUT_MS ?? 60000);

/**
 * Typed error — route handler එකට හරි HTTP status එක
 * තෝරගන්න පුළුවන් වෙනවා, හැමදේටම 500 දෙනවට වඩා.
 */
export class AiServiceError extends Error {
  constructor(
    message: string,
    public readonly kind: "timeout" | "unavailable" | "bad_response" | "upstream",
  ) {
    super(message);
    this.name = "AiServiceError";
  }
}

export async function analyzeTicket(text: string, subject?: string) {
  // AbortController එක — slow dependency එකක් අපේ request එක
  // සදාකාලිකව එල්ලගෙන ඉන්න දෙන්නේ නෑ.
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const res = await fetch(`${AI_SERVICE_URL}/analyze`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ text, subject }),
      signal: controller.signal,
    });

    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new AiServiceError(
        `AI service returned ${res.status}: ${body.slice(0, 200)}`,
        res.status >= 500 ? "upstream" : "bad_response",
      );
    }

    const json = await res.json();
    const parsed = AnalyzeResponseSchema.safeParse(json);
    if (!parsed.success) {
      throw new AiServiceError(
        `AI service response did not match the expected shape: ${parsed.error.message}`,
        "bad_response",
      );
    }
    return parsed.data;
  } catch (err) {
    if (err instanceof AiServiceError) throw err;
    if (err instanceof Error && err.name === "AbortError") {
      throw new AiServiceError(`AI service timed out after ${TIMEOUT_MS}ms`, "timeout");
    }
    throw new AiServiceError(
      `Could not reach AI service: ${(err as Error).message}`,
      "unavailable",
    );
  } finally {
    clearTimeout(timer);
  }
}