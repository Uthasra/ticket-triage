import { z } from "zod";

/**
 * Edge එකේදීම validate කරනවා. මෙතන fail වුන request එකක්
 * AI service එකට යන්නේ නෑ — ඒ කියන්නේ නාස්ති වෙන LLM call එකක් නෑ.
 */
export const CreateTicketSchema = z.object({
  text: z.string().min(10).max(5000),
  subject: z.string().max(200).optional(),
  customerEmail: z.string().email().optional(),
});

export type CreateTicketInput = z.infer<typeof CreateTicketSchema>;

export const ListQuerySchema = z.object({
  category: z
    .enum(["billing", "technical", "account", "feature_request", "complaint", "other"])
    .optional(),
  priority: z.enum(["low", "medium", "high", "urgent"]).optional(),
  page: z.coerce.number().int().min(1).default(1),
  limit: z.coerce.number().int().min(1).max(100).default(20),
});

/**
 * AI service එකේ response එකත් validate කරනවා.
 * Downstream service එකකුත් untrusted input එකක් — ඒක contract එක
 * මාරු කළොත්, මෙතන පැහැදිලි error එකක් ඕන, undefined fields
 * system එක පුරා පැතිරෙනවට වඩා.
 */
export const AnalysisSchema = z.object({
  category: z.string(),
  priority: z.string(),
  summary: z.string(),
  entities: z.array(z.string()),
  confidence: z.number(),
  reasoning: z.string(),
});

export const AnalyzeResponseSchema = z.object({
  analysis: AnalysisSchema,
  model: z.string(),
  cached: z.boolean(),
  latency_ms: z.number(),
});