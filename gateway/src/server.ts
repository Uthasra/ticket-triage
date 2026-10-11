import express, { type NextFunction, type Request, type Response } from "express";
import { randomUUID } from "node:crypto";

import { AiServiceError, analyzeTicket } from "./aiClient.js";
import { getTicket, insertTicket, type TicketRow } from "./db.js";
import { CreateTicketSchema } from "./schemas.js";

const app = express();
app.use(express.json({ limit: "1mb" }));

const PORT = Number(process.env.PORT ?? 3000);

app.get("/health", (_req, res) => {
  res.json({ status: "ok", uptime: process.uptime() });
});

app.post("/tickets", async (req, res, next) => {
  const parsed = CreateTicketSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({
      error: "validation_failed",
      details: parsed.error.issues.map((i) => ({
        field: i.path.join("."),
        message: i.message,
      })),
    });
  }

  const { text, subject, customerEmail } = parsed.data;

  try {
    const result = await analyzeTicket(text, subject);
    const a = result.analysis;

    const row: TicketRow = {
      id: randomUUID(),
      subject: subject ?? null,
      text,
      customer_email: customerEmail ?? null,
      category: a.category,
      priority: a.priority,
      summary: a.summary,
      entities: JSON.stringify(a.entities),
      confidence: a.confidence,
      reasoning: a.reasoning,
      created_at: new Date().toISOString(),
    };

    insertTicket(row);

    res.status(201).json({
      id: row.id,
      category: a.category,
      priority: a.priority,
      summary: a.summary,
      entities: a.entities,
      confidence: a.confidence,
      reasoning: a.reasoning,
      createdAt: row.created_at,
      meta: { cached: result.cached, latencyMs: result.latency_ms, model: result.model },
    });
  } catch (err) {
    next(err);   // error handler එකට යවනවා
  }
});

app.get("/tickets/:id", (req, res) => {
  const row = getTicket(req.params.id);
  if (!row) {
    return res.status(404).json({ error: "not_found", message: "No ticket with that id" });
  }
  res.json({ ...row, entities: JSON.parse(row.entities) });
});

/* ---------------------------------------------------- error handler */

// එක තැනක් — හැම error එකක්ම හරි status code එකකට map කරනවා.
// මේක නැත්නම් dependency failure එකයි අපේ bug එකයි එකම විදිහට පේනවා.
app.use((err: Error, _req: Request, res: Response, _next: NextFunction) => {
  if (err instanceof AiServiceError) {
    const status =
      err.kind === "timeout" ? 504 :
      err.kind === "unavailable" ? 503 :
      502;
    console.error(`[ai-service:${err.kind}]`, err.message);
    return res.status(status).json({
      error: "analysis_failed",
      message: "Ticket could not be analysed right now. Please retry shortly.",
      kind: err.kind,
    });
  }

  console.error("[unhandled]", err);
  res.status(500).json({ error: "internal_error", message: "Something went wrong" });
});

app.listen(PORT, () => {
  console.log(`Gateway listening on :${PORT}`);
});

export default app;