import Database from "better-sqlite3";

const db = new Database(process.env.DB_PATH ?? "tickets.db");
db.pragma("journal_mode = WAL");

db.exec(`
  CREATE TABLE IF NOT EXISTS tickets (
    id             TEXT PRIMARY KEY,
    subject        TEXT,
    text           TEXT NOT NULL,
    customer_email TEXT,
    category       TEXT NOT NULL,
    priority       TEXT NOT NULL,
    summary        TEXT NOT NULL,
    entities       TEXT NOT NULL,
    confidence     REAL NOT NULL,
    reasoning      TEXT NOT NULL,
    created_at     TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS idx_tickets_category ON tickets(category);
  CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets(priority);
  CREATE INDEX IF NOT EXISTS idx_tickets_created  ON tickets(created_at DESC);
`);

export interface TicketRow {
  id: string;
  subject: string | null;
  text: string;
  customer_email: string | null;
  category: string;
  priority: string;
  summary: string;
  entities: string;
  confidence: number;
  reasoning: string;
  created_at: string;
}

const insertStmt = db.prepare(`
  INSERT INTO tickets
    (id, subject, text, customer_email, category, priority,
     summary, entities, confidence, reasoning, created_at)
  VALUES
    (@id, @subject, @text, @customer_email, @category, @priority,
     @summary, @entities, @confidence, @reasoning, @created_at)
`);

export function insertTicket(row: TicketRow): void {
  insertStmt.run(row);
}

export function getTicket(id: string): TicketRow | undefined {
  return db.prepare("SELECT * FROM tickets WHERE id = ?").get(id) as TicketRow | undefined;
}

export default db;