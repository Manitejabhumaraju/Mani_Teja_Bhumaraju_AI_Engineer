import express from 'express';
import cors from 'cors';
import Database from 'better-sqlite3';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DB_PATH = path.resolve(__dirname, '../data/loadshare.db');
const PORT = 3002;

const app = express();
app.use(cors());
app.use(express.json());

let db;
try {
  db = new Database(DB_PATH, { readonly: true });
  console.log(`✅ Database connected: ${DB_PATH}`);
} catch (err) {
  console.error(`❌ Failed to open database: ${err.message}`);
  process.exit(1);
}

// REST API endpoints
app.get('/tables', (req, res) => {
  const tables = db.prepare(
    "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
  ).all();
  res.json({ tables: tables.map(t => t.name) });
});

app.get('/schema/:table', (req, res) => {
  const info = db.prepare(`PRAGMA table_info(${req.params.table})`).all();
  res.json({ 
    columns: info.map(col => ({ name: col.name, type: col.type }))
  });
});

app.post('/query', (req, res) => {
  try {
    const sql = req.body.sql?.trim();
    
    if (!sql) {
      return res.status(400).json({ error: 'Missing sql parameter' });
    }
    
    if (!sql.toUpperCase().startsWith('SELECT')) {
      return res.status(400).json({ error: 'Only SELECT queries allowed' });
    }
    
    if (/\b(DROP|DELETE|INSERT|UPDATE|ALTER|CREATE)\b/i.test(sql)) {
      return res.status(400).json({ error: 'DML/DDL not allowed' });
    }

    const rows = db.prepare(sql).all();
    res.json({ rows, count: rows.length });
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', db: DB_PATH });
});

app.listen(PORT, () => {
  console.log(`🚀 SQLite MCP Server (HTTP) running on http://localhost:${PORT}`);
  console.log(`   Database: ${DB_PATH}`);
  console.log(`   `);
  console.log(`   Available endpoints:`);
  console.log(`     GET  /health - Health check`);
  console.log(`     GET  /tables - List all tables`);
  console.log(`     GET  /schema/:table - Get table schema`);
  console.log(`     POST /query - Execute SELECT query`);
  console.log(`   `);
  console.log(`   ✅ Ready for Python backend connection`);
});

process.on('SIGINT', () => {
  db.close();
  console.log('\n👋 SQLite MCP server stopped');
  process.exit(0);
});
