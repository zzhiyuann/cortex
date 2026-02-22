#!/usr/bin/env node
/**
 * Claude Code SDK sidecar — persistent Node.js HTTP service that wraps
 * the Claude Agent SDK for fast programmatic access (~3s vs ~7s CLI).
 *
 * Dispatcher sends HTTP POST to localhost:18899/query with JSON body:
 *   { prompt, sessionId?, resume?, maxTurns?, model? }
 *
 * Returns JSON:
 *   { result, sessionId, elapsed }
 *
 * The Node.js process stays warm, so there's no cold-start overhead
 * after the first request. Full tool access and session continuity.
 */

import { createServer } from "node:http";
import { query } from "@anthropic-ai/claude-agent-sdk";

const PORT = parseInt(process.env.SIDECAR_PORT || "18899", 10);

async function handleQuery(body) {
  const { prompt, sessionId, resume, maxTurns, model, cwd } = body;

  const opts = {
    maxTurns: maxTurns || 1,
    dangerouslySkipPermissions: true,
  };
  if (sessionId && resume) {
    opts.continue = true;
    opts.sessionId = sessionId;
  } else if (sessionId) {
    opts.sessionId = sessionId;
  }
  if (model) opts.model = model;

  const t0 = Date.now();
  const session = await query({
    prompt,
    options: opts,
    cwd: cwd || process.env.HOME,
  });

  let result = "";
  let sid = null;
  for await (const msg of session) {
    if (msg.session_id && !sid) sid = msg.session_id;
    if (msg.type === "result") {
      result = msg.result || result;
      break;
    }
    if (msg.type === "assistant") {
      for (const b of msg.message?.content || []) {
        if (b.type === "text") result = b.text;
      }
    }
  }

  return {
    result: result.trim(),
    sessionId: sid,
    elapsed: Date.now() - t0,
  };
}

const server = createServer(async (req, res) => {
  if (req.method !== "POST" || req.url !== "/query") {
    res.writeHead(404);
    res.end("Not found");
    return;
  }

  let body = "";
  for await (const chunk of req) body += chunk;

  try {
    const parsed = JSON.parse(body);
    const result = await handleQuery(parsed);
    res.writeHead(200, { "Content-Type": "application/json" });
    res.end(JSON.stringify(result));
  } catch (err) {
    res.writeHead(500, { "Content-Type": "application/json" });
    res.end(JSON.stringify({ error: err.message, result: "" }));
  }
});

server.listen(PORT, "127.0.0.1", () => {
  console.log(`claude-sidecar listening on 127.0.0.1:${PORT}`);
});

// Graceful shutdown
for (const sig of ["SIGTERM", "SIGINT"]) {
  process.on(sig, () => {
    console.log(`${sig} received, shutting down`);
    server.close();
    process.exit(0);
  });
}
