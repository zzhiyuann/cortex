# Cortex Examples

Walkthroughs showing how Cortex extends Claude Code via MCP servers and hooks.

## 1. Forge: Claude Code Builds Its Own Tools

**Setup:** Add Forge as an MCP server so Claude Code can generate tools on the fly.

```json
// ~/.claude/claude_desktop_config.json
{
  "mcpServers": {
    "forge": {
      "command": "forge-mcp"
    }
  }
}
```

**What happens:** When you ask Claude Code to do something that requires a tool it doesn't have, it can build one.

```
You: I need to convert all the YAML files in this project to TOML format.

Claude: I don't have a YAML-to-TOML converter, but I can build one.
        [calls forge_create with "convert YAML files to TOML format"]

Forge:  Generating tool... Running tests... 4/4 passed ✓
        Tool 'yaml_to_toml' installed as MCP tool.

Claude: Done! The tool is installed. After a restart, I'll be able to
        use it natively. For now, here's what it generated:
        [shows the generated code]
```

**Key insight:** Forge doesn't just generate code — it tests it, iterates on failures, and installs it into Claude Code's MCP config. Next session, the tool is available natively.

---

## 2. A2A Hub: Multi-Agent Delegation

**Setup:** Start the A2A Hub and register specialized agents.

```bash
# Terminal 1: Start the hub
a2a-hub start

# Terminal 2: Start a code review agent
python examples/code_review_agent.py

# Terminal 3: Start a research agent
python examples/research_agent.py
```

**With MCP Bridge:** Claude Code can discover and delegate to these agents.

```json
// Add to MCP config
{
  "mcpServers": {
    "a2a": {
      "command": "python",
      "args": ["-m", "a2a_hub.mcp_bridge"]
    }
  }
}
```

```
You: Review the authentication module for security issues, and research
     best practices for JWT token rotation.

Claude: I'll delegate these to specialized agents.
        [calls discover_agents with capability="security-audit"]
        [calls delegate_task to code-reviewer: "audit auth module"]
        [calls delegate_task to researcher: "JWT rotation best practices"]

        Code reviewer found 2 issues:
        1. Token expiry not checked on refresh
        2. Secret key hardcoded in config

        Researcher recommends:
        - Rotate signing keys every 90 days
        - Use asymmetric keys (RS256) for production
        - Implement key ID (kid) header for smooth rotation
```

---

## 3. Vibe Replay: Capture and Share Sessions

**Setup:** Install hooks into Claude Code (one-time).

```bash
vibe-replay install
```

This adds PostToolUse and Stop hooks to Claude Code's config. Every tool call is now captured automatically.

**After a coding session:**

```bash
# List captured sessions
$ vibe-replay sessions

  ID        Project          Duration  Events  Phase
  abc123    my-webapp        45m       92      testing
  def456    api-service      1h 12m    156     implementation
  ghi789    data-pipeline    23m       41      debugging

# Generate a shareable HTML replay
$ vibe-replay replay abc123 --format html --open
```

The HTML replay shows:
- **Timeline** with color-coded phases (exploration → implementation → testing)
- **Decision points** identified by the analyzer
- **Turning points** where debugging led to breakthroughs
- **File hotspots** showing which files got the most attention
- **Insights** extracted from the session

**Share with your team:** The HTML is self-contained (no CDN, no external deps). Send the file, open in any browser.

---

## 4. Dispatcher: Code from Your Phone

**Setup:**

```bash
# One-time setup
dispatcher init
# → Enter your Telegram bot token and chat ID

# Start the daemon
dispatcher start
```

**From Telegram:**

```
You:  fix the flaky test in test_auth.py
Bot:  🔄 Starting task in ~/projects/api...
Bot:  ✅ Fixed! The test was using a hardcoded timestamp.
      Changed to use freezegun for deterministic time.
      Files modified: tests/test_auth.py

You:  what about the other tests?
Bot:  (auto-links as follow-up to previous task)
      All 47 tests pass. The flaky test was the only issue.
```

**Project routing:** Configure keywords per project, and Dispatcher auto-routes messages:

```yaml
# ~/.config/dispatcher/config.yaml
projects:
  api:
    path: ~/projects/api
    keywords: [api, backend, auth, test]
  webapp:
    path: ~/projects/webapp
    keywords: [webapp, frontend, react, ui]
```

---

## 5. Full Stack: All Components Together

The real power is when components work together:

1. **You send a Telegram message:** "Add rate limiting to the API"
2. **Dispatcher** routes it to the API project and spawns Claude Code
3. **Claude Code** delegates research to a specialized agent via **A2A Hub**
4. Claude Code needs a Redis rate limiter tool — **Forge** generates one
5. The entire session is captured by **Vibe Replay**
6. Key decisions are extracted and stored in **Memory**
7. Next session, Claude Code recalls "we chose token bucket algorithm because..." via Memory

Each component adds a capability. Together, they transform an isolated coding assistant into a persistent, collaborative, self-improving system.
