# Vibe Replay -- Product Brief

**Date:** 2026-02-22
**Author:** Product Review (PM audit)
**Status:** Honest assessment for v1.0 planning

---

## 1. Current Product Audit

### What Works

**The capture pipeline is solid.** The PostToolUse + Stop hook architecture is correct. Capture is fast, non-blocking, and fail-safe (never crashes the host session). JSONL append-only storage is the right choice for write-heavy, read-seldom data. The hook installation/uninstall is clean and backs up settings before modifying them. This is the kind of infrastructure that should be invisible, and it is.

**The HTML template is genuinely good-looking.** Dark/light theme, GitHub-aesthetic CSS variables, responsive layout, sticky sidebar, proper diff coloring, phase icons, filter buttons. For a v0.1, the visual foundation is above average. Someone seeing this for the first time would say "oh, this looks professional."

**The data model is well-structured.** Pydantic models are clean. The Event -> TimelinePhase -> SessionReplay -> Insight hierarchy makes sense. The separation between raw events (JSONL) and analyzed replay (JSON) is correct.

**The CLI is feature-complete for a prototype.** install/uninstall/sessions/show/replay/export/analyze/wisdom/serve covers the full lifecycle. Rich output in the terminal is a nice touch.

### What's Weak

**The analysis is shallow.** This is the biggest problem. The "insights" are mechanical pattern matches, not actual insights:
- "Hotspot file: index.html" -- yes, I modified it a lot, so what?
- "Multiple exploration phases" -- okay, and?
- "Read-heavy session" / "Implementation-heavy session" -- this is restating the tool counts in prose form.

Nobody is going to look at these and think "wow, I learned something." The insights lack the "so what?" layer. They describe *what* happened without interpreting *why it matters*. Compare this to what a thoughtful colleague would say watching your session: "You spent 20 minutes exploring three different auth patterns before landing on JWT -- the SQLAlchemy sessions approach was a dead end because of the stateless constraint you discovered in the FastAPI docs." THAT is an insight. What we have is a statistics restatement.

**The session summary is embarrassingly generic.** The actual generated output says: "Worked on 17 file(s) | primarily testing, configuration." This is the headline -- the first thing someone sees -- and it says nothing meaningful. For a tool about "reflection and wisdom," the generated narrative is closer to `wc -l` than to actual reflection.

**Phase detection is fragile and noisy.** The cortex-birth-replay has 14 phases for 92 events. That is way too many. The merging heuristic (absorb runs < 2-3 events) is not aggressive enough. A 33-minute session should have maybe 4-6 phases, not 14. The result is that the timeline feels like a flat list with extra steps, not a meaningful narrative arc.

**Project name is "Unknown Project."** The stop hook captures metadata but leaves `project` as empty string. The capture hook does not attempt to infer the project name from the working directory, git remote, or Claude Code session context. This is a basic UX failure -- every replay should have a meaningful title.

**Code diffs are barely usable.** The diff extraction for Edit is just `-{old}\n+{new}` -- no context lines, no file path header, no line numbers. For Write, it dumps the entire new file content (truncated at 3000 chars). These are not diffs anyone would want to read. Compare to what GitHub shows in a PR -- context, line numbers, syntax highlighting. The current diffs are raw string dumps.

**No user prompt / conversation capture.** The most critical piece of context -- what the user ASKED the AI to do -- is completely absent. We capture every tool call but not the human intent that drove them. Without this, the replay is like watching someone code with no audio. You can see what they did but not why.

**The "serve" command is a toy.** The index page is an inline HTML string in Python. It works, but it is not something you would show anyone. There is no session detail view beyond re-rendering the full replay. No search, no filtering, no pagination.

**The MCP server is a nice idea but probably not the product.** Per the founder's insight, Vibe Replay is a visualization/sharing tool, not an AI memory layer. The MCP server should exist but should not be a primary focus.

### What's Missing Entirely

- **Sharing.** There is no way to share a replay that does not involve manually emailing an HTML file. No URL, no hosting, no "copy link." For a product whose core value proposition is "shareable replays," this is a glaring omission.
- **User narrative / prompt context.** As noted above. The "why" behind every action is missing.
- **Session naming/titling.** No way to give a session a human-readable name or description after the fact.
- **Visual progress bar / minimap.** In the HTML replay, there is no way to see the overall session shape at a glance. No progress bar showing explore/implement/debug phases as colored segments. You have to scroll through the timeline to understand the session arc.
- **Playback / animation.** The name is "Vibe Replay" but there is no actual replay -- no step-through, no autoplay, no temporal navigation. It is a static document, not a replay.
- **Search within a session.** Cannot search events or diffs within a replay.
- **Comparison.** No way to compare two sessions or see how a codebase evolved across sessions.
- **File-centric view.** Cannot view all changes to a specific file across the session, grouped together. The timeline is chronological-only.
- **Syntax highlighting in diffs.** No Prism, no Highlight.js, nothing. Diffs are unstyled monospace text.

---

## 2. Competitive Landscape

There is no direct competitor that does exactly "AI coding session replay." But there are adjacent products that define user expectations:

### Loom / Screen recording tools
**What they do well:** One-click capture, instant shareable link, viewer analytics, async communication. The key insight: the value of Loom is not the recording, it is the URL. If you cannot share it in 2 seconds, it does not exist.

**Lesson for Vibe Replay:** The share flow must be trivially easy. Generate a self-contained HTML file and give the user a way to host/share it instantly (even if that is just `pbcopy` the file path or a drag-and-drop to a hosting service).

### GitHub Pull Requests / Code Review
**What they do well:** Structured narrative (title, description, file-by-file diffs), conversation threading, before/after context, syntax highlighting. PRs are the gold standard for "here is what changed and why."

**Lesson for Vibe Replay:** The file-centric view matters. People want to see "what happened to auth.py" not just "event #47 was an Edit." Syntax highlighting is table stakes for anything that shows code.

### Jupyter Notebooks
**What they do well:** Interleave narrative with code execution. The reader sees the thought process alongside the results. Notebooks are "literate programming" -- story + code in one document.

**Lesson for Vibe Replay:** The replay should read like a story, not a log file. Narrative context between phases would transform the experience.

### git log / git blame
**What they do well:** Temporal attribution (who changed what, when). `git blame` lets you trace any line back to its origin. `git log --stat` gives a high-level view of a project's evolution.

**Lesson for Vibe Replay:** Vibe Replay could be "git blame but for AI sessions" -- trace any piece of code back to the session and decision that produced it.

### Posthog / Session Replay (web analytics)
**What they do well:** Visual replay of user sessions, heatmaps, rage click detection, funnel analysis. They turn raw event streams into visual narratives.

**Lesson for Vibe Replay:** The minimap/progress bar concept. Posthog shows a colored bar at the top indicating page transitions and activity intensity. Vibe Replay should have something similar showing the explore/build/debug rhythm.

### Streaming / VOD platforms (Twitch, YouTube)
**What they do well:** Timestamps, chapters, highlights, clip creation. A 4-hour stream becomes consumable through chapter markers and timestamps.

**Lesson for Vibe Replay:** Long sessions need navigation. Chapter markers (phases), jump-to timestamps, and highlights (key decisions, turning points) are essential for sessions with 100+ events.

---

## 3. Feature Priorities

### P0: Must-have for v1.0
These are the minimum to make someone say "this is actually useful, I want to use this again."

1. **Session progress minimap / phase bar**
   A horizontal bar at the top of the replay showing the session's phase composition as colored segments (blue=explore, green=implement, red=debug, etc.). Clickable to jump to that phase. This is the single highest-impact visual improvement -- it gives the user the "shape" of the session at a glance. Without it, the replay is just a long scrollable list.

2. **Auto-detect project name**
   Infer the project name from: (a) the working directory basename in the first Bash/Read event, (b) git remote origin, or (c) the Claude Code session context if available. "Unknown Project" is not acceptable for the main headline.

3. **Meaningful session summary (LLM-generated or heuristic-upgraded)**
   Replace "Worked on 17 file(s) | primarily testing, configuration" with something like: "Set up the Cortex project: created package structure, configured CI, built the landing page HTML/CSS, and tested the development server." This could be done with a simple heuristic (extract key file names, group by directory, use the phase sequence as a narrative frame) or, optionally, by calling a small LLM. Either way, the summary needs to tell a story.

4. **Reduce phase fragmentation**
   More aggressive merging: sessions under 60 minutes should have at most 6-8 phases. Short phases (< 3 events or < 2 minutes) should always be absorbed into neighbors. The timeline should feel like a chapter list, not a commit log.

5. **Syntax-highlighted code diffs**
   Embed a lightweight syntax highlighter (Prism.js is ~15KB minified and can be inlined). Add proper unified diff format with context lines. This is table stakes for anything that shows code.

6. **Playback mode**
   Add a "Play" button that auto-advances through events with a configurable speed (e.g., 1 event/second). Show a progress indicator. Allow pause/resume and scrubbing. This is the difference between a "replay" and a "report." It does not need to be fancy -- even a simple auto-scroll with highlighting would work.

7. **Search within session**
   A search box in the replay HTML that filters events by keyword (file name, tool name, summary text). For a 92-event session this barely matters, but for 500+ event sessions it is essential.

### P1: Nice-to-have (polish and delight)

8. **Session naming / annotation**
   CLI command to rename a session or add a description: `vibe-replay tag <id> --name "Built auth system" --tags "auth,fastapi"`. This metadata should appear in the replay header and session list.

9. **File-centric view tab**
   A second tab in the replay HTML that groups all events by file instead of by time. "auth.py: Read (10:15) -> Edit (10:23) -> Edit (10:31) -> Read (10:45)". This answers "what happened to this file?" directly.

10. **One-click export to clipboard / share**
    A button in the HTML replay that copies a shareable version to clipboard (or triggers download). In the CLI, `vibe-replay share <id>` that outputs a self-contained HTML file to a predictable location. Stretch: upload to a paste service and return a URL.

11. **Improved insight quality**
    Upgrade insights from "what happened" to "what it means":
    - Instead of "Hotspot file: index.html" -> "index.html was the central file, modified 6 times across 3 phases (implementation, debugging, refactoring) -- it may benefit from being split into smaller modules."
    - Instead of "Multiple exploration phases" -> "The session alternated between exploration and implementation 3 times, suggesting the requirements were discovered incrementally rather than known upfront."
    - Add: time-between-error-and-fix as a difficulty indicator. Add: ratio of exploration to implementation as a familiarity indicator.

12. **Keyboard shortcuts in HTML replay**
    `j/k` to navigate events, `Enter` to expand/collapse, `f` to toggle filters, `/` to search. Power users will expect these.

13. **Activity heatmap**
    A small visualization showing event density over time. Identifies bursts of activity and quiet periods. Could be a simple sparkline or a GitHub-style contribution grid adapted for a single session.

### P2: Future vision

14. **Cross-session dashboard**
    A persistent web dashboard (the `serve` command, upgraded) that shows all sessions with search, filtering by project/date/tag, and aggregate statistics. Think "your coding history, visualized."

15. **AI-generated narrative**
    Use an LLM to generate a paragraph-by-paragraph narrative of the session: "The developer started by exploring the existing codebase, reading 8 files to understand the current architecture. They then made a key decision to use a decorator pattern for auth middleware, which required modifying 3 files. A test failure at 10:35 led to a 15-minute debugging detour..." This transforms the replay from a structured log into a readable story.

16. **Team sharing / hosting**
    A lightweight service (or GitHub Pages integration) to host replays with shareable URLs. This is where the "sharing" promise gets fully realized.

17. **Diff viewer per file (GitHub-style)**
    For each file modified in the session, show the cumulative diff (initial state -> final state) with all intermediate changes annotated. This is what code review actually needs.

18. **Session comparison**
    Compare two sessions side-by-side: same project, different approaches. Or: compare a debugging session with the original implementation session to see what went wrong.

19. **Capture user prompts**
    If Claude Code hooks can provide the user's messages (not just tool calls), capture them as first-class events. This would transform the replay from "what the AI did" to "what the human asked and what the AI did in response."

20. **Plugin system for custom hooks**
    Let users define custom event types, custom analysis passes, and custom renderers. Power users who want to track specific patterns (e.g., "how often do I refactor after first implementation?") could build custom analyzers.

---

## 4. Core User Story

**Persona:** Dev lead at a startup, managing 3 engineers who all use Claude Code daily.

**Scenario:** An engineer spent 2 hours building a new payment integration. The next day, the dev lead needs to review what was done -- not just the final PR, but the reasoning: Why Stripe over Braintree? Why did they restructure the webhook handler? What was that 30-minute debugging detour about?

**Without Vibe Replay:** The dev lead reads the PR diff (just final-state code), maybe asks the engineer in Slack ("why did you change the webhook handler?"), gets a partial answer, and approves the PR with an incomplete mental model. Knowledge about the decision-making process is lost.

**With Vibe Replay:** The dev lead opens the HTML replay. The phase minimap immediately shows: 15min exploration, 40min implementation, 30min debugging, 15min testing, 20min refactoring. They click the debugging phase and see the engineer hit a webhook signature verification error, tried three approaches, and landed on the one that handles both test and live keys. They click the "KEY DECISION" marker on event #47 and see the moment the engineer chose Stripe -- it was right after reading Braintree's docs and discovering their webhook retry policy was incompatible with the existing queue system. The dev lead now understands not just what was built, but *why*, and can give informed feedback. They send the replay link to the other engineers: "Check out how [name] handled the webhook signing -- good pattern we should reuse."

**The indispensable moment:** The replay turns a 2-hour opaque coding session into a 5-minute digestible story. It is the difference between reading a git diff and watching someone think.

---

## 5. Product Principles

### 1. The replay is the product, not the data.
Vibe Replay's value is in the rendered output -- the HTML file that someone opens and immediately understands. Raw JSONL, SQLite indexes, and analysis pipelines are implementation details. Every engineering decision should be evaluated by: "does this make the HTML replay better?"

### 2. Narrative over metrics.
Counting tool calls and listing file names is not insight. The replay should tell a story: beginning (what was the goal?), middle (what decisions were made?), end (what was the outcome?). Numbers should support the narrative, not replace it. If an insight cannot complete the sentence "This matters because ___," it should not be shown.

### 3. Share in under 10 seconds.
The path from "finished coding session" to "someone else is looking at my replay" should be under 10 seconds. One command, one file, one link. Self-contained HTML is the right format precisely because it requires no infrastructure. Do not add dependencies that break this property.

### 4. Never slow down the session.
Capture must be invisible. If the hooks add perceptible latency to Claude Code, the product is broken. Analysis and rendering can be slow (they happen after the fact) but capture must be instant and fail-safe. The current architecture (lightweight Python script, JSONL append) is correct. Do not add network calls, LLM invocations, or heavy processing to the capture path.

### 5. Opinionated defaults, customizable depth.
The default replay should be useful without any configuration. But power users should be able to: adjust phase detection sensitivity, add custom tags, choose which event types to show, and customize the visual theme. Design for the 80% case out of the box, and let the 20% customize.

---

## Appendix: What This Product Is NOT

- **Not an AI memory system.** Claude can read raw logs. Vibe Replay is for humans who want visual, structured, shareable replays.
- **Not a code review tool.** It shows the *process* that led to the code, not the final code itself. It complements PRs, not replaces them.
- **Not a monitoring/observability tool.** It is not about catching errors or tracking performance. It is about understanding decision-making.
- **Not a coding assistant.** It does not suggest improvements or generate code. It reflects on what already happened.
