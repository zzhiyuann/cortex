"""CLI interface for Vibe Replay.

Provides commands for managing hooks, viewing sessions,
generating replays, and exporting data.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from . import __version__
from .analyzer import aggregate_learnings, analyze_session
from .hooks import check_hooks, install_hooks, uninstall_hooks
from .renderer import render_html, render_json, render_markdown
from .store import SessionStore


console = Console()


def _get_store() -> SessionStore:
    """Get the default session store."""
    return SessionStore()


@click.group()
@click.version_option(version=__version__, prog_name="vibe-replay")
def cli():
    """Vibe Replay — Capture, reflect, and share your AI coding sessions.

    Extracts structured wisdom from AI coding sessions: decisions made,
    patterns discovered, lessons learned.
    """
    pass


@cli.command()
def install():
    """Install Vibe Replay hooks into Claude Code."""
    with console.status("[bold cyan]Installing hooks..."):
        result = install_hooks()

    console.print()
    console.print(
        Panel(
            f"[bold green]Hooks installed successfully![/]\n\n"
            f"[dim]Capture hook:[/] {result['capture_hook']}\n"
            f"[dim]Stop hook:[/]    {result['stop_hook']}\n"
            f"[dim]Settings:[/]     {result['settings_path']}\n\n"
            f"[yellow]Your Claude Code sessions will now be captured automatically.[/]\n"
            f"[dim]Run [bold]vibe-replay sessions[/bold] to see captured sessions.[/]",
            title="[bold]Vibe Replay[/]",
            border_style="green",
        )
    )


@cli.command()
def uninstall():
    """Remove Vibe Replay hooks from Claude Code."""
    result = uninstall_hooks()
    console.print(
        f"[green]Hooks removed.[/] ({result['hooks_removed']} hook(s) removed)"
    )


@cli.command()
def status():
    """Check if Vibe Replay hooks are installed."""
    result = check_hooks()

    if result["installed"]:
        console.print("[green]Vibe Replay hooks are installed.[/]")
        for hook in result["hooks"]:
            console.print(f"  [dim]{hook['type']}:[/] {hook['command']}")
    else:
        console.print("[yellow]Vibe Replay hooks are not installed.[/]")
        console.print("[dim]Run [bold]vibe-replay install[/bold] to set up.[/]")


@cli.command()
@click.option("--project", "-p", default=None, help="Filter by project name.")
@click.option("--limit", "-n", default=20, help="Number of sessions to show.")
def sessions(project: str | None, limit: int):
    """List captured sessions."""
    store = _get_store()
    session_list = store.list_sessions(project=project, limit=limit)

    if not session_list:
        console.print("[yellow]No sessions captured yet.[/]")
        console.print("[dim]Make sure hooks are installed: [bold]vibe-replay install[/bold][/]")
        return

    table = Table(
        title="Captured Sessions",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("Session ID", style="cyan", max_width=20)
    table.add_column("Date", style="dim")
    table.add_column("Duration", justify="right")
    table.add_column("Events", justify="right", style="green")
    table.add_column("Files", justify="right")
    table.add_column("Summary", max_width=40)

    for s in session_list:
        duration = ""
        if s.duration_seconds:
            mins = int(s.duration_seconds // 60)
            secs = int(s.duration_seconds % 60)
            duration = f"{mins}m {secs}s" if mins else f"{secs}s"

        table.add_row(
            s.session_id[:18] + "..." if len(s.session_id) > 20 else s.session_id,
            s.start_time.strftime("%Y-%m-%d %H:%M") if s.start_time else "?",
            duration or "?",
            str(s.event_count),
            str(len(s.files_modified)),
            s.summary[:40] if s.summary else "[dim]—[/]",
        )

    console.print(table)


@cli.command()
@click.argument("session_id")
def show(session_id: str):
    """Show a session summary in the terminal."""
    store = _get_store()

    # Try to find the session (support partial IDs)
    session_id = _resolve_session_id(store, session_id)
    if not session_id:
        console.print("[red]Session not found.[/]")
        return

    replay = store.get_replay(session_id)
    if not replay:
        console.print("[yellow]No analysis found. Running analysis...[/]")
        replay = analyze_session(store, session_id)

    meta = replay.metadata

    # Header
    console.print()
    console.print(
        Panel(
            f"[bold]{meta.project or meta.session_id}[/]\n"
            f"[dim]Session:[/] {meta.session_id}\n"
            f"[dim]Date:[/]    {meta.start_time.strftime('%Y-%m-%d %H:%M')}\n"
            f"[dim]Duration:[/] {replay.statistics.get('duration_human', '?')}\n"
            f"[dim]Events:[/]  {meta.event_count} | "
            f"[dim]Files:[/] {len(meta.files_modified)} | "
            f"[dim]Changes:[/] {replay.statistics.get('code_changes', 0)}\n\n"
            f"{meta.summary or ''}",
            title="[bold cyan]Session Replay[/]",
            border_style="cyan",
        )
    )

    # Timeline
    if replay.timeline:
        console.print("\n[bold]Timeline[/]")
        for phase in replay.timeline:
            icon = {
                "exploration": "[blue]EXPLORE[/]",
                "implementation": "[green]IMPLEMENT[/]",
                "debugging": "[red]DEBUG[/]",
                "testing": "[yellow]TEST[/]",
                "refactoring": "[magenta]REFACTOR[/]",
                "configuration": "[cyan]CONFIG[/]",
                "documentation": "[white]DOCS[/]",
            }.get(phase.phase.value, "[dim]OTHER[/]")

            console.print(
                f"  {icon:>22} "
                f"[dim]{phase.start_time.strftime('%H:%M')}-{phase.end_time.strftime('%H:%M')}[/] "
                f"({phase.event_count} events) {phase.summary}"
            )

    # Insights
    if replay.insights:
        console.print("\n[bold]Insights[/]")
        for insight in replay.insights:
            icon = {
                "decision": "[blue]DECISION[/]",
                "learning": "[green]LEARNING[/]",
                "mistake": "[yellow]DETOUR[/]",
                "pattern": "[magenta]PATTERN[/]",
                "turning_point": "[red]TURNING[/]",
            }.get(insight.insight_type.value, "")
            console.print(f"  {icon} [bold]{insight.title}[/]")
            console.print(f"    [dim]{insight.description}[/]")

    # Top tools
    if meta.tools_used:
        console.print("\n[bold]Tools Used[/]")
        for tool, count in sorted(
            meta.tools_used.items(), key=lambda x: x[1], reverse=True
        )[:8]:
            bar = "[green]" + "|" * min(count, 40) + "[/]"
            console.print(f"  {tool:>12} {bar} {count}")

    console.print()


@cli.command()
@click.argument("session_id")
def replay(session_id: str):
    """Generate HTML replay and open in browser."""
    store = _get_store()
    session_id = _resolve_session_id(store, session_id)
    if not session_id:
        console.print("[red]Session not found.[/]")
        return

    with console.status("[bold cyan]Generating replay..."):
        replay_data = store.get_replay(session_id)
        if not replay_data:
            replay_data = analyze_session(store, session_id)

        events = store.get_events(session_id)
        html = render_html(replay_data, events)

    # Save to temp file and open
    output_path = Path(tempfile.mktemp(suffix=".html", prefix="vibe-replay-"))
    output_path.write_text(html)

    console.print(f"[green]Replay saved to:[/] {output_path}")
    webbrowser.open(f"file://{output_path}")


@cli.command(name="export")
@click.argument("session_id")
@click.option(
    "--format", "-f", "fmt",
    type=click.Choice(["html", "md", "json"]),
    default="html",
    help="Export format.",
)
@click.option("--output", "-o", default=None, help="Output file path.")
def export_cmd(session_id: str, fmt: str, output: str | None):
    """Export a session replay."""
    store = _get_store()
    session_id = _resolve_session_id(store, session_id)
    if not session_id:
        console.print("[red]Session not found.[/]")
        return

    with console.status(f"[bold cyan]Exporting as {fmt}..."):
        replay_data = store.get_replay(session_id)
        if not replay_data:
            replay_data = analyze_session(store, session_id)

        events = store.get_events(session_id)

        if fmt == "html":
            content = render_html(replay_data, events)
            ext = ".html"
        elif fmt == "md":
            content = render_markdown(replay_data, events)
            ext = ".md"
        else:
            content = render_json(replay_data, events)
            ext = ".json"

    if output:
        out_path = Path(output)
    else:
        out_path = Path(f"replay-{session_id[:12]}{ext}")

    out_path.write_text(content)
    console.print(f"[green]Exported to:[/] {out_path}")


@cli.command()
@click.argument("session_id")
def analyze(session_id: str):
    """Run or re-run analysis on a session."""
    store = _get_store()
    session_id = _resolve_session_id(store, session_id)
    if not session_id:
        console.print("[red]Session not found.[/]")
        return

    with console.status("[bold cyan]Analyzing session..."):
        replay_data = analyze_session(store, session_id)

    console.print(f"[green]Analysis complete.[/]")
    console.print(
        f"  Phases: {len(replay_data.timeline)} | "
        f"Insights: {len(replay_data.insights)} | "
        f"Decisions: {len(replay_data.key_decision_indices)} | "
        f"Turning points: {len(replay_data.turning_point_indices)}"
    )


@cli.command()
@click.option("--limit", "-n", default=50, help="Sessions to consider.")
def wisdom(limit: int):
    """Show aggregated learnings across all sessions."""
    store = _get_store()

    with console.status("[bold cyan]Aggregating wisdom..."):
        insights = aggregate_learnings(store, limit=limit)

    if not insights:
        console.print("[yellow]No insights found yet. Capture some sessions first![/]")
        return

    console.print(
        Panel(
            f"[bold]Aggregated Wisdom[/] from up to {limit} sessions\n"
            f"[dim]{len(insights)} unique insight(s) found[/]",
            border_style="magenta",
        )
    )

    for insight in insights:
        type_style = {
            "decision": "blue",
            "learning": "green",
            "mistake": "yellow",
            "pattern": "magenta",
            "turning_point": "red",
        }.get(insight.insight_type.value, "white")

        console.print(
            f"\n  [{type_style}][{insight.insight_type.value.upper()}][/{type_style}] "
            f"[bold]{insight.title}[/]"
        )
        console.print(f"  [dim]{insight.description}[/]")
        console.print(
            f"  [dim]Confidence: {insight.confidence:.0%}[/]"
        )


@cli.command()
@click.option("--port", "-p", default=8765, help="Port to serve on.")
@click.option("--host", "-h", default="localhost", help="Host to bind to.")
def serve(port: int, host: str):
    """Start a local web server to browse all replays."""
    import http.server
    import socketserver
    import threading

    store = _get_store()

    class ReplayHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/" or self.path == "/index":
                self._serve_index()
            elif self.path.startswith("/session/"):
                session_id = self.path.split("/session/")[1].strip("/")
                self._serve_session(session_id)
            else:
                self.send_error(404)

        def _serve_index(self):
            sessions_list = store.list_sessions(limit=100)
            html = _build_index_html(sessions_list)
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        def _serve_session(self, session_id):
            session_id = _resolve_session_id(store, session_id)
            if not session_id:
                self.send_error(404, "Session not found")
                return

            replay_data = store.get_replay(session_id)
            if not replay_data:
                replay_data = analyze_session(store, session_id)

            events = store.get_events(session_id)
            html = render_html(replay_data, events)

            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(html.encode())

        def log_message(self, format, *args):
            # Suppress default logging
            pass

    with socketserver.TCPServer((host, port), ReplayHandler) as httpd:
        url = f"http://{host}:{port}"
        console.print(
            Panel(
                f"[bold green]Vibe Replay server running![/]\n\n"
                f"[dim]URL:[/] [link={url}]{url}[/link]\n"
                f"[dim]Press Ctrl+C to stop.[/]",
                border_style="green",
            )
        )
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            console.print("\n[dim]Server stopped.[/]")


def _git_run(args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run a git command in the given directory."""
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


def _parse_github_remote(cwd: Path) -> tuple[str, str] | None:
    """Parse GitHub username and repo name from git remote URL.

    Returns (username, repo) or None if not a GitHub remote.
    """
    try:
        result = _git_run(["remote", "get-url", "origin"], cwd=cwd)
        url = result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

    # SSH: git@github.com:user/repo.git
    m = re.match(r"git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)

    # HTTPS: https://github.com/user/repo.git
    m = re.match(r"https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?$", url)
    if m:
        return m.group(1), m.group(2)

    return None


def _update_replays_index(replays_dir: Path) -> None:
    """Generate an index.html listing all replay HTML files in the directory."""
    html_files = sorted(replays_dir.glob("*.html"))
    html_files = [f for f in html_files if f.name != "index.html"]

    rows = ""
    for f in html_files:
        name = f.stem.replace("-", " ").replace("_", " ")
        rows += f'<li><a href="{f.name}">{name}</a></li>\n'

    index_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Vibe Replay — Shared Replays</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #0d1117; color: #e6edf3; padding: 2rem; max-width: 800px; margin: 0 auto; }}
h1 {{ color: #58a6ff; }}
p {{ color: #8b949e; }}
ul {{ list-style: none; padding: 0; }}
li {{ padding: 0.5rem 0; border-bottom: 1px solid #21262d; }}
a {{ color: #58a6ff; text-decoration: none; font-size: 1.05rem; }}
a:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>&#127916; Vibe Replay</h1>
<p>Shared AI coding session replays</p>
<ul>
{rows}
</ul>
<p style="margin-top:2rem;font-size:0.8rem;color:#6e7681;">
Generated by <a href="https://github.com/zzhiyuann/vibe-replay">Vibe Replay</a>
</p>
</body>
</html>"""

    (replays_dir / "index.html").write_text(index_html)


@cli.command()
@click.argument("session_id")
@click.option(
    "--repo",
    default=None,
    help="Path to the cortex repository (default: ~/projects/cortex).",
)
@click.option("--open/--no-open", "open_browser", default=True, help="Open URL in browser.")
def share(session_id: str, repo: str | None, open_browser: bool):
    """Share a replay publicly via GitHub Pages."""
    store = _get_store()
    session_id = _resolve_session_id(store, session_id)
    if not session_id:
        console.print("[red]Session not found.[/]")
        return

    repo_path = Path(repo) if repo else Path.home() / "projects" / "cortex"
    if not (repo_path / ".git").exists():
        console.print(f"[red]Git repository not found at {repo_path}[/]")
        console.print("[dim]Use --repo to specify the cortex repo path.[/]")
        return

    # Parse GitHub remote for URL construction
    remote_info = _parse_github_remote(repo_path)
    if not remote_info:
        console.print("[red]Could not parse GitHub remote URL from the repository.[/]")
        return

    username, repo_name = remote_info

    with console.status("[bold cyan]Generating and sharing replay..."):
        # Generate HTML
        replay_data = store.get_replay(session_id)
        if not replay_data:
            replay_data = analyze_session(store, session_id)

        events = store.get_events(session_id)

        # Build filename
        project = replay_data.metadata.project or "session"
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", project).strip("-").lower()
        date_str = replay_data.metadata.start_time.strftime("%Y%m%d")
        filename = f"{safe_name}-{date_str}-{session_id[:8]}.html"

        # Construct the public URL
        share_url = f"https://{username}.github.io/{repo_name}/replays/{filename}"

        # Render with share URL embedded
        html = render_html(replay_data, events, share_url=share_url)

        # Write to replays/ directory
        replays_dir = repo_path / "replays"
        replays_dir.mkdir(parents=True, exist_ok=True)
        output_file = replays_dir / filename
        output_file.write_text(html)

        # Update index
        _update_replays_index(replays_dir)

        # Git add, commit, push
        try:
            _git_run(["add", "replays/"], cwd=repo_path)
            _git_run(
                ["commit", "-m", f"Add replay: {project} ({date_str})"],
                cwd=repo_path,
            )
            _git_run(["push"], cwd=repo_path)
        except subprocess.CalledProcessError as e:
            console.print(f"[red]Git operation failed:[/] {e.stderr or e.stdout}")
            console.print(f"[dim]File saved at: {output_file}[/]")
            return

    console.print(
        Panel(
            f"[bold green]Replay shared successfully![/]\n\n"
            f"[dim]File:[/] {output_file}\n"
            f"[dim]URL:[/]  [link={share_url}]{share_url}[/link]\n\n"
            f"[yellow]Note: GitHub Pages may take a minute to update.[/]",
            title="[bold]Vibe Replay Share[/]",
            border_style="green",
        )
    )

    if open_browser:
        webbrowser.open(share_url)


def _resolve_session_id(store: SessionStore, partial_id: str) -> str | None:
    """Resolve a partial session ID to a full one."""
    if store.session_exists(partial_id):
        return partial_id

    # Try prefix match
    sessions_dir = store.sessions_dir
    if not sessions_dir.exists():
        return None

    matches = [
        d.name
        for d in sessions_dir.iterdir()
        if d.is_dir() and d.name.startswith(partial_id)
    ]

    if len(matches) == 1:
        return matches[0]
    elif len(matches) > 1:
        console.print(f"[yellow]Ambiguous session ID. Matches:[/]")
        for m in matches[:10]:
            console.print(f"  {m}")
        return None

    return None


def _build_index_html(sessions: list) -> str:
    """Build a simple HTML index page listing all sessions."""
    rows = ""
    for s in sessions:
        duration = ""
        if s.duration_seconds:
            mins = int(s.duration_seconds // 60)
            duration = f"{mins}m"
        rows += f"""
        <tr onclick="window.location='/session/{s.session_id}'" style="cursor:pointer">
            <td><code>{s.session_id[:16]}...</code></td>
            <td>{s.start_time.strftime('%Y-%m-%d %H:%M') if s.start_time else '?'}</td>
            <td>{duration or '?'}</td>
            <td>{s.event_count}</td>
            <td>{s.summary or '—'}</td>
        </tr>
        """

    return f"""<!DOCTYPE html>
<html><head>
<title>Vibe Replay</title>
<style>
body {{ font-family: -apple-system, sans-serif; background: #0d1117; color: #e6edf3; padding: 2rem; }}
h1 {{ color: #58a6ff; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 1rem; }}
th, td {{ padding: 0.75rem; text-align: left; border-bottom: 1px solid #21262d; }}
th {{ color: #8b949e; font-size: 0.85rem; text-transform: uppercase; }}
tr:hover {{ background: #161b22; }}
code {{ background: #21262d; padding: 0.2rem 0.4rem; border-radius: 4px; font-size: 0.85rem; }}
</style>
</head><body>
<h1>&#127916; Vibe Replay</h1>
<p style="color:#8b949e">Captured AI coding sessions</p>
<table>
<thead><tr><th>Session</th><th>Date</th><th>Duration</th><th>Events</th><th>Summary</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</body></html>"""


def main():
    """Entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
