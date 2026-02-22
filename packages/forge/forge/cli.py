"""CLI interface for Forge — Self-Evolving Tool Agent.

Provides interactive and non-interactive commands for creating,
testing, listing, and managing forge-created tools.
"""

from __future__ import annotations

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.syntax import Syntax

from forge import __version__
from forge.engine import ForgeEngine
from forge.models import InstallTarget, OutputType, SessionState
from forge.storage import delete_tool, get_tool_source, list_tools, load_tool
from forge.tester import run_tests

console = Console()
engine = ForgeEngine()


@click.group()
@click.version_option(version=__version__, prog_name="forge")
def cli() -> None:
    """Forge — Self-Evolving Tool Agent.

    Describe what you need in natural language. Forge generates the tool,
    tests it, iterates until it works, and installs it into your toolkit.
    """
    pass


@cli.command()
@click.argument("description")
@click.option(
    "--type", "-t",
    "output_type",
    type=click.Choice(["mcp", "cli", "python"], case_sensitive=False),
    default="python",
    help="Output type: MCP server tool, CLI command, or Python function.",
)
@click.option(
    "--no-clarify",
    is_flag=True,
    default=False,
    help="Skip clarification questions and generate immediately.",
)
@click.option(
    "--install",
    "install_to",
    type=click.Choice(["mcp", "cli", "local"], case_sensitive=False),
    default=None,
    help="Automatically install after generation.",
)
def create(
    description: str,
    output_type: str,
    no_clarify: bool,
    install_to: str | None,
) -> None:
    """Create a new tool from a natural language description.

    Examples:

        forge create "convert CSV to JSON with filtering"

        forge create "download a URL and extract all links" --type cli

        forge create "count words in a text file" --type mcp --install mcp
    """
    otype = OutputType(output_type)
    console.print(
        Panel(
            f"[bold]Creating tool:[/bold] {description}\n"
            f"[dim]Output type: {otype.value}[/dim]",
            title="Forge",
            border_style="cyan",
        )
    )

    session = engine.create_session(description, otype)

    # --- Clarification ---
    if not no_clarify:
        console.print("\n[bold cyan]Step 1: Clarification[/bold cyan]")
        result = engine.clarify(session)

        if result.has_questions:
            console.print(f"  Found {len(result.questions)} question(s):\n")
            answers = {}
            for q in result.questions:
                prompt = f"  [yellow]?[/yellow] {q.question}"
                if q.options:
                    prompt += f"\n    Options: {', '.join(q.options)}"
                if q.default:
                    prompt += f"\n    [dim]Default: {q.default}[/dim]"

                answer = click.prompt(prompt, default=q.default or "", show_default=False)
                answers[q.id] = answer

            engine.answer(session, answers)
            console.print("  [green]All questions answered.[/green]\n")
        else:
            console.print("  [green]No clarification needed.[/green]\n")
    else:
        session.update_state(SessionState.READY)
        console.print("[dim]Skipping clarification.[/dim]\n")

    # --- Generation ---
    console.print("[bold cyan]Step 2: Generating tool...[/bold cyan]")
    gen_result = engine.generate(session)

    if not gen_result.success:
        console.print(f"  [red]Generation failed:[/red] {gen_result.error}")
        raise SystemExit(1)

    console.print(f"  [green]Code generated.[/green] ({len(session.generated_code)} chars)")

    # --- Testing ---
    last_test = session.test_results[-1] if session.test_results else None
    if last_test:
        if last_test.passed:
            console.print(
                f"  [green]Tests passed![/green] "
                f"({last_test.total} tests, 0 failures)"
            )
        else:
            console.print(
                f"  [yellow]Tests failed.[/yellow] "
                f"({last_test.total} tests, {last_test.failures} failures)"
            )

    # --- Iteration ---
    iteration = 0
    while session.state == SessionState.ITERATING:
        iteration += 1
        console.print(f"\n[bold cyan]Iteration {iteration}:[/bold cyan] Fixing issues...")
        engine.iterate(session)

        last_test = session.test_results[-1]
        if last_test.passed:
            console.print(f"  [green]Tests passed on iteration {iteration}![/green]")
        else:
            console.print(
                f"  [yellow]Still failing.[/yellow] "
                f"({last_test.failures} failures, errors: {last_test.errors[:2]})"
            )

    # --- Result ---
    if session.state == SessionState.SUCCEEDED:
        console.print(
            Panel(
                f"[green bold]Tool created successfully![/green bold]\n\n"
                f"Name: {session.spec.name}\n"
                f"Tests: {session.test_results[-1].total} passed\n"
                f"Iterations: {session.iteration}",
                title="Success",
                border_style="green",
            )
        )

        # Show generated code
        console.print("\n[bold]Generated code:[/bold]")
        console.print(Syntax(session.generated_code, "python", theme="monokai"))

        # --- Installation ---
        if install_to:
            target = InstallTarget(install_to)
            console.print(f"\n[bold cyan]Installing to {target.value}...[/bold cyan]")
            inst_result = engine.install(session, target)
            if inst_result.success:
                console.print(f"  [green]{inst_result.message}[/green]")
            else:
                console.print(f"  [red]Installation failed:[/red] {inst_result.error}")
    else:
        console.print(
            Panel(
                f"[red bold]Tool creation failed.[/red bold]\n\n"
                f"Iterations: {session.iteration}/{session.max_iterations}\n"
                f"Last errors: {session.test_results[-1].errors[:3] if session.test_results else 'N/A'}",
                title="Failed",
                border_style="red",
            )
        )
        raise SystemExit(1)


@cli.command("list")
def list_cmd() -> None:
    """List all tools created by Forge."""
    tools = list_tools()

    if not tools:
        console.print("[dim]No tools created yet. Use 'forge create' to make one.[/dim]")
        return

    table = Table(title="Forge Tools", border_style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Type", style="cyan")
    table.add_column("Description")
    table.add_column("Tests", justify="center")
    table.add_column("Installed", justify="center")
    table.add_column("Created")

    for tool in tools:
        table.add_row(
            tool.name,
            tool.output_type.value,
            tool.description[:50] + ("..." if len(tool.description) > 50 else ""),
            "[green]PASS[/green]" if tool.test_passed else "[red]FAIL[/red]",
            "[green]Yes[/green]" if tool.installed else "[dim]No[/dim]",
            tool.created_at.strftime("%Y-%m-%d"),
        )

    console.print(table)


@cli.command()
@click.argument("name")
def test(name: str) -> None:
    """Re-run tests for a tool.

    NAME is the tool name (e.g., convert_csv_to_json).
    """
    from forge.storage import get_tool_source, get_tool_tests

    source = get_tool_source(name)
    tests = get_tool_tests(name)

    if not source or not tests:
        console.print(f"[red]Tool '{name}' not found.[/red]")
        raise SystemExit(1)

    console.print(f"[bold]Running tests for '{name}'...[/bold]")
    result = run_tests(source, tests, name)

    if result.passed:
        console.print(
            f"[green]All {result.total} tests passed![/green]"
        )
    else:
        console.print(
            f"[red]{result.failures} of {result.total} tests failed.[/red]"
        )
        console.print(f"\n[bold]Output:[/bold]\n{result.output}")


@cli.command()
@click.argument("name")
def show(name: str) -> None:
    """Show the source code of a tool."""
    source = get_tool_source(name)
    if not source:
        console.print(f"[red]Tool '{name}' not found.[/red]")
        raise SystemExit(1)

    meta = load_tool(name)
    if meta:
        console.print(
            Panel(
                f"[bold]{meta.display_name or meta.name}[/bold]\n"
                f"{meta.description}\n"
                f"Type: {meta.output_type.value} | "
                f"Tests: {'PASS' if meta.test_passed else 'FAIL'} | "
                f"Installed: {'Yes' if meta.installed else 'No'}",
                border_style="cyan",
            )
        )

    console.print(Syntax(source, "python", theme="monokai"))


@cli.command()
@click.argument("name")
@click.confirmation_option(prompt="Are you sure you want to uninstall this tool?")
def uninstall(name: str) -> None:
    """Remove a tool created by Forge."""
    meta = load_tool(name)

    if not meta:
        console.print(f"[red]Tool '{name}' not found.[/red]")
        raise SystemExit(1)

    # Uninstall from target
    if meta.installed and meta.install_path:
        from forge.installer import uninstall as do_uninstall
        target = InstallTarget(meta.output_type.value)
        result = do_uninstall(name, target)
        console.print(f"  {result.message}")

    # Delete from storage
    if delete_tool(name):
        console.print(f"[green]Tool '{name}' removed.[/green]")
    else:
        console.print(f"[yellow]Tool '{name}' not found in storage.[/yellow]")


@cli.command()
def serve() -> None:
    """Start the Forge MCP server.

    This allows Claude Code to invoke Forge tools directly through MCP.
    """
    console.print("[bold]Starting Forge MCP server...[/bold]")
    from forge.mcp_server import main as mcp_main
    mcp_main()


if __name__ == "__main__":
    cli()
