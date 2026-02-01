"""
ARH CLI - Command Line Interface

Main entry point for the Agent Reliability Harness.
Provides commands for testing agents and auditing documentation.
"""

import typer
import json
import os
from typing import Optional, List
from pathlib import Path

# Create the main app
app = typer.Typer(
    name="arh",
    help="Agent Reliability Harness - SRE for AI Agents",
    add_completion=False
)


def get_console():
    """Get Rich console (imported lazily to speed up CLI startup)."""
    from rich.console import Console
    return Console()


@app.command()
def test(
    agent_url: str = typer.Option(..., "--agent", "-a", help="Agent endpoint URL"),
    test_type: str = typer.Option("all", "--type", "-t", 
        help="Test type: all|robustness|consistency|groundedness|predictability"),
    prompts_file: Optional[Path] = typer.Option(None, "--prompts", "-p", 
        help="File with test prompts (one per line)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", 
        help="Output file for report"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", 
        help="API key for agent"),
    model: str = typer.Option("gpt-4o-mini", "--model", "-m",
        help="Model name to use")
):
    """Run reliability tests on an agent endpoint."""
    from ..core.agent_wrapper import AgentWrapper
    from ..core.harness import ReliabilityHarness
    
    console = get_console()
    console.print(f"[bold blue]Testing agent:[/bold blue] {agent_url}")
    
    # Load prompts
    if prompts_file and prompts_file.exists():
        prompts = prompts_file.read_text().strip().split('\n')
        console.print(f"[dim]Loaded {len(prompts)} prompts from {prompts_file}[/dim]")
    else:
        prompts = [
            "What is the capital of France?",
            "Explain photosynthesis briefly.",
            "What is 2 + 2?",
            "Who wrote Romeo and Juliet?",
            "What is machine learning?"
        ]
        console.print(f"[dim]Using {len(prompts)} default prompts[/dim]")
    
    # Setup agent
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    agent = AgentWrapper(endpoint=agent_url, auth_header=headers, model=model)
    harness = ReliabilityHarness(agent)
    
    # Run tests
    with console.status("[bold green]Running tests..."):
        if test_type == "all":
            harness.run_all(prompts)
        else:
            harness.run_test(test_type, prompts)
    
    # Generate report
    report = harness.generate_report()
    
    # Display results
    _display_reliability_results(console, report)
    
    # Save if output specified
    if output:
        output.write_text(json.dumps(report, indent=2, default=str))
        console.print(f"\n[green]Report saved to:[/green] {output}")
    
    # Return exit code based on verdict
    if report.get("verdict") == "BLOCK":
        raise typer.Exit(1)


@app.command()
def audit(
    document: Path = typer.Argument(..., help="Document to audit"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", 
        help="Output file for report"),
    hop_complexity: str = typer.Option("1,2", "--hops", 
        help="Hop complexity levels (comma-separated: 1-4)"),
    api_key: Optional[str] = typer.Option(None, "--api-key", "-k", 
        help="API key for LLM (or set OPENAI_API_KEY)"),
    simple: bool = typer.Option(False, "--simple", "-s",
        help="Use simple mode (no LLM required)")
):
    """Audit documentation for flaws using adversarial questions."""
    from ..auditor.auditor import AdversarialAuditor
    from ..auditor.proposer import HopComplexity
    from ..core.agent_wrapper import OpenAIWrapper, AgentWrapper
    from ..core.models import AgentResponse
    
    console = get_console()
    console.print(f"[bold blue]Auditing document:[/bold blue] {document}")
    
    if not document.exists():
        console.print("[red]Error: Document not found[/red]")
        raise typer.Exit(1)
    
    content = document.read_text()
    console.print(f"[dim]Document size: {len(content)} characters[/dim]")
    
    if simple:
        # Simple mode - no LLM required
        console.print("[yellow]Running in simple mode (keyword-based)[/yellow]")
        
        # Create a mock agent for simple mode
        class MockAgent(AgentWrapper):
            def __init__(self):
                super().__init__(endpoint="mock://", model="simple-mode")
            def query(self, prompt: str, **kwargs) -> AgentResponse:
                return AgentResponse(content="mock", latency_ms=0, model="simple")
        
        auditor = AdversarialAuditor(proposer_model=MockAgent())

        console.print(
            "[yellow]⚠ --simple is DEMO-ONLY: keyword matching, not a reliability "
            "signal. Run with --api-key for a real audit.[/yellow]"
        )
        with console.status("[bold green]Running simple audit..."):
            report = auditor.audit_simple(content, document_name=document.name)
    else:
        # Full LLM mode
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            console.print("[red]Error: API key required (--api-key or OPENAI_API_KEY env)[/red]")
            console.print("[dim]Tip: Use --simple for keyword-based audit without API[/dim]")
            raise typer.Exit(1)
        
        model = OpenAIWrapper(api_key=key, model="gpt-4o-mini")
        
        # Parse hop complexity
        try:
            hops = [HopComplexity(int(h.strip())) for h in hop_complexity.split(",")]
        except ValueError:
            console.print("[red]Error: Invalid hop complexity (use 1-4)[/red]")
            raise typer.Exit(1)
        
        auditor = AdversarialAuditor(
            proposer_model=model,
            hop_complexity=hops
        )
        
        with console.status("[bold green]Generating adversarial questions and auditing..."):
            report = auditor.audit(content, document_name=document.name)
    
    # Display results
    _display_audit_results(console, report)
    
    # Save if output specified
    if output:
        report_dict = auditor.generate_report_dict(report)
        output.write_text(json.dumps(report_dict, indent=2))
        console.print(f"\n[green]Report saved to:[/green] {output}")
    
    # Return exit code based on score
    if report.overall_score < 0.5:
        raise typer.Exit(1)


@app.command("trust-eval")
def trust_eval(
    agent_url: str = typer.Option(..., "--agent", "-a", help="Agent endpoint URL"),
    knowledge_base: Path = typer.Option(..., "--kb", "-k", 
        help="Knowledge base directory or file"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", 
        help="Output file for report"),
    api_key: Optional[str] = typer.Option(None, "--api-key", 
        help="API key for agent and auditor")
):
    """Run combined trust evaluation on agent + knowledge base."""
    console = get_console()
    console.print("[bold blue]Running combined trust evaluation...[/bold blue]")
    console.print(f"  Agent: {agent_url}")
    console.print(f"  Knowledge Base: {knowledge_base}")
    
    # This combines both agent testing and doc auditing
    # For now, show guidance message
    console.print("\n[yellow]Combined trust evaluation runs both:[/yellow]")
    console.print("  1. Agent reliability tests (robustness, consistency, etc.)")
    console.print("  2. Knowledge base audit (documentation flaws)")
    console.print("\n[dim]Use 'arh test' and 'arh audit' separately for more control[/dim]")
    
    # TODO: Full implementation would combine both
    console.print("\n[bold]For full trust evaluation, run:[/bold]")
    console.print(f"  arh test --agent {agent_url}")
    if knowledge_base.is_file():
        console.print(f"  arh audit {knowledge_base}")
    else:
        console.print(f"  arh audit {knowledge_base}/*.md")


@app.command()
def version():
    """Show ARH version."""
    console = get_console()
    console.print("[bold]Agent Reliability Harness (ARH)[/bold]")
    console.print("Version: 0.1.0")
    console.print("https://github.com/akhilkinnera01/agent-reliability-harness")


def _display_reliability_results(console, report: dict):
    """Display reliability test results with Rich formatting."""
    from rich.table import Table
    from rich.panel import Panel
    
    # Overall score panel
    score = report.get("overall_score", 0)
    verdict = report.get("verdict", "UNKNOWN")
    color = "green" if verdict == "PASS" else "yellow" if verdict == "CONDITIONAL_PASS" else "red"
    
    console.print(Panel(
        f"[bold]Overall Score:[/bold] {score:.1%}\n"
        f"[bold]Verdict:[/bold] [{color}]{verdict}[/{color}]",
        title="Agent Reliability Results"
    ))
    
    # Dimension table
    table = Table(title="Reliability Dimensions")
    table.add_column("Dimension", style="cyan")
    table.add_column("Score", justify="center")
    table.add_column("Status", justify="center")
    
    for name, data in report.get("dimensions", {}).items():
        status_color = "green" if data.get("status") == "pass" else "red"
        table.add_row(
            name.title(),
            f"{data.get('score', 0):.1%}",
            f"[{status_color}]{data.get('status', 'unknown').upper()}[/{status_color}]"
        )
    
    console.print(table)
    
    # Show failures
    for name, data in report.get("dimensions", {}).items():
        failures = data.get("failures", [])
        if failures:
            console.print(f"\n[bold red]{name.title()} Failures:[/bold red]")
            for failure in failures[:5]:
                console.print(f"  • {failure}")


def _display_audit_results(console, report):
    """Display audit results with Rich formatting."""
    from rich.table import Table
    from rich.panel import Panel
    
    score = report.overall_score
    color = "green" if score > 0.8 else "yellow" if score > 0.5 else "red"
    
    console.print(Panel(
        f"[bold]Documentation Score:[/bold] [{color}]{score:.1%}[/{color}]\n"
        f"[bold]Findings:[/bold] {len(report.findings)}",
        title="Documentation Audit Results"
    ))
    
    if report.findings:
        table = Table(title="Findings")
        table.add_column("Line", justify="center", width=6)
        table.add_column("Flaw Type", style="cyan")
        table.add_column("Severity", justify="center")
        table.add_column("Question", width=40, no_wrap=True)
        
        severity_colors = {
            "critical": "red bold",
            "high": "red",
            "medium": "yellow",
            "low": "dim"
        }
        
        for finding in report.findings[:10]:
            sev_color = severity_colors.get(finding.severity.value, "white")
            question = finding.question[:37] + "..." if len(finding.question) > 40 else finding.question
            table.add_row(
                str(finding.line),
                finding.flaw_type.value,
                f"[{sev_color}]{finding.severity.value.upper()}[/{sev_color}]",
                question
            )
        
        console.print(table)
        
        if len(report.findings) > 10:
            console.print(f"[dim]... and {len(report.findings) - 10} more findings[/dim]")


def main():
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
