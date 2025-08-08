"""Python Laptop Air Vehicles (PLAV) CLI module."""

from pathlib import Path
import time

import typer

app = typer.Typer(help="PLAV Command Line Interface")

@app.callback(invoke_without_command=True)
def main(ctx: typer.Context,
        version: bool = typer.Option(False, "--version", help="Show version and exit"),):
    """main idk"""
    if version:
        typer.echo("PLAV version 0.1.0")
        raise typer.Exit()

    if ctx.invoked_subcommand is None:
        typer.echo("No subcommand provided. Use --help.")
        raise typer.Exit(code=1)

@app.command()
def test():
    """Run a test command."""
    typer.echo("This is a test command.")


@app.command()
def start_count():
    """Start the count at 0."""
    count = 0
    typer.echo(f"Starting count at {count}. Use 'plav inc-count' to increment.")

    bruh = 0

    while True:
        cmd = typer.prompt(">")

        if cmd in ("exit", "quit"):
            typer.echo("🛑 Count stopped.")
            break

        if cmd == "inc-count":
            count += 1
            typer.echo(f"Count incremented to {count}. {bruh}")
        bruh = bruh + 1

    
