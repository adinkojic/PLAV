"""Python Laptop Air Vehicles (PLAV) CLI module."""

from pathlib import Path

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
def init_sim():
    """initalize simulation, preparing the sim object"""

