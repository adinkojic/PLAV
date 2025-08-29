"""Python Laptop Air Vehicles (PLAV) CLI module."""

from pathlib import Path
import time
import threading

import typer
from typing_extensions import Annotated

from plav.plav import Plav
from plav.wind_tunnel import WindTunnel

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
def wind_tunnel(scenario_name):
    """Run the interactive wind tunnel"""

    print("Starting Wind Tunnel with " + scenario_name)
    if ".json" not in scenario_name:
        scenario_name = scenario_name + ".json "

    tunnel = WindTunnel(scenario_name)
    tunnel.solve_forces(quiet = True) #pump once

    try:
        while True:
            line = typer.prompt(">")
            parts = line.strip().split()
            if not parts:
                continue
            cmd, *rest = parts

            try:
                if cmd in ("exit", "quit"):
                    typer.echo("Shutdown Wind Tunnel")
                    break
                if cmd in ("help"):
                    typer.echo("Available commands:")
                    typer.echo("  exit, quit: Shutdown Wind Tunnel")
                    typer.echo("  alpha <value>: Set angle of attack")
                    typer.echo("  beta <value>: Set sideslip angle")
                    typer.echo("  airspeed <value>: Set airspeed")
                    typer.echo("  trim <value>: Trim out the vehicle at the specified airspeed")
                    typer.echo("  density: Print air density")
                    typer.echo("  solve: Solve for forces and coefficients at current conditions")
                    typer.echo("  aileron <value>: Set aileron deflection")
                    typer.echo("  elevator <value>: Set elevator deflection")
                    typer.echo("  rudder <value>: Set rudder deflection")
                    typer.echo("  throttle <value>: Set throttle")
                    typer.echo("  realistic_mixing: Use realistic mixing for control surfaces")
                    typer.echo("  plav_mixing: Use PLAV mixing for control surfaces")
                if cmd in ("alpha"):
                    tunnel.change_alpha(float(rest[0]))
                if cmd in ("beta"):
                    tunnel.change_beta(float(rest[0]))
                if cmd in ("airspeed"):
                    tunnel.change_airspeed(float(rest[0]))
                if cmd in ("altitude"):
                    tunnel.change_altitude(float(rest[0]))
                if cmd in ("solve"):
                    tunnel.solve_forces()
                if cmd in ("reload"):
                    tunnel.reload_vehicle()
                    typer.echo("Vehicle reloaded")
                if cmd in ("trim"):
                    tunnel.trim_out()
                if cmd in ("density"):
                    tunnel.print_density()
                if cmd in ("aileron"):
                    tunnel.set_aileron(float(rest[0]))
                if cmd in ("elevator"):
                    tunnel.set_elevator(float(rest[0]))
                if cmd in ("rudder"):
                    tunnel.set_rudder(float(rest[0]))
                if cmd in ("throttle"):
                    tunnel.set_throttle(float(rest[0]))
                if cmd in ("realistic_mixing"):
                    tunnel.use_realistic_mixing()
                if cmd in ("plav_mixing"):
                    tunnel.use_plav_mixing()
                if cmd in ("balloon_diameter"):
                    typer.echo(f"Balloon Diameter: {tunnel.get_balloon_diameter()} ft")
                if cmd in ("echo"):
                    typer.echo("echo")
                    continue
            except IndexError:
                typer.echo("Missing argument value")
    except KeyboardInterrupt:
        pass
    finally:
        pass

@app.command()
def test():
    """test"""
    typer.echo('hello')

@app.command()
def offline_sim(scenario_name,
            no_gui: Annotated[bool, typer.Option("--nogui")] = False,
            output_file_name = "output.csv"):
    """Runs a hard simulation"""
    if ".json" not in scenario_name:
        scenario_name = scenario_name + ".json "


    typer.echo("Starting scenario " + scenario_name)
    plav_obj = Plav(scenario_name,[0,1500], no_gui = no_gui)

@app.command()
def sitl_sim(scenario_name,
            ardupilot_ip = "0.0.0.0",
            no_gui: Annotated[bool, typer.Option("--nogui")] = False,
            output_file_name = "output.csv"):
    """Simulates the Vehicle with ArduPilot's SITL"""
    if ".json" not in scenario_name:
        scenario_name = scenario_name + ".json "


    typer.echo("Starting scenario " + scenario_name)
    plav_obj = Plav(scenario_name,[0,0.01], no_gui = no_gui, real_time=True,
                    use_sitl=True, ardupilot_ip = ardupilot_ip)


@app.command()
def interactive_sim(scenario_name, output_file_name = "output.csv"):
    """Runs an interactive sim. Can reload scenarios, change vehicles, etc"""

    if ".json" not in scenario_name:
        scenario_name = scenario_name + ".json "

    #launch sim instance
    sim_args = {
        "scenario_file":scenario_name,
        "timespan": [0,30],
        "real_time": False,
        "no_gui": False,
        "export_to_csv": True
    }

    plav_obj = Plav(scenario_name,[0,30])

    try:
        while True:
            line = typer.prompt(">")
            parts = line.strip().split()
            if not parts:
                continue
            cmd, *rest = parts

            if cmd in ("exit", "quit"):
                typer.echo("🛑 Simulator terminated.")
                break
            if cmd in ("togglepause"):
                plav_obj.toggle_pause()
            if cmd in ("echo"):
                typer.echo("echo")
                continue
    except KeyboardInterrupt:
        pass
    finally:
        pass
