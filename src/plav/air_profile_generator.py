# upper_air_profile.py
from __future__ import annotations

import math
import argparse
from datetime import datetime, timezone
from typing import List, Dict
import requests
import numpy as np
import pandas as pd

API = "https://api.open-meteo.com/v1/gfs"  # GFS + HRRR (CONUS) blend

# Pressure levels available on this endpoint (hPa). (10–1000 hPa)
PLEV: List[int] = [
    1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 725, 700, 675, 650,
    625, 600, 575, 550, 525, 500, 475, 450, 425, 400, 375, 350, 325, 300, 275,
    250, 225, 200, 175, 150, 125, 100, 70, 50, 40, 30, 20, 15, 10
]

RD = 287.05  # dry-air gas constant [J/(kg·K)]

def _parse_time_utc(s: str) -> datetime:
    """Parse ISO8601. If no timezone, assume UTC."""
    dt = datetime.fromisoformat(s)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)

def _bearing_to_uv(dir_deg: float, spd_ms: float) -> tuple[float, float]:
    """Dir = FROM which it blows (met convention)."""
    r = math.radians(dir_deg)
    u = -spd_ms * math.sin(r)
    v = -spd_ms * math.cos(r)
    return u, v

def fetch_profile(lat: float, lon: float, when_utc: datetime) -> pd.DataFrame:
    """Return a vertical profile at (lat, lon) for the specified UTC hour."""
    # Snap to the model hour
    t_hour = when_utc.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hh = t_hour.strftime("%Y-%m-%dT%H:00")

    # Build hourly variable list across all pressure levels
    hourly_vars: List[str] = []
    for p in PLEV:
        hourly_vars += [
            f"geopotential_height_{p}hPa",
            f"temperature_{p}hPa",
            f"relative_humidity_{p}hPa",
            f"wind_speed_{p}hPa",
            f"wind_direction_{p}hPa",
        ]

    params = {
        "latitude": f"{lat:.5f}",
        "longitude": f"{lon:.5f}",
        "hourly": ",".join(hourly_vars),
        "timeformat": "unixtime",
        "start_hour": hh,
        "end_hour": hh,
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "cell_selection": "nearest",
        "elevation": "nan",   # disable downscaling; use grid-cell mean height
    }

    r = requests.get(API, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    hourly: Dict[str, list] = js["hourly"]

    if not hourly["time"]:
        raise RuntimeError("No data returned for requested hour — try an adjacent hour.")

    rows = []
    for p in PLEV:
        gh = float(hourly[f"geopotential_height_{p}hPa"][0])      # meters MSL
        tK = float(hourly[f"temperature_{p}hPa"][0]) + 273.15     # K
        rh = float(hourly[f"relative_humidity_{p}hPa"][0])        # %
        ws = float(hourly[f"wind_speed_{p}hPa"][0])               # m/s
        wd = float(hourly[f"wind_direction_{p}hPa"][0])           # deg (from)
        u, v = _bearing_to_uv(wd, ws)
        rho = (p * 100.0) / (RD * tK)                             # kg/m^3 (ideal gas)
        rows.append({
            "pressure_hPa": p,
            "geopotential_height_m": gh,
            "altitude_msl_m": gh,          # same as geopotential height here
            "temperature_K": tK,
            "relative_humidity_pct": rh,
            "wind_dir_deg": wd,
            "wind_speed_ms": ws,
            "u_ms": u,
            "v_ms": v,
            "density_kgm3": rho,
        })

    df = pd.DataFrame(rows).sort_values("pressure_hPa", ascending=False).reset_index(drop=True)
    df.attrs["requested_time_utc"] = t_hour.isoformat().replace("+00:00", "Z")
    df.attrs["grid_cell_elevation_m"] = float(js.get("elevation", float("nan")))
    df.attrs["source"] = "Open-Meteo GFS/HRRR pressure-level API"
    return df

def get_live_wind_profile(lat, lon, time):
    """gets the live wind profile for the specified location"""
    profile = fetch_profile(lat, lon, time)

    wind_alt_prof   = profile['altitude_msl_m'].to_numpy()
    wind_speed_prof = profile['wind_speed_ms'].to_numpy()
    wind_dir_prof   = profile['wind_dir_deg'].to_numpy()

    return wind_alt_prof, wind_speed_prof, wind_dir_prof

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fetch a one-time upper-air profile (GFS/HRRR).")
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--time", type=str, required=True, help="UTC time, e.g. 2025-08-29T20:00")
    ap.add_argument("--csv", type=str, default="", help="Optional output CSV path")
    args = ap.parse_args()

    when = _parse_time_utc(args.time)
    prof = fetch_profile(args.lat, args.lon, when)

    print(prof.head(12).to_string(index=False))
    print("\n…")
    print(prof.tail(12).to_string(index=False))

    if args.csv:
        prof.to_csv(args.csv, index=False)
        print(f"\nSaved CSV -> {args.csv}")
