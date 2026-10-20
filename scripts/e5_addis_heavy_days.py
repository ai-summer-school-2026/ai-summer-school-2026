"""Exercise 5: heavy rain days for Addis Ababa with threshold 20 mm/day."""
# %% imports and cleanup
import os

for k in [k for k in os.environ if k.startswith("AWS_")]:
    os.environ.pop(k)

import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import xarray as xr
import xclim

# Use python-workshop for the example file
WORKSHOP_DIR = pathlib.Path("/home/sayri/python-workshop/exercises")
sys.path.insert(0, str(WORKSHOP_DIR))
spec = importlib.util.spec_from_file_location("ex2", WORKSHOP_DIR / "02_read_realized.py")
src = open(spec.origin).read().split("t0 = time.time()")[0]
ns = {}
exec(src, ns)

# %% open data for Addis Ababa
ds = ns["open_sub"]("49r1-jja2025")
LAT, LON = 9.03, 38.74
THRESH_MM = 20  # mm/day

# %% select one forecast start time and compute daily totals
TIME_IDX = 45  # one representative forecast init time

init = pd.Timestamp(ds.time.values[TIME_IDX])
lead_h = (pd.to_timedelta(ds.step.values) / pd.Timedelta("1h")).astype(int)
day_idx = [i for i, h in enumerate(lead_h) if h % 24 == 0 and h > 0]

# Cumulative tp (m) -> daily mm, then count heavy days
cum = (ds["tp"].isel(time=TIME_IDX, step=day_idx)
       .sel(latitude=LAT, longitude=LON, method="nearest")
       .compute() * 1000.0).transpose("number", "step")
daily = cum.diff("step", label="upper")
daily = xr.concat([cum.isel(step=slice(0, 1)), daily], dim="step")

valid = init + pd.to_timedelta(lead_h[day_idx], unit="h")
pr = xr.DataArray(daily.values, dims=("number", "time"),
                  coords={"number": daily.number.values, "time": valid},
                  attrs={"units": "mm/day", "standard_name": "precipitation_flux"},
                  name="pr")

# %% count heavy rain days with xclim
heavy_days = xclim.indices.wetdays(pr, thresh=f"{THRESH_MM} mm/day", freq="YS")

# %% print result
n = pr.sizes["number"]
n_heavy = int((heavy_days.squeeze() >= 1).sum())
print(f"Addis Ababa ({LAT}, {LON}), init {init:%Y-%m-%d}, threshold {THRESH_MM} mm/day")
print(f"Members with at least one heavy-rain day: {n_heavy} of {n}")
