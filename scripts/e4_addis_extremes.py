"""Exercise 4: explore 27 years of observed rainfall extremes (CMORPH, 1998-2024) for Addis Ababa.
"""
import os
import pathlib
import time

# Drop stray AWS_* variables that would redirect the open.
for k in [k for k in os.environ if k.startswith("AWS_")]:
    os.environ.pop(k)

import icechunk
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

ENDPOINT = "https://data.source.coop"
BUCKET = "e4drr-project"
PREFIX = "observations/cmorph_rp_icechunk"
DURATION = "24hr"
LAT, LON = 9.03, 38.74  # Addis Ababa
out = pathlib.Path("out")
out.mkdir(exist_ok=True)

storage = icechunk.s3_storage(
    bucket=BUCKET, prefix=PREFIX, endpoint_url=ENDPOINT, region="us-east-1",
    anonymous=True, from_env=False, force_path_style=True)
repo = icechunk.Repository.open(storage, config=icechunk.RepositoryConfig.default())
ds = xr.open_zarr(repo.readonly_session("main").store, zarr_format=3, consolidated=False, chunks={})

print("dims:", dict(ds.sizes))
print("variables:", list(ds.data_vars))
print("durations:", ds.duration.values.tolist(), "| return periods:", ds.return_period.values.tolist())

# Point series read: 27 annual maxima at Addis Ababa.
t0 = time.time()
amax = ds["annual_maxima"].sel(duration=DURATION).sel(lat=LAT, lon=LON, method="nearest").compute()
rp_point = ds["return_period_precip"].sel(duration=DURATION).sel(lat=LAT, lon=LON, method="nearest").compute()
print(f"series read: {time.time() - t0:.1f}s, {ds.sizes['year']} chunks")

vals = amax.values
assert np.isfinite(vals).all(), "NaN in annual maxima: check the point is inside the domain"
print(f"Addis Ababa {DURATION} annual max: median {np.median(vals):.0f} mm, wettest {vals.max():.0f} mm "
      f"in {int(amax.year[vals.argmax()])}")

# Get 2-, 10-, and 50-year levels
rp_vals = dict(zip(rp_point.return_period.values.tolist(), np.round(rp_point.values, 1).tolist()))
print(f"return levels (mm): 2-yr={rp_vals.get(2, 'N/A')}, 10-yr={rp_vals.get(10, 'N/A')}, 50-yr={rp_vals.get(50, 'N/A')}")

# Plot: annual maxima with return levels
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.bar(amax.year, vals, color="steelblue")
for T, c in [(2, "grey"), (10, "darkorange"), (50, "crimson")]:
    ax.axhline(float(rp_point.sel(return_period=T)), color=c, ls="--", lw=1.5, label=f"{T}-yr level")
ax.set(title=f"Addis Ababa: annual max {DURATION} rainfall, 1998-2024", xlabel="year", ylabel="mm")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(out / "e4_addis_extremes.png", dpi=120)
print("saved", out / "e4_addis_extremes.png")
