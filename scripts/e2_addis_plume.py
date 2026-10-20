"""Exercise 2: Addis Ababa ensemble plume plot."""
# %% Imports and environment setup
import os
import pathlib
import time

for k in [k for k in os.environ if k.startswith("AWS_")]:
    os.environ.pop(k)

import icechunk
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr

ENDPOINT = "https://data.source.coop"
BUCKET = "e4drr-project"
BASE = "forecasts/ecmwf-ifs-ea-swio-realized-v3"
SUB = "49r1-jja2025"
LAT, LON = 9.03, 38.74  # Addis Ababa

# %% Open the store
def open_sub(sub):
    storage = icechunk.s3_storage(
        bucket=BUCKET, prefix=f"{BASE}/{sub}", endpoint_url=ENDPOINT, region="us-east-1",
        anonymous=True, from_env=False, force_path_style=True)
    cfg = icechunk.RepositoryConfig.default()
    cfg.manifest = icechunk.ManifestConfig(
        preload=icechunk.ManifestPreloadConfig(max_total_refs=0, max_arrays_to_scan=0))
    repo = icechunk.Repository.open(storage, config=cfg)
    session = repo.readonly_session("main")
    ds = xr.open_zarr(session.store, zarr_format=3, consolidated=False,
                      chunks={}, decode_timedelta=True)
    return ds

t0 = time.time()
ds = open_sub(SUB)
print(f"opened {SUB} in {time.time() - t0:.1f}s")
print("dims:", dict(ds.sizes), "| channels:", len(ds.data_vars))

# %% Select data: daily lead times only (0, 24, 48, ..., 240 h)
t1 = time.time()
# Select only daily steps (0 to 240 h in steps of 24 hours)
daily_steps = pd.to_timedelta(range(0, 241, 24), unit="h")
tp = (ds["tp"].isel(time=45).sel(step=daily_steps, method="nearest").sel(latitude=LAT, longitude=LON, method="nearest").compute()) * 1000.0
print(f"pulled {tp.shape} (members x steps) in {time.time() - t1:.1f}s")

# %% Check tp at 240h across members
tp_240 = tp.isel(step=-1)
print(f"tp at 240h: min {tp_240.min():.2f}, mean {tp_240.mean():.2f}, max {tp_240.max():.2f}")

# %% Compute 10-day totals from daily precipitation
lead = pd.to_timedelta(tp.step.values) / pd.Timedelta("1h")
lead_int = lead.astype(int)
# Daily precipitation from cumulative tp (difference between consecutive lead times)
daily_tp = tp.diff(dim="step")
# Prepend the first value (at step 0)
daily_tp = xr.concat([tp.isel(step=0).expand_dims("step"), daily_tp], dim="step")
# 10-day rolling sum (10 days = 10 lead times)
daily_tp_10 = daily_tp.rolling(step=10, center=False).sum()
max_10day = float(daily_tp_10.max())
print(f"largest 10-day total: {max_10day:.1f} mm")

# %% Plot ensemble plume (cumulative tp, not 10-day totals)
fig, ax = plt.subplots(figsize=(9, 4.5))
for m in range(tp.sizes["number"]):
    ax.plot(lead_int, tp.isel(number=m), color="steelblue", alpha=0.18, lw=0.8)
ax.plot(lead_int, tp.isel(number=0), color="black", lw=2, label="control")
ax.plot(lead_int, tp.mean("number"), color="crimson", lw=2, label="ensemble mean")
ax.set(xlabel="lead time (h)", ylabel="cumulative precipitation (mm)",
       title=f"ECMWF IFS ensemble tp at Addis Ababa, init {str(ds.time.values[45])[:10]}")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
out = pathlib.Path("out"); out.mkdir(exist_ok=True)
fig.savefig(out / "e2_addis_tp.png", dpi=120)
print("saved", out / "e2_addis_tp.png")
