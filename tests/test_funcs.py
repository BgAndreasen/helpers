import numpy as np
from .utils_dummy_roms import make_dummy_roms_ds

def make_dummy_roms_ds_one_component_missing():
    ds = make_dummy_roms_ds()
    # wipe out v over half the domain to simulate narrow fjord missing v
    v = ds["v"].values
    v[..., :, : v.shape[-1] // 2] = np.nan
    ds["v"] = (ds["v"].dims, v)
    return ds

def test_speed_uses_single_component_when_other_missing(monkeypatch):
    from helpers import importers

    # bypass __init__
    reader = importers.ROMSReader.__new__(importers.ROMSReader)
    reader.ds = make_dummy_roms_ds_one_component_missing()

    reader.compute_uv_on_rho()
    reader.compute_speed_on_rho()
    ds = reader.ds

    sp = ds.speed_rho
    u = ds.u_rho
    v = ds.v_rho

    # where v is NaN but u is finite, speed should equal |u|
    mask = np.isfinite(u.values) & np.isnan(v.values)
    assert np.allclose(
        sp.values[mask],
        np.abs(u.values[mask]),
        atol=1e-6,
    )

def test_compute_uv_on_rho_on_real_subset(real_small_reader):
    reader = real_small_reader
    reader.compute_uv_on_rho()
    ds = reader.ds

    assert "u_rho" in ds and "v_rho" in ds
    assert ds.u_rho.dims == ds.v_rho.dims
    assert float(ds.u_rho.isnull().mean()) < 0.9  # not almost all NaN