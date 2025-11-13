# tests/test_compute_uv_on_rho.py
import numpy as np
import xarray as xr
import pytest

from helpers.importers import ROMSReader   # adjust import path
from .utils_dummy_roms import make_dummy_roms_ds

@pytest.fixture
def dummy_reader():
    # bypass __init__ so we don't open real files
    reader = ROMSReader.__new__(ROMSReader)
    reader.ds = make_dummy_roms_ds()
    return reader

def test_compute_uv_on_rho_shapes_and_nans(dummy_reader):
    reader = dummy_reader
    reader.compute_uv_on_rho()
    ds = reader.ds

    assert "u_rho" in ds and "v_rho" in ds and "uv_avail" in ds

    # correct dims
    assert ds.u_rho.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
    assert ds.v_rho.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")
    assert ds.uv_avail.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")

    # not all NaN inside ocean (ignore land corner)
    frac_u = float(np.isfinite(ds.u_rho.values).mean())
    frac_v = float(np.isfinite(ds.v_rho.values).mean())
    assert frac_u > 0.5
    assert frac_v > 0.5

    # uv_avail is boolean
    assert ds.uv_avail.dtype == bool
    frac_avail = float(ds.uv_avail.values.mean())
    assert 0.1 < frac_avail <= 1.0

def test_speed_from_compute_speed_on_rho(dummy_reader):
    reader = dummy_reader
    reader.compute_uv_on_rho()
    reader.compute_speed_on_rho()
    ds = reader.ds

    assert "speed_rho" in ds
    sp = ds.speed_rho

    # same dims as u_rho/v_rho
    assert sp.dims == ("ocean_time", "s_rho", "eta_rho", "xi_rho")

    # speed should be non-negative
    assert sp.min() >= 0

    # should have some finite values
    frac_sp = float(np.isfinite(sp.values).mean())
    assert frac_sp > 0.5
