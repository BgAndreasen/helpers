import pytest
import xarray as xr
from helpers.importers import ROMSReader

@pytest.fixture
def real_small_reader():
    reader = ROMSReader.__new__(ROMSReader)
    ds = xr.open_dataset("tests/data/roms_small.nc")
    reader.ds = ds
    return reader

def test_compute_uv_on_rho_on_real_subset(real_small_reader):
    reader = real_small_reader
    reader.compute_uv_on_rho()
    ds = reader.ds

    assert "u_rho" in ds and "v_rho" in ds
    assert ds.u_rho.dims == ds.v_rho.dims
    assert float(ds.u_rho.isnull().mean()) < 0.9  # not almost all NaN
