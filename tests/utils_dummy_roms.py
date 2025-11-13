import numpy as np
import xarray as xr

def make_dummy_roms_ds(
    nx_rho=6, ny_rho=5, nz=3, nt=4,
    add_masks=True,
):
    """
    Make a tiny, self-consistent ROMS-like C-grid dataset:

    - u: (time, s_rho, eta_u, xi_u)
    - v: (time, s_rho, eta_v, xi_v)
    - angle: (eta_rho, xi_rho)
    - optional masks: mask_u, mask_v, mask_rho
    """

    # dims
    xi_rho  = np.arange(nx_rho)
    eta_rho = np.arange(ny_rho)
    s_rho   = np.arange(nz)
    time    = np.arange(nt)

    xi_u  = np.arange(nx_rho - 1)   # u staggered in xi
    eta_u = np.arange(ny_rho)

    xi_v  = np.arange(nx_rho)
    eta_v = np.arange(ny_rho - 1)   # v staggered in eta

    # coords
    coords = {
        "ocean_time": ("ocean_time", time),
        "s_rho":      ("s_rho", s_rho),
        "xi_rho":     ("xi_rho", xi_rho),
        "eta_rho":    ("eta_rho", eta_rho),
        "xi_u":       ("xi_u", xi_u),
        "eta_u":      ("eta_u", eta_u),
        "xi_v":       ("xi_v", xi_v),
        "eta_v":      ("eta_v", eta_v),
    }

    # simple lon/lat fields (not critical, but nice to have)
    lon_rho, lat_rho = np.meshgrid(xi_rho * 0.01 - 7.0,
                                   eta_rho * 0.01 + 62.0)
    angle = np.zeros_like(lon_rho)  # no rotation for dummy

    # build u, v patterns that are NOT NaN
    # u(time, s_rho, eta_u, xi_u)
    tt_u, zz_u, yy_u, xx_u = np.meshgrid(
        time, s_rho, eta_u, xi_u, indexing="ij"
    )
    u_data = (
        0.1 * xx_u + 0.01 * yy_u + 0.5 * zz_u
        + np.sin(tt_u / 2.0)
    )

    # v(time, s_rho, eta_v, xi_v)
    tt_v, zz_v, yy_v, xx_v = np.meshgrid(
        time, s_rho, eta_v, xi_v, indexing="ij"
    )
    v_data = (
        -0.05 * xx_v + 0.02 * yy_v + 0.3 * zz_v
        + np.cos(tt_v / 3.0)
    )

    data_vars = {
        "u": (("ocean_time", "s_rho", "eta_u", "xi_u"), u_data),
        "v": (("ocean_time", "s_rho", "eta_v", "xi_v"), v_data),
        "angle": (("eta_rho", "xi_rho"), angle),
        "lon_rho": (("eta_rho", "xi_rho"), lon_rho),
        "lat_rho": (("eta_rho", "xi_rho"), lat_rho),
    }

    if add_masks:
        mask_u = np.ones((ny_rho, nx_rho - 1), dtype="i1")
        mask_v = np.ones((ny_rho - 1, nx_rho), dtype="i1")
        mask_rho = np.ones((ny_rho, nx_rho), dtype="i1")

        # optionally poke some land at one corner
        mask_u[0, 0] = 0
        mask_v[-1, -1] = 0
        mask_rho[0, 0] = 0

        data_vars["mask_u"] = (("eta_u", "xi_u"), mask_u)
        data_vars["mask_v"] = (("eta_v", "xi_v"), mask_v)
        data_vars["mask_rho"] = (("eta_rho", "xi_rho"), mask_rho)

    ds = xr.Dataset(data_vars=data_vars, coords=coords)
    return ds
