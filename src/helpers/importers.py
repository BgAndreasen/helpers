import xarray as xr
import numpy as np

def to_center_masked(da, stag_dim, out_dim, mask_face=None, mode="extend", fill_value=np.nan):
    """
    Average a staggered field to centers along one axis with land-aware, one-sided handling.

    - Land faces masked (mask_face==1 kept; others -> NaN) BEFORE averaging.
        - One-sided average if only one neighbor is wet; NaN if neither is wet.
        - `mode` controls only the *domain edges* (not land):
            'extend'   -> replicate edge values
            'fill'     -> constant pad with fill_value (e.g., 0.0)
            'periodic' -> wrap endpoints
        Returns: (centered_component, availability_mask_bool)
    """
    a = da if mask_face is None else da.where(mask_face == 1)

    # pad along the staggered axis (domain edges only; won't cross land)
    if mode == "extend":
        p = a.pad({stag_dim: (1, 1)}, mode="edge")
    elif mode == "fill":
        p = a.pad({stag_dim: (1, 1)}, mode="constant", constant_values=fill_value)
    elif mode == "periodic":
        p = xr.concat([a.isel({stag_dim: -1}), a, a.isel({stag_dim: 0})], dim=stag_dim)
    else:
        raise ValueError("mode must be 'extend' | 'fill' | 'periodic'")

    left  = p.isel({stag_dim: slice(0, -1)})
    right = p.isel({stag_dim: slice(1,  None)})

    nwet = (~xr.ufuncs.isnan(left)).astype(int) + (~xr.ufuncs.isnan(right)).astype(int)
    s = xr.where(xr.ufuncs.isnan(left), 0, left) + xr.where(xr.ufuncs.isnan(right), 0, right)
    comp = xr.where(nwet > 0, s / xr.where(nwet == 0, 1, nwet), np.nan)  # safe divide

    # rename staggered axis to center axis on BOTH outputs
    comp = comp.rename({stag_dim: out_dim})
    avail = (nwet > 0).rename({stag_dim: out_dim}).astype(bool)
    
    return comp, avail

def safe_rename(da, mapping):
    """Rename only the dims that actually exist on the DataArray."""
    mapping = {k: v for k, v in mapping.items() if k in da.dims}
    return da.rename(mapping)

def rotate_uv_to_geograpthic(U, V, angle):
    ang = angle
    if "units" in ang.attrs and "deg" in ang.attrs["units"].lower():
        ang = xr.apply_ufunc(np.deg2rad, ang)
        
    # Rotate grid (ξ,η) -> geographic (E,N)
    # sometimes this is rotated differently, just be avare!!
    # TODO: is there a way to make sure this doesn't get buggered??
    Ue = U*np.cos(ang) - V*np.sin(ang)   # eastward
    Vn = U*np.sin(ang) + V*np.cos(ang)   # northward
    return(Ue, Vn)

def calc_principal_heading(u, v, tidal_mode=True):
    """
    This is an adaption of the function in the MHKit python package,
    https://github.com/MHKiT-Software/MHKiT-Python

    u, v : 2D numpy arrays with shape (time, depth)
    Returns:
      1D numpy array with shape (depth,)
    """
    # combine into complex
    dt = u + 1j * v

    if tidal_mode:
        # flip negative imag
        mask_neg = np.imag(dt) <= 0
        dt = np.where(mask_neg, -dt, dt)

        # double angles
        angles = np.angle(dt)
        dt = dt * np.exp(1j * angles)

        # mask invalid
        dt = np.where(np.isfinite(dt), dt, np.nan)

        # mean over time‑axis (axis=0)
        mean_dt = np.nanmean(dt, axis=0)
        pang = np.angle(mean_dt) / 2
    else:
        mean_dt = np.nanmean(dt, axis=0)
        pang = np.angle(mean_dt)

    # to degrees, 0–360
    p_heading = (np.rad2deg(pang) % 360).round(4)
    return p_heading.astype(np.float32)

class ROMSReader:
    """
    A class to load ROMS output, compute Z depths, interpolate to fixed Z depths,
    and derive horizontal speed with a single high-level process method.
    """
    def __init__(self, files, **open_kwargs):
        """
        Initialize by opening and merging ROMS NetCDF files.

        Parameters
        ----------
        files : str or list
            Glob pattern or list of ROMS file paths.
        open_kwargs : dict
            Passed to xr.open_mfdataset (e.g. chunks, decode_times).
        """
        self.ds = xr.open_mfdataset(
            files,
            combine="nested",
            concat_dim="ocean_time",
            coords="minimal",
            data_vars="minimal",
            compat="override",
            **open_kwargs
        )
    
    def compute_uv_on_rho(self, mode_x="extend", mode_y="extend"):
        """
        Compute u and v on rho points with mask-aware, one-sided interpolation,
        then (optionally) |U|, and write back into self.ds.

        A rho cell is considered valid if *either* component is available,
        then speed uses the available components (NaN only if neither exists).
        """
        ds = self.ds

        u = self.ds["u"]
        v = self.ds["v"]
        mu = self.ds.get("mask_u")
        mv = self.ds.get("mask_v")
        mr = self.ds.get("mask_rho")

        # u -> rho (along xi)
        u_rho, u_av = to_center_masked(u, "xi_u", "xi_rho", mask_face=mu, mode=mode_x)
        # align the other horizontal dim to eta_rho if needed
        u_rho = safe_rename(u_rho, {"eta_u": "eta_rho"})
        u_av  = safe_rename(u_av,  {"eta_u": "eta_rho", "xi_u": "xi_rho"})

        # v -> rho (along eta)
        v_rho, v_av = to_center_masked(v, "eta_v", "eta_rho", mask_face=mv, mode=mode_y)
        v_rho = safe_rename(v_rho, {"xi_v": "xi_rho"})
        v_av  = safe_rename(v_av,  {"xi_v": "xi_rho", "eta_v": "eta_rho"})
        
        # enforce rho mask but don't lose cells where only one component exists
        avail = u_av | v_av
        if mr is not None:
            avail = avail & (mr == 1)

        u_rho = u_rho.where(u_av & avail)
        v_rho = v_rho.where(v_av & avail)

        # attrs for clarity
        u_rho.name = "u_rho"
        v_rho.name = "v_rho"
        
        # write back
        ds[u_rho.name] = u_rho
        ds[v_rho.name] = v_rho
        ds["uv_avail"] = avail

        # should the u_rho and v_rho, just be rotated here?
        ds[u_rho.name], ds[v_rho.name] = rotate_uv_to_geograpthic(
                U = ds[u_rho.name], V = ds[v_rho.name], angle = ds.angle
            )
        
        ds.u_rho.attrs.update({
            "long_name": "u velocity on rho points, rotated to geographic (E,N)",
            "units": u.attrs.get("units", "")
            })
        ds.v_rho.attrs.update({
            "long_name": "v velocity on rho points, rotated to geographic (E,N)",
            "units": v.attrs.get("units", "")
            })

        self.ds = ds
        return self

    
    def compute_speed_on_rho(self):
        
        ds = self.ds

        def _compute(U, V, name, mask=None):
            sp = (U.fillna(0.0)**2 + V.fillna(0.0)**2) ** 0.5
            if mask is not None:
                sp = sp.where(mask)
            sp.name = name

            unit = U.attrs.get('units','') if U.attrs.get('units')==V.attrs.get('units') else ''
            sp.attrs['units'] = unit
            sp.attrs['long_name'] = 'horizontal speed at rho points'
            return sp
        
        if 'u_rho' in ds and 'v_rho' in ds:
            mask = ds.get("uv_avail", None)
            ds['speed_rho'] = _compute(ds.u_rho, ds.v_rho, 'speed_rho', mask)

        if 'u_rho_z' in ds and 'v_rho_z' in ds:
            mask = ds.get("uv_avail", None)
            ds['speed_rho_z'] = _compute(ds.u_rho_z, ds.v_rho_z, 'speed_rho_z', mask)

        self.ds = ds
        return self


    def add_z_rho(self):
        """
        Compute mid-layer depths z_rho and attach as coordinate.
        """
        ds = self.ds
        h = ds.h
        zeta = ds.zeta
        hc = ds.hc
        sc = ds.s_rho
        Cs = ds.Cs_r
        Vt = int(ds.Vtransform.item())
        if Vt == 1:
            z0 = (hc*sc + h*Cs) / (hc + h)
        else:
            z0 = (hc*(sc - Cs) + h*Cs) / (hc + h)
        z = zeta + (zeta + h) * z0
        z = z.transpose("ocean_time","s_rho","eta_rho","xi_rho").rename("z_rho")
        self.ds = ds.assign_coords(z_rho=z)
        return self

    def add_z_w(self):
        """
        Compute interface depths z_w and attach as coordinate.
        """
        ds = self.ds
        h = ds.h
        zeta = ds.zeta
        hc = ds.hc
        sw = ds.s_w
        Csw = ds.Cs_w
        Vt = int(ds.Vtransform.item())
        if Vt == 1:
            z0 = (hc*sw + h*Csw) / (hc + h)
        else:
            z0 = (hc*(sw - Csw) + h*Csw) / (hc + h)
        z = zeta + (zeta + h) * z0
        z = z.transpose("ocean_time","s_w","eta_rho","xi_rho").rename("z_w")
        self.ds = ds.assign_coords(z_w=z)
        return self

    def interp_to_fixed_depths(self, var_list, z_new):
        """
        Interpolate variables from sigma-levels onto fixed-depth array z_new.

        Parameters
        ----------
        var_list : list of str
            Variables to remap (e.g. ['temp','u_rho']).
        z_new : array-like
            1D target depths (e.g. np.arange(0,-h_max,-10)).
        z_coord : str
            Depth coordinate name ('z_rho' or 'z_w').
        s_coord : str
            Matching sigma-dim name ('s_rho' or 's_w').

        Returns
        -------
        self, with new DataArrays var+'_z' on dims (ocean_time, z, eta_rho, xi_rho).
        """
        ds = self.ds
        z_arr = np.asarray(z_new)
        z_da = xr.DataArray(z_arr, dims=["z"], coords={"z": z_arr})

        def _interp(profile, z0, zt):
            return np.interp(zt, z0, profile, left=np.nan, right=profile[-1])

        for var in var_list:
            da = ds[var]
            
            # detect sigma dimension
            if 's_rho' in da.dims:
                s_dim, z_coord = "s_rho", "z_rho"
            elif 's_w' in da.dims:
                s_dim, z_coord = "s_w", "z_w"
            else:
                raise ValueError(f"Variable {var!r} has no sigma dimension")
            
            da = da.chunk({s_dim : -1})

            da_z = xr.apply_ufunc(
                _interp,
                da, ds[z_coord], z_da,
                input_core_dims=[[s_dim],[s_dim],["z"]],
                output_core_dims=[["z"]],
                vectorize=True, 
                dask="parallelized",
                dask_gufunc_kwargs={"output_sizes":{"z": z_arr.size}},
                output_dtypes=[da.dtype]
            )

            # assign coords and transpose
            da_z = (
                da_z
                .assign_coords(z=("z", z_arr))
                .transpose("ocean_time","z","eta_rho","xi_rho")
                .rename(f"{var}_z")
            )
            # re‐copy attrs from the original
            da_z.attrs = da.attrs
            ds[var + "_z"] = da_z

        self.ds = ds
        return self


    def process(self, var_list, z_new):
        """
        High-level pipeline: compute z_rho, z_w, u/v on rho-grid,
        interpolate to fixed depths, and add speed.

        Returns
        -------
        xarray.Dataset
        """
        return (
            self
            .add_z_rho()
            .add_z_w()
            .compute_uv_on_rho()
            .interp_to_fixed_depths(var_list, z_new)
            .compute_speed_on_rho()
            .ds
        )
    
    def denser(self, var_list, factor=2):
        """interpolate to denser grid"""

        ds = self.ds

        # factor=2 means ~2x finer in each horizontal direction
        factor = 2
        M = ds.sizes["xi_rho"]
        N = ds.sizes["eta_rho"]

        xi_fine  = np.linspace(0, M-1, M*factor)
        eta_fine = np.linspace(0, N-1, N*factor)
        
        u_fine = ds.u_rho.interp(xi_rho=xi_fine, eta_rho=eta_fine, method="linear")
        v_fine = ds.v_rho.interp(xi_rho=xi_fine, eta_rho=eta_fine, method="linear")

