import xarray as xr
import numpy as np

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

    def add_uv_to_rho_grid(self):
        """
        Interpolate U and V from their native C-grid onto the rho-grid.
        Produces u_rho and v_rho on dims (ocean_time, s_rho, eta_rho, xi_rho)
        and drops original C-grid coordinates
        """
        ds = self.ds
        u_rho = ds.u.interp(eta_u=ds.eta_rho, xi_u=ds.xi_rho)
        v_rho = ds.v.interp(eta_v=ds.eta_rho, xi_v=ds.xi_rho)

        # assign and drop old C-grid coords
        ds = ds.assign(u_rho=u_rho, v_rho=v_rho)
        ds = ds.reset_coords(['eta_u','xi_u','eta_v','xi_v'], drop=True)
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

    def add_speed(self):
        """
        Add horizontal speed on sigma and z-grids when u_rho/v_rho or u_rho_z/v_rho_z exist.
        """
        ds = self.ds
        def _compute(U, V, name):
            sp = np.sqrt(U**2 + V**2)
            unit = U.attrs.get('units','') if U.attrs.get('units')==V.attrs.get('units') else ''
            sp = sp.rename(name)
            sp.attrs['units'] = unit
            sp.attrs['long_name'] = 'horizontal speed'
            return sp
        if 'u_rho' in ds and 'v_rho' in ds:
            ds['speed_rho'] = _compute(ds.u_rho, ds.v_rho, 'speed_rho')
        if 'u_rho_z' in ds and 'v_rho_z' in ds:
            ds['speed_rho_z'] = _compute(ds.u_rho_z, ds.v_rho_z, 'speed_rho_z')
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
            .add_uv_to_rho_grid()
            .interp_to_fixed_depths(var_list, z_new)
            .add_speed()
            .ds
        )
