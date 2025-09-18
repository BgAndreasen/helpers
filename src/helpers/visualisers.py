import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import xarray as xr
import numpy as np
import geopandas as gpd
import cmocean
from IPython.display import HTML, display
import warnings

def cm2inch(*tupl):
    """convert centimeters to inches."""
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)

def annotate_model_setup_info(ax, fontsize=8, **kwargs):
    """
    Annotate an axis with model setup information
    """

    text = (
        f"Model setup:"
        f"\nTitle: {kwargs['attr_title']}"
        f"\nFile: {kwargs['attr_file']}"
        f"\nDepth parameter: {kwargs['depth_var']}, mean: {kwargs['mean_depth']} [{kwargs['min_depth']}, {kwargs['max_depth']}]"
    )
    ax.annotate(
        text=text,
        xy=(0, -0.05), 
        xycoords=('axes fraction', 'axes fraction'),
        color='grey',
        size=fontsize,
        va='top'
        )

def add_model_setup_info(xpos, ypos, fontsize=8, **kwargs):
    plt.rc('text', usetex=True)
    plt.gcf().text(x = xpos, y = ypos,
                s = (
                   f"Model setup:"
                   f"\nTitle: {kwargs['attr_title']}"
                   f"\nFile: {kwargs['attr_file']}"
                   f"\nAnimation setup: s_rho: {kwargs['mean_s_rho']}, mean z: {kwargs['mean_z_rho']} m"
                   ),
               fontsize = fontsize,
               color = "grey")

def annotate_dataowner(text, ax, **kwargs):
    ax.annotate(
        text = text,
        xy=(0.95, 0.05), 
        xycoords=('axes fraction', 'axes fraction'),
        fontsize=10, color='grey', 
        ha = 'right',
        va = 'bottom',
        alpha=0.5,
        zorder = 70)

class farcoast_anno_text:
    """
    Stores annotation metadata for FarCoast figures.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
    
    def add(self, **kwargs):
        self.__dict__.update(kwargs)


def annotate_watermark(text, ax, **kwargs):
    """
    Annotate a watermark on the axis.
    """
    ax.annotate(
        text = text,
        xy=(0.05, 0.05), 
        xycoords=('axes fraction', 'axes fraction'),
        fontsize=30, 
        color='grey', 
        alpha=0.5,
        zorder = 70,
        rotation=45)



def annotate_title(ax, fontsize=8, **kwargs):
    text = (
        f"Research projects: ADepoPlan and FMHAT"
        f"\n{kwargs['start_hour']} "
        f"to {kwargs['end_hour']}"
    )
    ax.annotate(text=text,
            xy=(0.01, 0.99), 
            xycoords=('axes fraction', 'axes fraction'),
            color = 'dimgrey',
            zorder = 60,
            size=fontsize,
            #style='italic',
            va='top'
            )




class farcoast_animation:
    """
    Creates animations from FarCoast modelling data.

    Parameters:
    ds (xarray ds): dataset from farcoast_import
    var_animate (str): variable to animate
    var_percentile (int): percentage of variable values shown in plots, helpful if there is a blow up in the model

    Returns:
    Figure Animation
    
    """

    def __init__(
        self, 
        var_animate,
        farcoast_ds = None, #backwards compatability, I was an IDIOT!!!
        ds = None,
        var_percentile = 100,
        var_max = None,
        first_time_frame = 4, 
        final_time_frame = None, # None, to use max
        depth_slice = [None, None],
        boundary = None,
        quiver_every_x_box = 2, 
        quiver_scale = 5,
        quiver_max = None,
        quiver_normalise = False, # not implemented
        dpi = 150,
        fps = 12, 
        write_output = True,
        watermark = True,
        output_type = "gif",
        save_folder = "gifs/",
        bathy_path = "../data/geodata/dybsimplified/",
        coastline_path = "../data/geodata/lendiskort.gdb/",
        stations = None,
        null_proj = False,
        **kwargs
        ):

        # if someone still uses farcoast_ds it will be changed
        if farcoast_ds is not None:
            warnings.warn(
                "'farcoast_ds' is deprecated; please use 'ds' instead",
                DeprecationWarning,
                stacklevel=2
            )

        if (ds is None) == (farcoast_ds is None):
            raise TypeError("Exactly one of 'ds' or 'farcoast_ds' must be provided")

        ds = ds if ds is not None else farcoast_ds

        # stashing block, to make cloning easier
        self._init_kwargs = dict(
            ds=ds,
            var_animate=var_animate,
            var_percentile=var_percentile,
            var_max=var_max,
            first_time_frame=first_time_frame,
            final_time_frame=final_time_frame,
            depth_slice=depth_slice,
            boundary=boundary,
            quiver_every_x_box=quiver_every_x_box,
            quiver_scale=quiver_scale,
            quiver_max=quiver_max,
            quiver_normalise=quiver_normalise,
            dpi=dpi,
            fps=fps,
            write_output=write_output,
            watermark=watermark,
            output_type=output_type,
            save_folder=save_folder,
            bathy_path=bathy_path,
            coastline_path=coastline_path,
            null_proj=null_proj,
            stations=stations,
            **kwargs
        )

        # assign to self
        self.__dict__.update(self._init_kwargs)

        # add
        self.fig = None
        self.ax = None
        self.ds_mean = None
        self.resample = None
        self.image = None
        self.arrows = None
        self.animation = None

        # the projection the data is in
        # TODO: find the projection of the ROMS output!!
        epsg_org = 4326
        self.map_proj_original = ccrs.PlateCarree()
        # the projection we want to transform to
        epsg = 5316 

        # TODO: get this out of function ;)
        if boundary is None:
            fjord_ymax, fjord_xmin = (self.ds.lon_rho.min(), self.ds.lat_rho.min())
            fjord_ymin, fjord_xmax = (self.ds.lon_rho.max(), self.ds.lat_rho.max())
            use_manual_boundaries = False
        else:
            use_manual_boundaries = True
        
            if boundary == "VES":
                # Vestmanna boundary
                fjord_ymax, fjord_xmin = (62.156740, -7.197381)
                fjord_ymin, fjord_xmax = (62.134209, -7.140186)
            elif boundary == "VESsund":
                # Vestmanna sund
                fjord_ymax, fjord_xmin = (62.16, -7.24)
                fjord_ymin, fjord_xmax = (62.10, -7.12)
            elif boundary == "GOT":
                # Gøtuvík boundary
                fjord_ymax, fjord_xmin = (62.220, -6.76)
                fjord_ymin, fjord_xmax = (62.120, -6.5)
            elif boundary == "GOTzoom":
                # Gøtuvík boundary
                fjord_ymax, fjord_xmin = (62.20, -6.76)
                fjord_ymin, fjord_xmax = (62.14, -6.63)
            else:
                raise Exception(f'{self.boundary} is not defined in function')
        
        # read gdb and shape files
        self.geo_coastline = gpd.read_file(self.coastline_path, layer = 'oyggjar')
        self.geo_bathy = gpd.read_file(self.bathy_path, layer = 'dyb')
        if self.stations is not None:
            self.stations = gpd.GeoDataFrame(
                self.stations,
                geometry=gpd.points_from_xy(self.stations.lon, self.stations.lat),
                crs="EPSG:4326"
                )

        # make sure everything is in the same projection
        if self.null_proj:
            self.map_proj = self.map_proj_original
            map_proj_transform_label = f"EPSG {epsg}"
            self.geo_coastline = self.geo_coastline.to_crs(epsg=epsg_org)
            self.geo_bathy = self.geo_bathy.to_crs(epsg=epsg_org)
            if self.stations is not None:
                self.stations = self.stations.to_crs(epsg=epsg_org)
            epsg_filename = epsg_org
        else:
            self.map_proj = ccrs.epsg(epsg)
            map_proj_transform_label = f"EPSG {epsg}"
            self.geo_coastline = self.geo_coastline.to_crs(epsg=epsg)
            self.geo_bathy = self.geo_bathy.to_crs(epsg=epsg)
            if self.stations is not None:
                self.stations = self.stations.to_crs(epsg=epsg)
            epsg_filename = epsg

        # set the min and max used for axes
        self.fxmin, self.fymin = self.map_proj.transform_point(x=fjord_xmin, y=fjord_ymin, src_crs=self.map_proj_original)
        self.fxmax, self.fymax = self.map_proj.transform_point(x=fjord_xmax, y=fjord_ymax, src_crs=self.map_proj_original)


        # extract info from the FarCoast files
        self.fig_text = farcoast_anno_text(
            attr_title = self.ds.attrs["title"],
            attr_file = self.ds.attrs["file"]
        )
        
        self._initiate_plot()

        # TODO: take this out of init
        
        
        # prepare the data for plotting
        #self._prepare_data_subset(self.first_time_frame, self.final_time_frame)
        # configue colorbar scaling
        #self._configure_colormap_and_scaling(self.ds_mean)

        # image "mesh"
        #self.image = self._plot_image(ax=self.ax, data=self.ds_mean.isel(ocean_time = 0))

        # inner_kwargs = {}

        # if "speed" in self.var_animate:

        #     self.arrows = self._plot_arrows(
        #         ax = self.ax, 
        #         data = self.resample.isel(ocean_time = 0),
        #         u = self.fig_text.u,
        #         v = self.fig_text.v,
        #         )
        #     inner_kwargs = {'arrows': self.arrows}

        # self.animation = self.animate()

        # # Apply all styling (coastline, limits, ticks, etc.)
        # self._style_axes(self.ax, **inner_kwargs)

        # # Add colorbar
        # self._add_colorbar(self.ax, self.image)


        # write the plot in location specified
        # TODO: take this out of init
        if(write_output):
            if(use_manual_boundaries):
                boundary = "bbox_"
            else:
                boundary = ""
            

            if not any(self.depth_slice):
                filename = (
                    f"{self.save_folder}ROMS_func_animation_{boundary}{self.fig_text.attr_file}"
                    f"_time{first_time_frame}to{self.ocean_time_max}"
                    f"_{self.var_animate}_covers_{self.var_percentile}_dpi{dpi}"
                    )
            else:
                depths = "".join(list(map(str, self.depth_slice)))
                filename = (
                    f"{self.save_folder}ROMS_func_animation_{boundary}{self.fig_text.attr_file}"
                    f"_time{first_time_frame}to{self.ocean_time_max}"
                    f"_{self.var_animate}_depth{depths}_covers_{self.var_percentile}_dpi{dpi}"
                    )

            if "speed" in self.var_animate:
                filename = f"{filename}_quiver{self.quiver_every_x_box}"
            
            filename = f"{filename}_epsg{epsg_filename}"

            if self.output_type == "gif":
                filename = f"{filename}.gif"
                self.animation.save(
                    filename = filename,
                    fps = self.fps,
                    writer = "pillow",
                    dpi = self.dpi,
                    savefig_kwargs = {'bbox_inches': 'thight'}
                )

            if self.output_type == "html":
                filename = f"{filename}.html"
                html = self.animation.to_jshtml()
                with open(filename, "w") as f:
                    f.write(html)

        plt.close(self.fig)    

    def _initiate_plot(self):
        # figure setup
        if self.null_proj:
            self.fig, self.ax = plt.subplots(
                figsize=cm2inch(15, 15), facecolor="white", dpi=self.dpi
                )  # No projection at all
            self.transform = self.ax.transData
        else:
            self.fig, self.ax = plt.subplots(
                figsize=cm2inch(15, 15), facecolor="white", dpi=self.dpi, 
                subplot_kw={'projection': self.map_proj}
                )
            self.transform = self.map_proj_original
    
    def _prepare_first_frame(self):
        #her skal meira gerast!!!
        self._prepare_data_subset_new(self.ds, t_index = 0)
        
    
    def _plot_image(self, ax, data, transform=None):
        transform = transform or self.transform
        Z = data.values
        X = data.lon_rho.values
        Y = data.lat_rho.values

        return ax.pcolormesh(
            X, Y, Z,
            vmin=self.minconc,
            vmax=self.maxconc,
            cmap=self.cmap_var,
            transform=transform,
            animated=True
        )
    
    def _plot_arrows(self, ax, data, u, v, transform=None, normalise=None):
        """
        The supplied u and v have to be east and north alligned, not grid alligned!!
        u and v are the names of these two in the data
        """
        transform = transform or self.transform
        normalise = normalise or self.quiver_normalise

        X = data['lon_rho'].values
        Y = data['lat_rho'].values
        U = data[u].values
        V = data[v].values

        if normalise:
            magnitude = np.sqrt(U**2 + V**2)
            magnitude[magnitude == 0] = 1.
            U = U / magnitude
            V = V / magnitude

        return ax.quiver(
            X, Y,
            U, V,
            transform=transform,
            animated=True,
            #units='xy',
            #scale_units = 'xy',
            angles = 'uv',
            scale=self.quiver_scale,
            minlength=0,
            width=0.003,
            headlength=3.5,
            headaxislength=3.5,
            headwidth=2.5,
            pivot='tail',
            zorder=70,
        )
    
    def _add_quiver_key(self, ax, arrows, x=0.1, y=0.08):
        if self.quiver_max is None:
            veclength = 0.5
        else:
            veclength = self.quiver_max

        """Add a quiverkey to the plot"""
        label = '%3.1f m/s' % veclength
        return ax.quiverkey(
                arrows,
                X=x, Y=y,
                U=veclength,
                label=label,
                labelpos='S',
                coordinates='axes',
                zorder = 70
                )
    
    def update(self, t):
        # set dynamic title
        self.ax.set_title(
            label = f"time = {np.datetime_as_string(t, 'h')}",
            x = 0.15,
            color = "black",
            style = 'italic',
            fontsize = 10
            )
        
        # data
        data_slice = self.ds_mean.sel(ocean_time = t)
        
        # update main image
        self.image.set_array(data_slice.values.ravel())

        #update arrows if applicaple
        if "speed" in self.var_animate:
            # Get updated vector fields for time t
            u_t = self.resample[self.fig_text.u].sel(ocean_time=t).values
            v_t = self.resample[self.fig_text.v].sel(ocean_time=t).values
            self.arrows.set_UVC(U=u_t, V=v_t)
    
    def animate(self):
        return FuncAnimation(
            self.fig,
            self.update,
            frames=self.ds_mean.ocean_time.values,
            blit=False,
        )
    
    def _style_axes(self, ax, modelinfo=True, show_xlabel=True, show_ylabel=True, **kwargs):
        # clear any left overs
        #ax.cla()

        # Coastline, bathy, limits
        self.geo_coastline.boundary.plot(ax=ax, edgecolor='grey')
        self.geo_coastline.plot(ax=ax, color='lightgrey', alpha=0.8, zorder=50)
        self.geo_bathy.plot(ax=ax, linewidth=0.2, color='grey')
   
        # set up plot parameters
        mpl.rcParams["font.size"] = 10
        mpl.rcParams['axes.edgecolor'] = 'lightgrey'

        ax.set_ylim([self.fymin, self.fymax])
        ax.set_xlim([self.fxmin, self.fxmax])
        ax.set_facecolor("white")

        # add stations
        if self.stations is not None:
            self.stations.plot(ax=ax, color='black', edgecolor = "black", zorder = 50)
        

        # Projection-specific gridlines or ticks
        if self.null_proj:
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(
                axis='both', which='both', 
                labelsize=6, labelcolor='grey', 
                labelbottom=show_xlabel,
                labelleft=show_ylabel,
                )
        else:
            gl = ax.gridlines(
                crs=self.transform,
                linewidth=1, draw_labels=True,
                color='grey', alpha=0.5, linestyle='--'
            )
            gl.top_labels = False
            gl.right_labels = False
            gl.left_labels = show_ylabel,
            gl.bottom_labels = show_xlabel,
            gl.xlabel_style = {'size': 6, 'color': 'gray'}
            gl.ylabel_style = {'size': 6, 'color': 'gray'}
        
        if "speed" in self.var_animate and hasattr(self, "arrows"):
            self._add_quiver_key(ax=ax, arrows = kwargs['arrows'])
            #qk.set_zorder(1)

        annotate_title(ax, fontsize=7, **self.fig_text.__dict__)
        if modelinfo:
            annotate_model_setup_info(ax, fontsize=8, **self.fig_text.__dict__)
        if self.watermark:
            annotate_watermark(ax=ax, text="Created by FarCoast v2")
            annotate_dataowner(ax=ax, text="Firum")
        
        # TODO: not sure if this is needed?
        self.fig.subplots_adjust(bottom=0.15)

    def _draw_static_figure(
        self, data, resample=None, title=None, 
        ax=None, nrows=1, ncols=1, 
        figsize=None, transform=None, modelinfo=True,
        show_xlabel=True, show_ylabel=True, colorbar=True,
        **kwargs):
        """General-purpose figure plotter for any static data array."""

        if figsize is None:
            size = cm2inch(15*ncols, 15*nrows)
        else:
            size = cm2inch(figsize[0]*ncols, figsize[1]*nrows)
        
        # common kwargs
        common = dict(
            figsize=size, facecolor="white", dpi=self.dpi, 
            squeeze = False, sharex='col', sharey='row',
            subplot_kw = ({} if self.null_proj else {'projection': self.map_proj})
            )

        if ax is None:
            # figure setup
            if self.null_proj:
                fig, axes = plt.subplots(nrows, ncols, **common)  # No projection at all
                axes = list(axes.flatten())
                transform = axes[0].transData
            else:
                fig, axes = plt.subplots(nrows, ncols, **common, 
                    #subplot_kw={'projection': self.map_proj}
                    )
                axes = list(axes.flatten())
                transform = self.map_proj_original
            
        else:
            fig = None
            axes = [ax]
            transform = transform

        # Use the internal image plotting function
        mesh = self._plot_image(ax=axes[0], data=data, transform=transform)

        inner_kwargs = {}

        # Optional arrows
        if "speed" in self.var_animate and resample is not None:
            u = resample[self.fig_text.u]
            v = resample[self.fig_text.v]
            arrows = self._plot_arrows(
                ax=axes[0],
                data=resample,
                u=u.name,
                v=v.name,
                transform=transform
            )
            inner_kwargs = {'arrows': arrows}

        self._configure_colormap_and_scaling(data)
        self._style_axes(
            ax = axes[0], modelinfo=modelinfo, 
            show_ylabel=show_ylabel, show_xlabel=show_xlabel, **inner_kwargs)
        if colorbar:
            self._add_colorbar(ax = axes[0], mesh=mesh)

        # Title
        if title:
            axes[0].set_title(title, fontsize=10, color="black")

        return fig, axes, transform
    
    def _timeslice(self, data, t_index):
        """
        t_index can be:
          - an int               → positional (one time)
          - an ISO date string   → label (one time)
          - a tuple/list of two  → (start, end) period
          - a slice object       → slice(start, end)
        Returns either a single frame or a time‐period slice.
        """
        
        time_vals = data.ocean_time.values
        n_times = len(time_vals)
        
        def to_ns(dt):
            if isinstance(dt, int):
                if not (-n_times <= dt < n_times):
                    raise IndexError(
                        f"index {dt!r} out of bounds [0..{n_times-1}]"
                    )
                t0 = time_vals[dt]

            elif isinstance(dt, str):
                t0 = np.datetime64(dt)
                if t0 not in time_vals:
                    raise IndexError(
                        f"Time {dt!r} not in ocean_time index; valid is {time_vals[0]}...{time_vals[-1]}."
                    )
            
            return t0

        # single‐value selection
        if isinstance(t_index, (int, str, np.datetime64)):
            t0 = to_ns(t_index)
            return data.sel(ocean_time=t0)

        # tuple/list → (start, end)
        if isinstance(t_index, (tuple, list)) and len(t_index) == 2:
            start, end = t_index
            t0, t1 = to_ns(start), to_ns(end)
            if t0 > t1:
                raise ValueError(f"start time {t0} is after end time {t1}")
            return data.sel(ocean_time=slice(t0, t1))

        # slice object → slice(start, stop)
        if isinstance(t_index, slice):
            t0 = to_ns(t_index.start) if t_index.start is not None else None
            t1 = to_ns(t_index.stop ) if t_index.stop  is not None else None
            return data.sel(ocean_time=slice(t0, t1))

        # anything else is invalid
        raise TypeError(
            "t_index must be int, ISO-string, np.datetime64, "
            "a 2-tuple/list, or a slice object"
        )
    
    def draw_frame(
        self, t_index, title=None, 
        ax=None, nrows=1, ncols=1, 
        figsize=None, transform=None, silent=False, modelinfo=True, 
        show_xlabel = True, show_ylabel = True, colorbar=True, **kwargs):
        """Draws a single frame at a specific time index."""
        
        data = self._timeslice(self.ds_mean, t_index)
        t0 = data.ocean_time.values

        if data.ocean_time.size > 1:
            raise ValueError(f"Only select one time frame!, {data.ocean_time.size} were selected")

        if "speed" in self.var_animate:
            resampled = self.resample.sel(ocean_time=t0)
        else:
            resampled = None
        title = title or f"Frame at {np.datetime_as_string(t0, 'h')}"
        fig, ax, transform = self._draw_static_figure(
            data, resample=resampled, title=title, 
            ax = ax, nrows=nrows, ncols=ncols, figsize=figsize, modelinfo=modelinfo, 
            show_xlabel = show_xlabel, show_ylabel = show_ylabel, colorbar=colorbar, **kwargs,
            )
        if silent:
            return None
        else:
            return fig, ax, transform
    
    def draw_frames_grid(self, t_indices, nrows, ncols, figsize=None):

        fig, axes, transform = self.draw_frame(
            t_index=t_indices[0], nrows=nrows, ncols=ncols, figsize=figsize,
            modelinfo = False, show_ylabel=False, show_xlabel = False, colorbar = False)
        # fig, axes, transform = self._draw_static_figure(
        #     data=self._timeslice(self.ds_mean, t_indices[0]), ax=None, nrows=nrows, ncols=ncols, figsize=figsize,
        #     modelinfo = False, show_ylabel=False, show_xlabel = False, colorbar = False)

        for i, (ax, t) in enumerate(zip(axes, t_indices)):
            # ax in last row or first column
            is_bottom = (i // ncols) == (nrows - 1)
            is_left   = (i %  ncols) == 0
            is_right = (i % ncols) == (ncols -1)

            #ax.cla()
            #ax.gridlines(draw_labels=False)

            self.draw_frame(
                t_index=t, ax=ax, transform=transform, modelinfo = False, 
                show_ylabel = is_left, show_xlabel = is_bottom, colorbar = is_right)

        # 5) finalize
        #fig.tight_layout()
        fig.canvas.draw()
    
    def draw_timeperiod_frame(self, t_index, reduce_func="mean", **kwargs):
        """
        Draw a frame summarizing a time‐period slice using a reduction (mean, max, etc).

        Parameters:
        - t_index: see _timeslice() function
        - reduce_func: "mean", "max", "min" or a custom callable
        """

        data = self._timeslice(self.ds_mean, t_index)
        t0 = data.ocean_time.values.min()
        t1 = data.ocean_time.values.max()

        # Extract and reduce main variable
        reducer = {
            "mean": lambda x: x.mean(dim="ocean_time"),
            #"max": lambda x: x.max(dim="ocean_time"),
            #"min": lambda x: x.min(dim="ocean_time"),
        }.get(reduce_func, reduce_func)
        
        data = reducer(data)

        if self.var_animate == "speed":
            resampled = self._timeslice(self.resample, t_index)
            resampled = reducer(resampled)
        else:
            resampled = None
        # Title and render
        title = f"{reduce_func} from {np.datetime_as_string(t0, 'h')} to {np.datetime_as_string(t1, 'h')}"
        self._draw_static_figure(data, resample=resampled, title=title, **kwargs)

    def draw_timeperiod_frame1(self, t_start, t_end, reduce_func = "mean"):
        """
        Update the animation to show a reduction over a time period [t_start, t_end].

        Parameters:
        - t_start: int or datetime-like (start index or time)
        - t_end: int or datetime-like (end index or time)
        - reduce_func: str or callable — one of "mean", "max", or a custom function
        """
        # Extract the time values
        time_vals = self.ds_mean.ocean_time.values
        t0 = time_vals[t_start] if isinstance(t_start, int) else np.datetime64(t_start)
        t1 = time_vals[t_end]   if isinstance(t_end, int) else np.datetime64(t_end)

        # Compute mean over time period
        ds_slice = self.ds_mean.sel(ocean_time=slice(t0, t1))
        
        # Resolve reduction function
        reduce_map = {
            "mean": lambda x: x.mean(dim="ocean_time"),
            "max": lambda x: x.max(dim="ocean_time"),
            "min": lambda x: x.min(dim="ocean_time"),
            # Add more as needed
        }
        reducer = reduce_map[reduce_func] if isinstance(reduce_func, str) else reduce_func

        # Reduce
        ds_reduced = reducer(ds_slice)
        
        # Dynamic title
        self.ax.set_title("")
        self.ax.text(
            0.02, 1.05,  # slightly above the top of the axes
            f"{reduce_func} from {np.datetime_as_string(t0, 'h')} to {np.datetime_as_string(t1, 'h')}",
            transform=self.ax.transAxes,
            ha='left', va='bottom',
            fontsize=10,
            style='italic',
            color='black',
            clip_on=False  # make sure it's not clipped by axes boundary
            )

        # Update image
        self.image.set_array(ds_reduced)

        # Update arrows if present
        if "speed" in self.var_animate:
            u = reducer(self.resample[self.fig_text.u].sel(ocean_time=slice(t0, t1)))
            v = reducer(self.resample[self.fig_text.v].sel(ocean_time=slice(t0, t1)))
            self.arrows.set_UVC(U=u, V=v)

        self.fig.canvas.draw()
        display(self.fig)

    def to_jshtml(self):
        """Return HTML for embedding in Quarto or Jupyter."""
        return HTML(self.animation.to_jshtml())
    
    def to_html5_video(self):
        """Return HTML for embedding in Quarto or Jupyter."""
        return HTML(self.animation.to_html5_video())
    
    def embed_animation_responsive(self, max_width=800):
        """
        Convert a matplotlib.animation.FuncAnimation to responsive HTML5 video with autoplay and no borders.

        Parameters:
        - anim: The FuncAnimation object.
        - max_width: Max width of the video container in pixels (default: 800).

        Returns:
        - HTML object for display in Jupyter/Quarto.
        """
        if getattr(self, "animation", None) is None:
            raise RuntimeError("Animation not built yet. Call .build() first.")

        html_video = self.animation.to_html5_video()
        styled_html = f'''
        <div style="max-width: {max_width}px; margin: auto; padding: 0; overflow:hidden;">
            {html_video.replace(
                '<video ',
                f'<video autoplay loop muted '
                f'style="width:100%; height:auto; '
                f'display:block; margin: -2px -2px -2px -2px; '
                f'padding:0; border:none; putline:none;" '
                )
            }
        </div>
        '''
        return HTML(styled_html)

    def show(self):
        """Display the static plot (first frame)."""
        plt.show(self.animation)

    def save(self, filename, fps=None, dpi=None):
        """Save to GIF with configured fps/dpi."""
        if fps is None:
            fps = self.fps
        if dpi is None:
            dpi = self.dpi

        self.animation.save(filename, fps=fps, dpi=dpi)
    
    
    def _add_colorbar(self, ax, mesh):
        """Attach a vertical colorbar to the provided axes and mesh."""
        # Create an inset axis next to the main plot
        cax = inset_axes(
            ax,
            width="3%",  # Width of colorbar (relative to parent)
            height="100%",  # Full height
            loc="lower left",
            bbox_to_anchor=(1.0, 0.0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0
        )

        # Decide colorbar extension mode
        if self.var_percentile == 100 and self.var_max is not None:
            extend = 'max'
        elif self.var_percentile == 100:
            extend = 'neither'
        else:
            extend = 'both'
        
        # Create and style colorbar
        cbar = ax.figure.colorbar(mesh, cax=cax, orientation='vertical', extend=extend)
        cbar.set_label(self.colorbar_label, labelpad=5, fontsize=10)
        cbar.ax.tick_params(labelsize=6, color='grey')
    
    def _prepare_data_subset_new(self, t_index=None):
        
        if t_index is None:
            self.ocean_time_max = self.final_time_frame or self.ds.ocean_time.size
            time_slice = slice(self.first_time_frame, self.ocean_time_max)
        else:
            time_slice = t_index
        
        # relevant timeslice
        ds_var = self._timeslice(data = self.ds[self.var_animate], t_index = time_slice)
        # Determine depth dimension
        if "s_rho" in ds_var.dims:
            depth_dim = "s_rho"
            u_name, v_name = "u_rho", "v_rho"
            depth_slice = {depth_dim: slice(*self.depth_slice)}

        elif "z" in ds_var.dims:
            depth_dim = "z"
            u_name, v_name = "u_rho_z", "v_rho_z"
            a, b = self.depth_slice
            print(a, b)
            if a is not None:
                i0 = int(ds_var.indexes[depth_dim].get_indexer([a], method="nearest")[0])
            else: 
                i0 = a
            if b is not None:
                i1 = int(ds_var.indexes[depth_dim].get_indexer([b], method="nearest")[0]) + 1
            else:
                i1 = b
            print(i0, i1)
            depth_slice = {depth_dim: slice(i0,i1)}
        else:
            raise ValueError("Depth dimension (s_rho or z) not found in variable.")

        # Save for later
        self.fig_text.add(
            depth_var=depth_dim,
            depth_used=f"{depth_dim} depths",
            u=u_name,
            v=v_name
        )

        # Calculate depth metadata
        depth_vals = ds_var[depth_dim].isel(**depth_slice).values
        self.fig_text.add(
            mean_depth=np.round(np.nanmean(depth_vals), 2),
            min_depth=np.round(np.nanmin(depth_vals), 2),
            max_depth=np.round(np.nanmax(depth_vals), 2)
        )

         # Time-sliced subset of the variable averaged over depth
        if "speed" in self.var_animate:
            
            # Average u and v over depth
            u = self.ds[u_name].isel(ocean_time=time_slice).isel(**depth_slice)
            v = self.ds[v_name].isel(ocean_time=time_slice).isel(**depth_slice)
            ang = self.ds['angle']
            if "units" in ang.attrs and "deg" in ang.attrs["units"].lower():
                ang = xr.apply_ufunc(np.deg2rad, ang)

            # SCALAR-MEAN SPEED for mesh
            speed = np.sqrt(u**2 + v**2).mean(dim=depth_dim, skipna=True)
            speed.attrs["long_name"] = "Mean current speed (scalar average)"
            speed.attrs["units"] = "m/s"
            self.ds_mean = speed.load()

            # VECTOR-MEAN u/v for arrows
            u_depthavg = u.mean(dim=depth_dim, skipna=True)
            v_depthavg = v.mean(dim=depth_dim, skipna=True)
            print(u_depthavg.shape)

            # Rotate grid (ξ,η) -> geographic (E,N)
            # sometimes this is rotated differently, just be avare!!
            # TODO: is there a way to make sure this doesn't get buggered??
            Ue = u_depthavg*np.cos(ang) - v_depthavg*np.sin(ang)   # eastward
            Vn = u_depthavg*np.sin(ang) + v_depthavg*np.cos(ang)   # northward

            quiver_step = self.quiver_every_x_box
            idx = dict(eta_rho=slice(None, None, quiver_step), xi_rho=slice(None, None, quiver_step))

            # Coarsen for quiver plot
            # TODO: there should be an option for no arrows!
            self.resample = xr.Dataset(
                data_vars={
                    u_name: Ue.isel(**idx),
                    v_name: Vn.isel(**idx),
                },
                coords={
                    "lon_rho": ds_var["lon_rho"].isel(**idx),
                    "lat_rho": ds_var["lat_rho"].isel(**idx),
                    "ocean_time": ds_var["ocean_time"],
                },
            ).load()
            print(self.resample[u_name].shape)

            # Optional arrow masking
            if self.var_max is not None:
                mag2 = self.resample[u_name]**2 + self.resample[v_name]**2
                self.resample = self.resample.where(mag2 < self.var_max**2, 0)
            if self.quiver_max is not None:
                mag3 = self.resample[u_name]**2 + self.resample[v_name]**2
                self.resample = self.resample.where(mag3 < self.quiver_max**2, 0)
        
        else:
            # Regular variable averaging over depth
            ds_subset = ds_var.isel(**depth_slice)
            self.ds_mean = ds_subset.mean(dim=depth_dim, skipna=True, keep_attrs=True).load()

        
        # Add time bounds to fig_text
        self.fig_text.add(
            start_hour = np.datetime_as_string(self.ds_mean.isel(ocean_time = 0).ocean_time.values,'h'),
            end_hour = np.datetime_as_string(self.ds_mean.isel(ocean_time = -1).ocean_time.values,'h')
            )



    def _prepare_data_subset(self, first_time_frame, final_time_frame):
        """
        Prepares the subset of data for plotting, including:
        - slicing over time
        - slicing over depth (s_rho or z)
        - calculating resample fields for quiver arrows
        - computing depth metadata and storing in self.fig_text
        - calculating mean over depth and storing in self.ds_mean
        """
        # size of ocean time
        self.ocean_time_max = final_time_frame or self.ds.ocean_time.size
        time_slice = slice(first_time_frame, self.ocean_time_max)
        
        # Determine depth dimension
        ds_var = self.ds[self.var_animate].isel(ocean_time=time_slice)
        if "s_rho" in ds_var.dims:
            depth_dim = "s_rho"
            u_name, v_name = "u_rho", "v_rho"
            depth_slice = dict(s_rho=slice(*self.depth_slice))
            u_slice = dict(s_rho=slice(*self.depth_slice))
            v_slice = dict(s_rho=slice(*self.depth_slice))

        elif "z" in ds_var.dims:
            depth_dim = "z"
            u_name, v_name = "u_rho_z", "v_rho_z"
            depth_slice = dict(z=slice(*self.depth_slice))
            u_slice = dict(z=slice(*self.depth_slice))
            v_slice = dict(z=slice(*self.depth_slice))
        else:
            raise ValueError("Depth dimension (s_rho or z) not found in variable.")

        # Save for later
        self.fig_text.add(
            depth_var=depth_dim,
            depth_used=f"{depth_dim} depths",
            u=u_name,
            v=v_name
        )
        # Calculate depth metadata
        depth_vals = ds_var[depth_dim].isel(**depth_slice).values
        self.fig_text.add(
            mean_depth=np.round(np.nanmean(depth_vals), 2),
            min_depth=np.round(np.nanmin(depth_vals), 2),
            max_depth=np.round(np.nanmax(depth_vals), 2)
        )

        # Time-sliced subset of the variable averaged over depth
        if "speed" in self.var_animate:
            
            # Average u and v over depth
            # Extract u/v and average over depth
            u = self.ds[u_name].isel(ocean_time=time_slice).isel(**u_slice)
            v = self.ds[v_name].isel(ocean_time=time_slice).isel(**v_slice)

            # SCALAR-MEAN SPEED for mesh
            #speed = np.sqrt(u**2 + v**2).mean(dim=depth_dim, skipna=True)
            # dim behaviour for s_rho and z
            use_dim_speed = depth_dim
            speed = np.sqrt(u**2 + v**2).mean(dim=use_dim_speed, skipna=True)
            #speed = np.sqrt(u**2 + v**2)
            #print(speed.dims)
            # HER!!!
            speed.attrs["long_name"] = "Mean current speed (scalar average)"
            speed.attrs["units"] = "m/s"
            self.ds_mean = speed.load()

            # VECTOR-MEAN u/v for arrows
            # dimensions are strange
            depth_dim_u = depth_dim
            depth_dim_v = depth_dim
            u_depthavg = u.mean(dim=depth_dim_u, skipna=True)
            v_depthavg = v.mean(dim=depth_dim_v, skipna=True)

            # Coarsen for quiver plot
            # TODO: there should be an option for no arrows!
            self.resample = xr.Dataset({
                u_name: u_depthavg.isel(
                    eta_rho=slice(None, None, self.quiver_every_x_box),
                    xi_rho=slice(None, None, self.quiver_every_x_box)
                ),
                v_name: v_depthavg.isel(
                    eta_rho=slice(None, None, self.quiver_every_x_box),
                    xi_rho=slice(None, None, self.quiver_every_x_box)
                )
            }).load()

            # Optional arrow masking
            if self.var_max is not None:
                mag2 = self.resample[u_name]**2 + self.resample[v_name]**2
                self.resample = self.resample.where(mag2 < self.var_max**2, 0)
            if self.quiver_max is not None:
                mag3 = self.resample[u_name]**2 + self.resample[v_name]**2
                self.resample = self.resample.where(mag3 < self.quiver_max**2, 0)
        
        else:
            # Regular variable averaging over depth
            ds_subset = ds_var.isel(**depth_slice)
            self.ds_mean = ds_subset.mean(dim=depth_dim, skipna=True, keep_attrs=True).load()

        
        # Add time bounds to fig_text
        self.fig_text.add(
            start_hour = np.datetime_as_string(self.ds_mean.isel(ocean_time = 0).ocean_time.values,'h'),
            end_hour = np.datetime_as_string(self.ds_mean.isel(ocean_time = -1).ocean_time.values,'h')
            )

    def _configure_colormap_and_scaling(self, data):
        """
        Sets colormap, colorbar label, and min/max concentration values based on ds_mean.
        """
        # Fetch metadata
        attr_long_name = data.attrs["long_name"]
        attr_units = "" if "salt" in self.var_animate else data.attrs["units"]

        # set color map and quivers
        if "temp" in self.var_animate:
            self.cmap_var = cmocean.cm.thermal
        elif "salt" in self.var_animate:
            self.cmap_var = cmocean.cm.haline
            #cmap_var = plt.cm.nipy_spectral
        elif "speed" in self.var_animate:
            self.cmap_var = cmocean.cm.speed
        else:
            self.cmap_var = plt.cm.plasma

        # Compose colorbar label
        if "salt" in self.var_animate:
            self.colorbar_label = f"{attr_long_name}"
        else:
            self.colorbar_label = f"{attr_long_name} ({attr_units})"

        if self.var_percentile != 100:
            self.colorbar_label += f" - color covers {self.var_percentile}%"

        # Mask values below minimum conc
        clean_values = data.values[~np.isnan(data.values)]
        # percentiles
        var_min_perc = (100 - self.var_percentile)/2
        var_max_perc = var_min_perc + self.var_percentile

        self.minconc = round(np.percentile(clean_values, var_min_perc),ndigits=1)
        self.maxconc = round(np.percentile(clean_values, var_max_perc),ndigits=1)

        #manual max value
        if self.var_max is not None:
            self.maxconc = self.var_max

        # Optionally set up tick labels if needed in future
        self.color_levels = np.arange(self.minconc, self.maxconc, 0.2).tolist()

    def clone(self, **override_kwargs):
        """
        Return a brand-new farcoast_animation with identical settings,
        except for any you override here.
        """
        # copy original init-args
        new_kwargs = self._init_kwargs.copy()
        # apply any overrides
        new_kwargs.update(override_kwargs)
        # build and return a fresh object
        return type(self)(**new_kwargs)
    
    def build(self):
        """
        Explicitly build the first frame and the FuncAnimation.

        Usage:
            ani = farcoast_animation(..., write_output=False).build()
            ani.embed_animation_responsive()
        """
        # 1) prepare subset (time window, depth averaging, quiver fields)
        self._prepare_data_subset_new(t_index=None)

        # 2) set colormap and scaling from the prepared data
        self._configure_colormap_and_scaling(self.ds_mean)

        # 3) draw first frame (image)
        self.image = self._plot_image(
            ax=self.ax,
            data=self.ds_mean.isel(ocean_time=0)
        )

        # 4) arrows if animating a speed field
        inner_kwargs = {}
        if "speed" in self.var_animate:
            self.arrows = self._plot_arrows(
                ax=self.ax,
                data=self.resample.isel(ocean_time=0),
                u=self.fig_text.u,
                v=self.fig_text.v,
            )
            inner_kwargs = {"arrows": self.arrows}

        # 5) axes styling + colorbar
        self._style_axes(self.ax, **inner_kwargs)
        self._add_colorbar(self.ax, self.image)

        # 6) create the matplotlib.animation.FuncAnimation
        self.animation = self.animate()

        return self