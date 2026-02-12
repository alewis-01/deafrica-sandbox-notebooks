# Surface_mining_screening_refactor.py

import os
from pathlib import Path
import datacube
import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from IPython.display import display, Markdown

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from odc.geo.geom import Geometry
from deafrica_tools.bandindices import calculate_indices, dualpol_indices
from deafrica_tools.datahandling import load_ard
from deafrica_tools.plotting import map_shapefile, rgb
from deafrica_tools.spatial import xr_rasterize

from shapely.geometry import MultiPolygon, Polygon
from skimage.filters import threshold_otsu
from skimage.morphology import binary_dilation, disk

import rasterio
from rasterio.transform import from_bounds



# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def pixel_area_km2_from_coords(ds: xr.Dataset | xr.DataArray) -> float:
    """
    Robust pixel area (km^2) from x/y coordinate spacing (handles negative resolution).
    Works for projected CRS data (e.g., EPSG:6933).
    """
    # x/y resolution in metres (absolute)
    dx = float(abs(ds.x[1] - ds.x[0]))
    dy = float(abs(ds.y[1] - ds.y[0]))
    return (dx * dy) / 1_000_000.0  # m^2 -> km^2


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def convert_3D_polygon_to_2D(poly_3D: Polygon) -> Polygon:
    exterior_2d = [(x, y) for x, y, *_ in poly_3D.exterior.coords]
    interiors_2d = [[(x, y) for x, y, *_ in ring.coords] for ring in poly_3D.interiors]
    return Polygon(exterior_2d, interiors_2d)


def convert_3D_geometry_to_2D(geom_3D: Polygon | MultiPolygon) -> Polygon | MultiPolygon:
    if geom_3D.geom_type == "Polygon":
        return convert_3D_polygon_to_2D(geom_3D)
    if geom_3D.geom_type == "MultiPolygon":
        return MultiPolygon([convert_3D_polygon_to_2D(p) for p in geom_3D.geoms])
    return geom_3D

def show_title_after(text: str):
    """Caption shown AFTER an output cell renders."""
    try:
        display(Markdown(f"*{text}*"))
    except Exception:
        print(text)

def show_df_after(df: pd.DataFrame, caption: str | None = None):
    display(df)
    if caption:
        show_title_after(caption)

def show_plot_after(caption: str | None = None):
    """Call this AFTER plt.show()."""
    if caption:
        show_title_after(caption)

def save_dataarray_geotiff(
    da: xr.DataArray,
    out_path: str | Path,
    nodata: float | int | None = None,
    dtype: str | None = None,
    compress: str = "deflate",
):
    """
    Save a 2D DataArray to GeoTIFF.
    Requires da.odc.geobox to exist (common in ODC/DE Africa).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if da.ndim != 2:
        raise ValueError(f"Expected 2D DataArray, got shape {da.shape} with dims {da.dims}")

    if not hasattr(da, "odc") or not hasattr(da.odc, "geobox"):
        # Fallback: construct transform from bounds (works if coords are regular)
        xmin, xmax = float(da.x.min()), float(da.x.max())
        ymin, ymax = float(da.y.min()), float(da.y.max())
        width, height = da.sizes["x"], da.sizes["y"]
        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

        crs = None
        if hasattr(da, "geobox") and getattr(da.geobox, "crs", None) is not None:
            crs = da.geobox.crs
        # else: try attrs
        crs = crs or da.attrs.get("crs", None)
    else:
        gb = da.odc.geobox
        transform = gb.transform
        crs = gb.crs

    arr = da.values

    if dtype is None:
        dtype = str(arr.dtype)

    if nodata is None:
        # If float, use NaN; if int, set a safe sentinel
        if np.issubdtype(arr.dtype, np.floating):
            nodata = np.nan
        else:
            nodata = -9999

    # For float GeoTIFF, nodata cannot be NaN reliably in all stacks
    # so we convert NaNs to a finite nodata if needed.
    write_arr = arr
    write_nodata = nodata
    if np.issubdtype(np.dtype(dtype), np.floating):
        if np.isnan(nodata):
            # choose a conventional nodata
            write_nodata = -9999.0
        write_arr = np.where(np.isfinite(arr), arr, write_nodata).astype(dtype)
    else:
        # integer
        if np.issubdtype(arr.dtype, np.floating):
            write_arr = np.where(np.isfinite(arr), arr, write_nodata).astype(dtype)
        else:
            write_arr = arr.astype(dtype)

    profile = dict(
        driver="GTiff",
        height=write_arr.shape[0],
        width=write_arr.shape[1],
        count=1,
        dtype=dtype,
        crs=crs,
        transform=transform,
        nodata=write_nodata,
        compress=compress,
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(write_arr, 1)


def save_time_stack_geotiffs(
    da_time: xr.DataArray,
    out_dir: str | Path,
    prefix: str,
    nodata: float | int | None = None,
    dtype: str | None = None,
):
    """
    Save a (time, y, x) DataArray as one GeoTIFF per timestep.
    """
    out_dir = _ensure_dir(out_dir)
    years = pd.to_datetime(da_time.time.values).year
    for i, y in enumerate(years):
        save_dataarray_geotiff(
            da_time.isel(time=i),
            out_dir / f"{prefix}_{int(y)}.tif",
            nodata=nodata,
            dtype=dtype,
        )


# -----------------------------------------------------------------------------
# Core pipeline functions
# -----------------------------------------------------------------------------
def load_vector_file(vector_file: str):
    extension = vector_file.split(".")[-1].lower()
    if extension == "kml":
        gpd.io.file.fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(vector_file, driver="KML")
    else:
        gdf = gpd.read_file(vector_file)

    gdf["geometry"] = gdf["geometry"].apply(convert_3D_geometry_to_2D)
    geom = Geometry(gdf.unary_union, gdf.crs)

    # optional interactive preview
    map_shapefile(gdf, attribute=gdf.columns[0], fillOpacity=0, weight=3)
    return gdf, geom


def process_data(gdf, geom, start_year, end_year, product="s2", output_crs="epsg:6933"):
    """
    Load annual data for the AOI:
    - S2 geomedian annual + NDVI, OR
    - S1 RTC annual + RVI
    Also loads annual WOfS frequency and masks to AOI.
    """
    dc = datacube.Datacube(app="surface_mining")

    query = {"geopolygon": geom}

    if product == "s1":
        ds = load_ard(
            dc=dc,
            products=["s1_rtc"],
            time=(f"{start_year}", f"{end_year}"),
            measurements=["vv", "vh"],
            resolution=(-20, 20),
            output_crs=output_crs,
            group_by="solar_day",
            **query,
        )
        ds["vh/vv"] = ds.vh / ds.vv
        ds = dualpol_indices(ds, index="RVI")
        ds = ds.resample(time="1Y").median("time")

    elif product == "s2":
        ds = dc.load(
            product="gm_s2_annual",
            measurements=["red", "green", "blue", "nir"],
            time=(f"{start_year}", f"{end_year}"),
            resolution=(-10, 10),
            **query,
        )
        ds = calculate_indices(ds, ["NDVI"], satellite_mission="s2")
    else:
        raise ValueError("product must be 's1' or 's2'")

    ds_wofs = dc.load(
        product="wofs_ls_summary_annual",
        time=(f"{start_year}", f"{end_year}"),
        resampling="nearest",
        like=ds.geobox,
    ).frequency

    # Rasterize AOI to dataset grid and mask all outputs
    mask = xr_rasterize(gdf, ds).astype(bool)

    ds = ds.where(mask)
    ds_wofs = ds_wofs.where(mask)

    # water appeared in >10% of observations in a given year
    water_bool = ds_wofs > 0.1
    water_frequency_sum = water_bool.sum("time").where(mask)

    return ds, water_frequency_sum, mask


def calculate_vegetation_loss(ds: xr.Dataset, product="s2", threshold=-0.15):
    """
    Compute year-to-year change and vegetation loss boolean:
      loss = (index[t] - index[t-1]) < threshold
    Returns:
      loss_bool(time,y,x), loss_sum(y,x), change(time,y,x)
    """
    index = "NDVI" if product == "s2" else "RVI"
    if index not in ds:
        raise KeyError(f"{index} not found in dataset. Available: {list(ds.data_vars)}")

    change = ds[index] - ds[index].shift(time=1)

    if threshold == "otsu":
        thr = threshold_otsu(np.nan_to_num(change.values, nan=0.0))
    else:
        thr = float(threshold)

    loss_bool = (change < thr)
    # keep NaNs outside AOI masked as NaN (not False)
    loss_bool = loss_bool.where(np.isfinite(ds[index]))

    # sum of years with loss (counts True as 1)
    loss_sum = loss_bool.fillna(False).astype(np.uint8).sum("time")

    return loss_bool, loss_sum, change, thr


def possible_mining_masks(
    vegetation_loss_sum: xr.DataArray,
    water_frequency_sum: xr.DataArray,
    ds: xr.Dataset,
    buffer_m: float = 90.0,
):
    """
    Efficient mining screening:
    - base_mining = (veg_loss_sum>0) & (water_frequency_sum>0 for at least one year)
    - buffered_mining = raster buffer via binary dilation (fast)
    - veg_loss_in_buffer = veg_loss_sum within buffered mining
    Returns:
      base_mining_mask, buffered_mining_mask, veg_loss_in_buffer_mask
    """
    # base mining candidate pixels
    base_mining = (vegetation_loss_sum > 0) & (water_frequency_sum > 0)
    base_mining = base_mining.fillna(False)

    # buffer in pixels
    # ds coords are in metres in projected CRS (e.g., EPSG:6933)
    res_m = float(abs(ds.x[1] - ds.x[0]))
    radius_px = int(np.ceil(buffer_m / res_m))
    if radius_px < 1:
        radius_px = 1

    buffered = binary_dilation(base_mining.values.astype(bool), footprint=disk(radius_px))
    buffered_mining = xr.DataArray(
        buffered.astype(np.uint8),
        coords=base_mining.coords,
        dims=base_mining.dims,
        name="buffered_mining",
    )

    veg_loss_in_buffer = (vegetation_loss_sum > 0) & (buffered_mining == 1)
    veg_loss_in_buffer = veg_loss_in_buffer.fillna(False).astype(np.uint8)

    base_mining_mask = base_mining.astype(np.uint8)
    return base_mining_mask, buffered_mining, veg_loss_in_buffer


def build_summary_table(
    ds: xr.Dataset,
    vegetation_loss_bool: xr.DataArray,
    veg_loss_in_buffer_mask: xr.DataArray,
    product="s2",
):
    """
    Builds a per-year summary table:
    - total AOI area (km2)
    - any vegetation loss area (km2, %)
    - vegetation loss within possible mining buffer area (km2, %)
    Returns DataFrame.
    """
    index = "NDVI" if product == "s2" else "RVI"
    background = ds[index].isel(time=0)

    pix_area = pixel_area_km2_from_coords(ds)
    total_area = int(np.count_nonzero(np.isfinite(background.values))) * pix_area

    years = pd.to_datetime(vegetation_loss_bool.time.values).year

    # any vegetation loss per year
    loss_any = vegetation_loss_bool.fillna(False).astype(np.uint8)
    loss_any_area = loss_any.sum(dim=["y", "x"]).values * pix_area

    # vegetation loss within buffer (per year):
    # veg_loss_in_buffer_mask is (y,x); apply it to yearly loss
    loss_in_buffer = (loss_any == 1) & (veg_loss_in_buffer_mask == 1)
    loss_in_buffer_area = loss_in_buffer.sum(dim=["y", "x"]).values * pix_area

    df = pd.DataFrame(
        {
            "year": years,
            "any_veg_loss_km2": loss_any_area,
            "any_veg_loss_pct": (loss_any_area / total_area) * 100.0,
            "veg_loss_in_mining_buffer_km2": loss_in_buffer_area,
            "veg_loss_in_mining_buffer_pct": (loss_in_buffer_area / total_area) * 100.0,
        }
    )
    meta = pd.DataFrame(
        {
            "metric": ["total_aoi_area_km2"],
            "value": [total_area],
        }
    )
    return df, meta


# -----------------------------------------------------------------------------
# Optional plotting helpers (no recompute)
# -----------------------------------------------------------------------------
def plot_possible_mining_map(
    ds: xr.Dataset,
    veg_loss_in_buffer_mask: xr.DataArray,
    product="s2",
    out_png: str | Path | None = None,
):
    index = "NDVI" if product == "s2" else "RVI"
    bg = ds[index].isel(time=0)

    plt.figure(figsize=(12, 12))
    bg.plot.imshow(cmap="Greys", add_colorbar=False)
    veg_loss_in_buffer_mask.where(veg_loss_in_buffer_mask == 1).plot.imshow(
        cmap=ListedColormap(["Gold"]), add_colorbar=False
    )
    plt.legend([Patch(facecolor="Gold")], ["Possible Mining Site"], loc="upper left")
    plt.title("Possible Mining Areas")
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.show()


def run_surface_mining_screening(
    vector_file: str,
    start_year: str,
    end_year: str,
    product: str = "s2",
    threshold: float | str = -0.15,
    buffer_m: float = 90.0,
    out_dir: str = "results",
    export_yearly_loss_geotiffs: bool = True,
    export_yearly_loss_pngs: bool = True,
    dpi: int = 300,
    col_wrap: int = 6,
):
    """
    One-shot runner that:
      - loads AOI + imagery
      - computes vegetation loss + possible mining masks
      - prints/labels outputs in notebook when displayed
      - exports CSV tables + GeoTIFF rasters
      - saves key figures as PNG at dpi=300 (including year time-series)
    """

    # ---- Notebook display helpers (local so function is self-contained) ---    
    def plot_and_save_veg_loss_timeseries(
        ds: xr.Dataset,
        vegetation_loss_bool: xr.DataArray,
        veg_loss_in_buffer_mask: xr.DataArray,
        out_png: str | Path,
    ):
        pix_area = pixel_area_km2_from_coords(ds)
        years = pd.to_datetime(vegetation_loss_bool.time.values).year

        loss_any = vegetation_loss_bool.fillna(False).astype(np.uint8)
        loss_any_area = loss_any.sum(dim=["y", "x"]).values * pix_area

        loss_in_buffer = (loss_any == 1) & (veg_loss_in_buffer_mask == 1)
        loss_in_buffer_area = loss_in_buffer.sum(dim=["y", "x"]).values * pix_area

        show_title_after("Vegetation loss time series (and saved PNG at 300 dpi)")

        plt.figure(figsize=(11, 4))
        plt.plot(years, loss_any_area, marker="o", label="Any vegetation loss (km²)")
        plt.plot(
            years,
            loss_in_buffer_area,
            marker="^",
            label="Veg loss in mining buffer (km²)",
        )
        plt.grid(True)
        plt.xlabel("Year")
        plt.ylabel("Area (km²)")
        plt.title("Annual Vegetation Loss")
        plt.legend()

        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.show()

    def plot_and_save_yearly_loss_panels(
        vegetation_loss_bool: xr.DataArray,
        out_png: str | Path,
    ):
        years = pd.to_datetime(vegetation_loss_bool.time.values).year
        n = len(years)
        ncols = min(col_wrap, n)
        nrows = int(np.ceil(n / ncols))

        show_title_after("Yearly vegetation loss maps (and saved PNG at 300 dpi)")

        fig, axes = plt.subplots(
            nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows)
        )
        axes = np.atleast_1d(axes).ravel()

        for i, y in enumerate(years):
            ax = axes[i]
            vegetation_loss_bool.isel(time=i).fillna(False).astype(np.uint8).plot.imshow(
                ax=ax, add_colorbar=False
            )
            ax.set_title(str(int(y)))
            ax.set_axis_off()

        # Hide any extra axes
        for j in range(i + 1, len(axes)):
            axes[j].set_axis_off()

        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.show()

    def plot_possible_mining_map_local(
        ds: xr.Dataset,
        veg_loss_in_buffer_mask: xr.DataArray,
        out_png: str | Path,
    ):
        index = "NDVI" if product == "s2" else "RVI"
        bg = ds[index].isel(time=0)

        show_title_after("Possible mining map (and saved PNG at 300 dpi)")

        plt.figure(figsize=(12, 12))
        bg.plot.imshow(cmap="Greys", add_colorbar=False)
        veg_loss_in_buffer_mask.where(veg_loss_in_buffer_mask == 1).plot.imshow(
            cmap=ListedColormap(["Gold"]), add_colorbar=False
        )
        plt.legend([Patch(facecolor="Gold")], ["Possible Mining Site"], loc="upper left")
        plt.title("Possible Mining Areas (Veg loss within buffered mining candidates)")

        out_png = Path(out_png)
        out_png.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=dpi, bbox_inches="tight")
        plt.show()

    def export_yearly_loss_pngs_local(
        vegetation_loss_bool: xr.DataArray,
        out_folder: str | Path,
    ):
        out_folder = Path(out_folder)
        out_folder.mkdir(parents=True, exist_ok=True)
        years = pd.to_datetime(vegetation_loss_bool.time.values).year

        show_title_after("Saving one PNG per year (vegetation loss masks)")

        for i, y in enumerate(years):
            plt.figure(figsize=(7, 7))
            vegetation_loss_bool.isel(time=i).fillna(False).astype(np.uint8).plot.imshow(
                add_colorbar=False
            )
            plt.title(f"Vegetation Loss Mask - {int(y)}")
            plt.axis("off")
            # plt.savefig(out_folder / f"veg_loss_{int(y)}.png", dpi=dpi, bbox_inches="tight")
            plt.close()

    # ---- Run pipeline ----
    out_dir = _ensure_dir(out_dir)

    show_title_after("1) Loading AOI vector")
    gdf, geom = load_vector_file(vector_file)

    show_title_after("2) Loading imagery and WOfS")
    ds, water_frequency_sum, mask = process_data(
        gdf, geom, start_year, end_year, product=product
    )

    show_title_after("3) Calculating vegetation loss")
    veg_loss_bool, veg_loss_sum, change, thr = calculate_vegetation_loss(
        ds, product=product, threshold=threshold
    )
    show_title_after(f"Vegetation loss threshold used: {thr}")

    show_title_after("4) Screening possible mining areas")
    base_mining_mask, buffered_mining_mask, veg_loss_in_buffer_mask = possible_mining_masks(
        veg_loss_sum, water_frequency_sum, ds, buffer_m=buffer_m
    )

    show_title_after("5) Building summary tables")
    df_yearly, df_meta = build_summary_table(
        ds, veg_loss_bool, veg_loss_in_buffer_mask, product=product
    )
    show_df_after(df_meta, "Summary metadata (also saved to CSV)")
    show_df_after(df_yearly, "Summary by year (also saved to CSV)")

    # ---- Export CSV ----
    show_title_after("6) Exporting CSV tables")
    df_yearly.to_csv(out_dir / "surface_mining_summary_by_year.csv", index=False)
    df_meta.to_csv(out_dir / "surface_mining_summary_meta.csv", index=False)

    # ---- Export GeoTIFFs ----
    show_title_after("7) Exporting GeoTIFF outputs")
    save_dataarray_geotiff(
        water_frequency_sum.astype(np.float32),
        out_dir / "water_frequency_sum.tif",
        nodata=-9999.0,
        dtype="float32",
    )
    save_dataarray_geotiff(
        veg_loss_sum.astype(np.uint16),
        out_dir / "vegetation_loss_sum.tif",
        nodata=0,
        dtype="uint16",
    )
    save_dataarray_geotiff(
        base_mining_mask.astype(np.uint8),
        out_dir / "possible_mining_base_mask.tif",
        nodata=0,
        dtype="uint8",
    )
    save_dataarray_geotiff(
        buffered_mining_mask.astype(np.uint8),
        out_dir / "possible_mining_buffer_mask.tif",
        nodata=0,
        dtype="uint8",
    )
    save_dataarray_geotiff(
        veg_loss_in_buffer_mask.astype(np.uint8),
        out_dir / "veg_loss_in_mining_buffer_mask.tif",
        nodata=0,
        dtype="uint8",
    )

    if export_yearly_loss_geotiffs:
        show_title_after("Exporting yearly vegetation loss masks as GeoTIFFs")
        save_time_stack_geotiffs(
            veg_loss_bool.fillna(False).astype(np.uint8),
            out_dir / "yearly_veg_loss_masks",
            prefix="veg_loss",
            nodata=0,
            dtype="uint8",
        )

    # ---- Quick-look RGB ----
    show_title_after("8) Quick-look RGB composites")
    
    # Display (as before)
    if product == "s2":
        rgb(ds, col="time", col_wrap=len(ds.time.values))
    else:
        med_s1 = ds[["vv", "vh", "vh/vv"]].median()
        rgb(
            ds[["vv", "vh", "vh/vv"]] / med_s1,
            bands=["vv", "vh", "vh/vv"],
            col="time",
            col_wrap=len(ds.time.values),
        )
    
    # NEW: Save each year’s RGB as an individual PNG (dpi=300)
    show_title_after("Saving RGB composites per year (PNG, dpi=300)")
    
    rgb_out_dir = out_dir / "rgb_yearly_pngs"
    rgb_out_dir.mkdir(parents=True, exist_ok=True)
    
    years = pd.to_datetime(ds.time.values).year
    
    for i, y in enumerate(years):
        if product == "s2":
            # Make a single-year RGB figure
            fig, ax = plt.subplots(figsize=(7, 7))
            rgb(ds, index=[i], ax=ax)  # uses red/green/blue from gm_s2_annual
            ax.set_title(f"RGB Composite - {int(y)}")
            fig.savefig(rgb_out_dir / f"rgb_{int(y)}.png", dpi=dpi, bbox_inches="tight")
            plt.close()
        else:
            # Sentinel-1 RGB using vv, vh, vh/vv (normalized by median)
            fig, ax = plt.subplots(figsize=(7, 7))
            med_s1 = ds[["vv", "vh", "vh/vv"]].median()
            rgb(
                (ds[["vv", "vh", "vh/vv"]] / med_s1),
                bands=["vv", "vh", "vh/vv"],
                index=[i],
                ax=ax,
            )
            ax.set_title(f"S1 RGB (vv, vh, vh/vv) - {int(y)}")
            fig.savefig(rgb_out_dir / f"s1_rgb_{int(y)}.png", dpi=dpi, bbox_inches="tight")
            plt.close()

    # ---- Save figures at dpi=300 ----
    show_title_after("9) Saving figures (PNG, dpi=300)")
    plot_possible_mining_map_local(
        ds,
        veg_loss_in_buffer_mask,
        out_png=out_dir / "Possible_Mining.png",
    )

    plot_and_save_veg_loss_timeseries(
        ds,
        veg_loss_bool,
        veg_loss_in_buffer_mask,
        out_png=out_dir / "veg_loss_timeseries.png",
    )

    # plot_and_save_yearly_loss_panels(
    #     veg_loss_bool,
    #     out_png=out_dir / "veg_loss_yearly_panels.png",
    # )

    # if export_yearly_loss_pngs:
    #     export_yearly_loss_pngs_local(
    #         veg_loss_bool,
    #         out_folder=out_dir / "yearly_veg_loss_pngs",
    #     )

    show_title_after("Done ✅ Outputs written to:")
    show_title_after('All outputs are saved in the result foler')

    return

