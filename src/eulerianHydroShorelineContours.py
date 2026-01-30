import xarray as xr
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import pyproj  # For coordinate transformations
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap
from colorpals import colour_scales

# -----------------------------------------------------------------------------
# USER DEFINED PARAMETERS
# -----------------------------------------------------------------------------
input_netcdf = "shorelinetest4.nc"   # Path to input elevation data
input_netcdf2 = "metoffice_foam1_amm7_NWS_SSC_hi20220101.nc"   # Path to input flow data if needed
chosen_time = 0  # Time index to choose from the flow data
output_netcdf = "distanceandflow4.nc"  # Output netCDF for signed distance field
output_image = "distance_and_flow4.png"  # Output image file for the plot
shoreline_tol = 35  # Tolerance to treat elevation≈0 as shoreline
contour_constant = 0.1  # Contour interval for the signed distance field
num_contours = 5  # Number of contours to plot
plot_title = "Distance and Flow Field"  # Title for the plot
# -----------------------------------------------------------------------------

def latlon_to_projected(lat, lon, projection="EPSG:32629"):
    """
    Convert latitude/longitude to projected coordinates (e.g., UTM).
    """
    transformer = pyproj.Transformer.from_crs("EPSG:32629", projection, always_xy=True)
    xx, yy = transformer.transform(lon, lat)
    return xx, yy

def main():
    cmap = colour_scales['blue_quadrant_palette_greybase_alpha']

    # Load dataset for elevation
    ds = xr.open_dataset(input_netcdf)
    lat = ds["lat"]        # 1D array of latitudes
    lon = ds["lon"]        # 1D array of longitudes
    elev = ds["elevation"] # 2D array [lat, lon]

    min_size = min(len(lon), len(lat))
    lon = lon[:min_size]
    lat = lat[:min_size]
    elev = elev[:min_size, :min_size]

    # Load dataset for flow data

    ds2 = xr.open_dataset(input_netcdf2)
    u = ds2["uo"].isel(time=chosen_time)  # Eastward velocity (m/s)
    v = ds2["vo"].isel(time=chosen_time)  # Northward velocity (m/s)

    

    # Project latitude/longitude to a Cartesian coordinate system
    # We are using UTM zone 29U for Hebrides
    #xx, yy = latlon_to_projected(lat, lon, projection="EPSG:32629")

    # Create a grid of projected coordinates
    #xx_grid, yy_grid = np.meshgrid(xx, yy)

    # Identify the shoreline (where elevation is ~ 0)
    shoreline_mask = np.isclose(elev, 0.0, atol=shoreline_tol)

    #Replace NaNs in u,v with 0
    u = u.fillna(0)
    v = v.fillna(0)

    # Interpolate flow data to match elevation grid
    # Use the correct dimension names: latitude and longitude
    u = u.interp(latitude=lat, longitude=lon, method="linear")
    v = v.interp(latitude=lat, longitude=lon, method="linear")

    

    # Compute the distance transform in projected coordinates
    dist_to_shore = distance_transform_edt(~shoreline_mask, sampling=(np.abs(lat[1] - lat[0]), np.abs(lon[1] - lon[0])))

    #  Assign sign based on whether elevation is above or below zero
    signed_dist = dist_to_shore.copy()
    signed_dist[elev < 0] *= -1  # Negative distance for water (elevation < 0)
    signed_dist[shoreline_mask] = 0  # Zero distance at the shoreline

    # Check if the dimensions match
    #print(f"Dimensions of elevation data: {xx_grid.shape}, {yy_grid.shape}")
    #print(f"Dimensions of flow data: {u.shape}, {v.shape}")

    # Set u, v = 0 where we have the shoreline, and where the signed distance is positive (land so no flow)
    u.values[shoreline_mask] = 0.0
    v.values[shoreline_mask] = 0.0
    u.values[signed_dist > 0] = 0.0
    v.values[signed_dist > 0] = 0.0



    # Plot the signed distance field

    # Normalisation for good colour gradient
    
    vmin = np.min(v)  # Minimum value in the data
    vmax = np.max(v)  # Maximum value in the data
    umin = np.min(u)
    umax = np.max(u)
    
    # For u data normalization:
    abs_max_u = max(np.abs(umin), np.abs(umax))
    normu = TwoSlopeNorm(vmin=-abs_max_u, vcenter=0, vmax=abs_max_u)

    # For v data normalization:
    abs_max_v = max(np.abs(vmin), np.abs(vmax))
    normv = TwoSlopeNorm(vmin=-abs_max_v, vcenter=0, vmax=abs_max_v)


    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot u
    cs_u = ax.pcolormesh(
        lon,
        lat,
        u,
        shading='gouraud',
        cmap=cmap['U']['colour_scale'],
        norm=normu,
        alpha=0.6  # Adjust alpha as desired
    )
    cbar_u = fig.colorbar(cs_u, ax=ax, label='u (Eastward velocity) [m/s]')

    # Plot v on the same axes
    cs_v = ax.pcolormesh(
        lon,
        lat,
        v,
        shading='gouraud',
        cmap=cmap['V']['colour_scale'],
        norm=normv,
        alpha=0.6  # Adjust alpha as desired
    )
    cbar_v = fig.colorbar(cs_v, ax=ax, label='v (Northward velocity) [m/s]')

    # Plot the distance contours
    contour_levels = np.arange(-num_contours*contour_constant, contour_constant, contour_constant)
    cont = ax.contour(lon, lat, signed_dist, levels=contour_levels, colors='k', linewidths=0.8)
    ax.clabel(cont, inline=True, fontsize=8, fmt='%.3f')

    ax.set_title(plot_title)
    ax.set_xlabel('Longitude [°]')
    ax.set_ylabel('Latitude [°]')
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    #plt.show()

    # Save to NetCDF with projected coordinates
    ds_out = xr.Dataset({
        'u_eastward': (('lat', 'lon'), u.values),
        'v_northward': (('lat', 'lon'), v.values),
        'signed_distance': (('lat', 'lon'), signed_dist)
    }, coords={'lat': lat, 'lon': lon})
    ds_out.to_netcdf(output_netcdf)

if __name__ == "__main__":
    main()