import xarray as xr
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import pyproj  # For coordinate transformations
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap


# -----------------------------------------------------------------------------
# USER DEFINED PARAMETERS
# -----------------------------------------------------------------------------
input_netcdf = "shorelinetest2.nc"   # Path to  input netCDF
output_netcdf = "signed_distance2.nc"  # Output netCDF file for signed distance field
output_image = "exampleDistanceField2.png"  # Output image file for the plot
shoreline_tol = 35  # Tolerance to treat elevation≈0 as shoreline
contour_constant = 0.01  # Contour separation interval for the signed distance field
num_contours = 5  # Number of contours to plot
plot_title = "Distance and Flow Field"  # Title for the plot

# -----------------------------------------------------------------------------

def main():

    # Define a custom colormap with black for the shoreline
    orig_cmap = plt.get_cmap("coolwarm")  # Original colormap
    new_colors = orig_cmap(np.linspace(0, 1, 256))  # Get the colormap's colors
    new_colors[128] = [0, 0, 0, 1]  # Set the middle index to black (shoreline region)
    # Create a new colormap with black for shoreline
    custom_cmap = LinearSegmentedColormap.from_list("custom_cmap", new_colors, N=256)

    # Load dataset for elevation
    ds = xr.open_dataset(input_netcdf)
    lat = ds["lat"]        # 1D array of latitudes
    lon = ds["lon"]        # 1D array of longitudes
    elev = ds["elevation"] # 2D array [lat, lon]

    # Trim the dataset to be a square
    min_size = min(len(lon), len(lat))
    lon = lon[:min_size]
    lat = lat[:min_size]
    elev = elev[:min_size, :min_size]

    # Identify the shoreline (where elevation is ~ 0)
    shoreline_mask = np.isclose(elev, 0.0, atol=shoreline_tol)

    # Plot the shoreline mask if desired

    plt.figure(figsize=(10, 8))
    plt.pcolormesh(lon, lat, shoreline_mask, shading='auto', cmap='binary')
    plt.colorbar(label='Shoreline Mask (1 = Shoreline)')
    plt.title('Shoreline Mask')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig("shoreline_mask.png", dpi=300, bbox_inches="tight")
    #plt.show()

    # Project latitude/longitude to a Cartesian coordinate system
    # We are using UTM zone 29U for Hebrides
    #xx, yy = latlon_to_projected(lat, lon, projection="EPSG:32629")

    # Create a grid of projected coordinates
    #xx_grid, yy_grid = np.meshgrid(xx, yy)

    # Compute the distance transform in projected coordinates
    dist_to_shore = distance_transform_edt(~shoreline_mask, sampling=(np.abs(lat[1] - lat[0]), np.abs(lon[1] - lon[0])))

    #Assign sign based on whether elevation is above or below zero
    signed_dist = dist_to_shore.copy()
    signed_dist[elev < 0] *= -1  # Negative distance for water (elevation < 0)
    signed_dist[shoreline_mask] = 0  # Zero distance at the shoreline

    # Plotting signed distance field

    # Normalisation for good colour gradient
    vmin = np.min(signed_dist)  # Minimum value in the data
    vmax = np.max(signed_dist)  # Maximum value in the data
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

    # Plotting
    plt.figure(figsize=(30, 24))
    plt.pcolormesh(lon, lat, signed_dist, shading='auto', cmap=custom_cmap, norm=norm)
    plt.colorbar(label='Signed Distance (positive = land, negative = water)')

    # Add contours
    #contour_levels = np.arange(-num_contours*contour_constant, 0, contour_constant)  # Define contour levels (e.g., every 100 meters)

    #contours = plt.contour(xx_grid, yy_grid, signed_dist, levels=contour_levels, colors='k', linewidths=0.5)  # Black lines
    # Show contour labels if desired
    # plt.clabel(contours, inline=True, fontsize=8, fmt='%d')  # Label contours with their values

    plt.title(' Distance to Shoreline ')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    #plt.show()

if __name__ == "__main__":
    main()