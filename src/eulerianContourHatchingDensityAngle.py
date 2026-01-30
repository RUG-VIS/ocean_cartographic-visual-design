import xarray as xr
import numpy as np
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from colorpals import colour_scales
from matplotlib.collections import LineCollection
from scipy.interpolate import RegularGridInterpolator
import numpy as np
import scipy.ndimage as ndi

# -----------------------------------------------------------------------------
# USER DEFINED PARAMETERS
# -----------------------------------------------------------------------------
input_netcdf = "shorelinetest2.nc"   # Path to input elevation data
input_netcdf2 = "metoffice_foam1_amm7_NWS_SSC_hi20220101.nc"   # Path to input flow data if needed
chosen_time = 0  # Time index to choose from the flow data
output_netcdf = "speedcontour_flow3.nc"  # Output netCDF for signed distance field
output_image = "DensityAnglehatching2.png"  # Output image file for the plot
shoreline_tol = 35  # Tolerance to treat elevation≈0 as shoreline
contour_constant = 0.08  # Contour interval for the signed distance field
num_contours = 4  # Number of contours to plot
plot_title = "Flow as Contour Density"  # Title for the plot
#weird stubby effect when chunk size smaller than the dash length
chunk_size = 3 # Number of points to use for each sub-segment
scale_factor = 3  # Scale factor for dash length
# -----------------------------------------------------------------------------

def theta(d, s, e, h, min_step=1):
    """
    Return a positive increment based on floor(...) unless
    that floor is zero or negative. In that case, we fall back to min_step.
    """
    if d == 0:
        return min_step
    val = (s*d)**e + h
    floored = np.floor(val) - h
    if floored <= 0:
        return min_step
    return (floored**(1/e))/s

# To handle negative distances explicitly:
def signed_theta(d, s, e, h):
    result = theta(np.abs(d), s, e, h)
    return -result if d < 0 else result

# Parameters for theta 

# step in distance units for each contour level
s = 5
# exponent for the distance steps
e = 0.5
# phase shift typically 0
h = 0
# line thickness typically 0.01
t = 0.05

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

    # Identify the shoreline by sign-change from neg to pos the morphological erosion
    shoreline_mask = np.where(elev >=0, 1, 0)
    erosion = ndi.binary_erosion(shoreline_mask, structure=np.ones((3, 3)))
    shoreline_mask = shoreline_mask & ~erosion

    # Visualize the shoreline mask
    '''
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(lon, lat, shoreline_mask, cmap='Greys', shading='auto')
    plt.title("Shoreline Mask")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label='Shoreline (True/False)')
    plt.show()
    '''

    # Compute the distance transform 
    dist_to_shore = distance_transform_edt(shoreline_mask == 0, sampling=(np.abs(lat[1] - lat[0]), np.abs(lon[1] - lon[0])))
    
    #  Assign sign based on whether elevation is above or below zero
    signed_dist = dist_to_shore.copy()
    signed_dist[elev < 0] *= -1  

    # Visualize the signed distance field
    '''
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(lon, lat, signed_dist, cmap='coolwarm', shading='auto')
    plt.title("Signed Distance Field")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.colorbar(label='Signed Distance [m]')
    plt.show()
    #'''

    #Replace NaNs in u,v with 0
    u = u.fillna(0)
    v = v.fillna(0)

    # Interpolate flow data to match elevation grid
    u = u.interp(latitude=lat, longitude=lon, method="linear")
    v = v.interp(latitude=lat, longitude=lon, method="linear")
    # Set u, v = 0 where we have the shoreline, and where the signed distance is positive (land so no flow)
    u.values[shoreline_mask] = 0.0
    v.values[shoreline_mask] = 0.0
    u.values[signed_dist > 0] = 0.0
    v.values[signed_dist > 0] = 0.0

    # Compute the magnitude of the flow field
    flow_mag = np.sqrt(u**2 + v**2)

    # Normalisation for good colour gradient
    
    vmin = np.min(v)
    vmax = np.max(v)  
    umin = np.min(u)
    umax = np.max(u)
    speed_max = np.max(flow_mag)
    speed_min = 0
    
    # For u data normalization:
    abs_max_u = max(np.abs(umin), np.abs(umax))
    normu = TwoSlopeNorm(vmin=-abs_max_u, vcenter=0, vmax=abs_max_u)

    # For v data normalization:
    abs_max_v = max(np.abs(vmin), np.abs(vmax))
    normv = TwoSlopeNorm(vmin=-abs_max_v, vcenter=0, vmax=abs_max_v)

    # norm for flow magnitude
    norm_speed = Normalize(vmin=speed_min, vmax=speed_max)

    #################################################################### Hatch Generation ########################################################

    fig, ax = plt.subplots(figsize=(10, 8))
  # Create a mask for water regions (signed_dist < 0) and non-nan flow
    water_mask = (signed_dist < 0) & (~np.isnan(flow_mag.values))
    rows, cols = np.where(water_mask)
    
    # Normalize flow magnitude to [0, 1]
    max_flow = np.nanmax(flow_mag.values)
    if max_flow == 0:
        flow_norm = np.zeros_like(flow_mag.values)
    else:
        flow_norm = flow_mag.values / max_flow
    
    # Generate random values for each grid point
    np.random.seed(42)  # For reproducibility
    rand_vals = np.random.rand(*flow_norm.shape)
    
    # Select points where rand_vals < flow_norm (higher probability where flow is stronger)
    selected = (rand_vals < flow_norm) & water_mask
    selected_rows, selected_cols = np.where(selected)
    
    # Prepare to collect line segments
    all_segments = []
    dash_length = 0.006  
    
    for r, c in zip(selected_rows, selected_cols):
        # Get coordinates
        lon_val = lon[c].item()
        lat_val = lat[r].item()
        
        # Get flow components
        u_val = u.values[r, c]
        v_val = v.values[r, c]
        
        # Calculate angle in radians
        angle = np.arctan2(v_val, u_val)
        
        # Compute endpoints of the dash
        dx = (dash_length / 2) * np.cos(angle)
        dy = (dash_length / 2) * np.sin(angle)
        
        start = (lon_val - dx, lat_val - dy)
        end = (lon_val + dx, lat_val + dy)
        all_segments.append([start, end])
    
    # Create LineCollection
    lc = LineCollection(
        all_segments,
        colors="k",
        linewidths=1.0,
        linestyles='solid'
    )
    ax.add_collection(lc)

    #####################################################################  Plotting ########################################################


    # Plot flow magnitude as a background
    cs_u = ax.pcolormesh(
        lon,
        lat,
        flow_mag,
        shading='gouraud',
        cmap='Oranges',
        norm=norm_speed,
        alpha=0.6  # Adjust alpha as desired
    )
    cbar_u = fig.colorbar(cs_u, ax=ax, label='Flow Magnitude')
   
    ax.contour(lon, lat, signed_dist, levels=[0], colors='black', linewidths=0.8, linestyles='solid')

    ax.set_title(plot_title)
    ax.set_xlabel('Longitude [°]')
    ax.set_ylabel('Latitude [°]')
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)

    # Save to NetCDF with projected coordinates
    '''
    ds_out = xr.Dataset({
        'u_eastward': (('lat', 'lon'), u.values),
        'v_northward': (('lat', 'lon'), v.values),
        'signed_distance': (('lat', 'lon'), signed_dist)
    }, coords={'lat': lat, 'lon': lon})
    ds_out.to_netcdf(output_netcdf)
    #'''
if __name__ == "__main__":
    main()