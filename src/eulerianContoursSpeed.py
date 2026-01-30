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
input_netcdf = "shorelinetest3.nc"   # Path to input elevation data
input_netcdf2 = "metoffice_foam1_amm7_NWS_SSC_hi20220101.nc"   # Path to input flow data if needed
chosen_time = 0  # Time index to choose from the flow data
output_netcdf = "speedcontour_flow3.nc"  # Output netCDF for signed distance field
output_image = "speedcontour_flow3.png"  # Output image file for the plot
shoreline_tol = 35  # Tolerance to treat elevation≈0 as shoreline
contour_constant = 0.2  # Contour interval for the signed distance field
num_contours = 10  # Number of contours to plot
plot_title = "Flow as Contour Density"  # Title for the plot
#weird stubby effect when chunk size smaller than the dash length
chunk_size = 80 # Number of points to use for each sub-segment
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
    #################################################################### Contour Generation ########################################################
    
    fig, ax = plt.subplots(figsize=(10, 8))
    '''
    ## Generate contour levels based on signed_dist values
    min_dist = np.nanmin(signed_dist)
    current_level = min_dist
    filtered_levels = []

    while current_level < 0:
        filtered_levels.append(current_level)
        spacing = theta(abs(current_level), s, e, h, min_step=0.01)

        # If spacing got computed as <= 0 for any reason, break to avoid infinite loop 
        if spacing <= 0:
            break
        
        next_level = current_level + spacing
        if next_level >= 0:
            break  
        current_level = next_level
    '''
    #invisible contour plot to get the contour shape
    #CS = plt.contour(lon, lat, signed_dist, levels=filtered_levels, colors="none", linewidths=0.8)
    # Define contour levels based on flow magnitude
    contour_levels = np.linspace(float(flow_mag.min()), float(flow_mag.max()), num_contours) # change this so zero isnt included

    # Generate contours based on flow speed instead of signed distance
    CS = ax.contour(lon, lat, flow_mag, levels=contour_levels, colors="none", linewidths=0.8)
    
    '''
    if not filtered_levels:
        print("Warning: No valid contour levels found!")
        return
    '''

    # Prepare to build custom line segments for flow field 

    # Create a 2D interpolator for flow magnitude 
    flow_interpolator = RegularGridInterpolator(
        (lat, lon),           
        flow_mag.values,       
        method='linear',      
        bounds_error=False,    
        fill_value=np.nan      
    )

    all_segments = []   # Will hold arrays of shape (M,2) for each sub-segment
    all_styles   = []   # Will hold dash styles for each sub-segment
    max_flow = np.max(flow_mag)

    def chunk_by_distance(points, max_length):
        """
        Split a line (array of [y, x]) into sub-segments
        with each sub-segment up to ~<= max_length in total distance.
        """
        sub_segments = []
        start_idx = 0
        accumulated_dist = 0.0

        for i in range(1, len(points)):
            dx = points[i, 1] - points[i-1, 1]
            dy = points[i, 0] - points[i-1, 0]
            dist = np.hypot(dx, dy)

            if accumulated_dist + dist > max_length:
                # take the slice from start_idx up to i
                sub_segments.append(points[start_idx:i+1])
                start_idx = i
                accumulated_dist = 0.0
            else:
                accumulated_dist += dist

        # leftover
        if start_idx < len(points) - 1:
            sub_segments.append(points[start_idx:])

        return sub_segments


    # Helper function to map flow to dash length
    MIN_DASH = 2.0
    MAX_DASH = 100.0
    MAX_SUBSEG_LEN = 0.02  # tweak this for lat/lon scale

    def flow_to_dash_len(flow_val, max_flow, scale_factor=1.0):
        raw_dash = (1 + 4*(flow_val / max_flow)) * scale_factor
        return max(MIN_DASH, min(raw_dash, MAX_DASH))

    all_segments = []
    all_styles   = []
    max_flow = np.nanmax(flow_mag)  # flow_mag is your xarray DataArray?

    for level_index, level_value in enumerate(CS.levels):
        segs_for_level = CS.allsegs[level_index]
        if not segs_for_level:
            continue

        for seg in segs_for_level:
            if len(seg) < 2:
                continue

            # break into geometry-based sub-segments
            sub_segs = chunk_by_distance(seg, MAX_SUBSEG_LEN)

            cumulative_offset = 0.0

            for xy_sub in sub_segs:
                # if sub-segment is trivial, skip
                if len(xy_sub) < 2:
                    continue

                # get local flow
                points_for_interp = np.column_stack((xy_sub[:,1], xy_sub[:,0]))
                flow_vals = flow_interpolator(points_for_interp)
                flow_vals = flow_vals[~np.isnan(flow_vals)]
                if len(flow_vals) == 0:
                    continue

                local_flow = np.mean(flow_vals)
                dash_len   = flow_to_dash_len(local_flow, max_flow)
                
                style = (cumulative_offset, (dash_len, dash_len))
                all_segments.append(xy_sub)
                all_styles.append(style)

                # measure the sub-segment length
                seg_length = 0.0
                for i in range(len(xy_sub) - 1):
                    dx = xy_sub[i+1,1] - xy_sub[i,1]
                    dy = xy_sub[i+1,0] - xy_sub[i,0]
                    seg_length += np.hypot(dx, dy)

                pattern_len = 2 * dash_len
                if pattern_len > 0:
                    cumulative_offset = (cumulative_offset + seg_length) % pattern_len

    lc = LineCollection(
        all_segments,
        linestyles=all_styles,
        colors="k",
        linewidths=1.0
    )

    #####################################################################  Plotting ########################################################
    ax.add_collection(lc)     

    # Plot u
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