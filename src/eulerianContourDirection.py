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
output_netcdf = "speedcontour_flow2.nc"  # Output netCDF for signed distance field
output_image = "Contouring_speed_with_Direction_as_Rotation2.png"  # Output image file for the plot
shoreline_tol = 35  # Tolerance to treat elevation≈0 as shoreline
contour_constant = 0.08  # Contour interval for the signed distance field
num_contours = 7  # Number of contours to plot
plot_title = "Flow Magnitude Contoured with Rotation Direction"  # Title for the plot
#weird stubby effect when chunk size smaller than the dash length
chunk_size = 3 # Number of points to use for each sub-segment
scale_factor = 3  # Scale factor for dash length
# -----------------------------------------------------------------------------

def main():

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
    speed_min = 0.1
    
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

    #invisible contour plot to get the contour shape

    contour_levels = np.linspace(float(speed_min), float(speed_max), num_contours) # change this so zero isnt included

    # Generate contours based on flow speed instead of signed distance

    CS = ax.contour(lon, lat, flow_mag, levels=contour_levels, linewidths=0.8, colors='none')


    # Prepare to build custom line segments for flow field 

    # Create a 2D interpolator for u,v, flow magnitude 

    flow_interpolator = RegularGridInterpolator(
        (lat, lon),           
        flow_mag.values,       
        method='linear',      
        bounds_error=False,    
        fill_value=np.nan      
    )

    u_interpolator = RegularGridInterpolator(
    (lat, lon), 
    u.values, 
    method='linear', 
    bounds_error=False, 
    fill_value=np.nan
)

    v_interpolator = RegularGridInterpolator(
        (lat, lon), 
        v.values, 
        method='linear', 
        bounds_error=False, 
        fill_value=np.nan
)

    all_segments = []
    max_flow = np.nanmax(flow_mag)
    dash_length = 0.006  

    for level_index, level_value in enumerate(CS.levels):
        segs_for_level = CS.allsegs[level_index]
        if not segs_for_level:
            continue
        
        for seg in segs_for_level:
            if len(seg) < 2:
                continue
            
            # Sample points along the contour segment
            for i in range(0, len(seg), chunk_size):
                x, y = seg[i, 0], seg[i, 1]
                
                # Interpolate flow magnitude at this point (lat, lon order)
                local_flow = flow_interpolator((y, x))
                local_u = u_interpolator((y, x))
                local_v = v_interpolator((y, x))
                if np.isnan(local_flow):
                    continue
                
                # Compute angle based on flow magnitude (0° to 90°)
                angle = np.arctan2(local_v, local_u)
                
                # Create a dash oriented at this angle
                dx = dash_length * np.cos(angle)
                dy = dash_length * np.sin(angle)
                dash_segment = [
                    [x - dx/2, y - dy/2],
                    [x + dx/2, y + dy/2]
                ]
                all_segments.append(dash_segment)

    # Create LineCollection
    lc = LineCollection(
        all_segments,
        colors="k",
        linewidths=1.0,
        linestyles='solid'
    )

    #####################################################################  Plotting ########################################################

    
    ax.add_collection(lc)

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