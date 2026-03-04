import os
import itertools
import traceback

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import matplotlib.ticker as ticker
import matplotlib.cm as cm
from matplotlib.colors import TwoSlopeNorm, Normalize, LinearSegmentedColormap, LogNorm
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon
from scipy.ndimage import label, binary_dilation, center_of_mass, distance_transform_edt
import scipy.ndimage as ndi 
from scipy.spatial import KDTree
from scipy.interpolate import RegularGridInterpolator, interp1d

import logging
# logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Matplotlib Global Settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'axes.edgecolor': 'black',
    'axes.labelcolor': 'black',
    'text.color': 'black',
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})
plot_dpi = 300
skip_title_plotting = True

# -----------------------------------------------------------------------------
# General Parameters
# -----------------------------------------------------------------------------
parchment_file = "parchment_v5.png"  # "parchment_v1.jpg"
bathytopograhy_dir = "/media/christian/MyPassport/data/hydrodynamic/ENWS/"
bathytopograhy_file = "gebco_2025_n59.03_s57.65_w-7.65_e-6.05.nc"   # Path to input elevation data; input_netcdf
# input_netcdf2 = "metoffice_foam1_amm7_NWS_SSC_hi20220101.nc"   # Path to input flow data
currents_dir = "/media/christian/MyPassport/data/hydrodynamic/ENWS/reanalysis2D-2024/currents/"
currents_file = "metoffice_foam1_amm7_NWS_CUR_b20240103_dm20240101.nc"  # input_netcdf2
# input_netcdf3 = "ThesisTemps.nc"   # Path to input  temperature data
temperature_dir = "/media/christian/MyPassport/data/hydrodynamic/ENWS/reanalysis2D-2024/temperature/"
temperature_file = "metoffice_foam1_amm7_NWS_TEM_b20240103_dm20240101.nc"  # input_netcdf3
output_dir = "/media/christian/My Passport/Documents/Papers/ISPRS2026/revision_flowhatches"
# output_dir = "/media/christian/My Passport/Documents/Papers/ISPRS2026/revision_flowcontours"
# output_dir = "/media/christian/My Passport/Documents/Papers/ISPRS2026/revision_stipples"

def round_to_nice(x):
    magnitude = 10 ** np.floor(np.log10(x))
    return np.round(x / magnitude) * magnitude

#---------------------------------------------------------------------------
# Scale Bar Function
#---------------------------------------------------------------------------

def add_scale_bar(ax, lon_range, lat_range, location='lower right'):
    """
    Add a scale bar to the map with rounded distance values,
    ensuring it stays within the plot boundaries.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the scale bar to
    lon_range : float or DataArray
        Longitude range of the map
    lat_range : float or DataArray
        Latitude range of the map
    location : str, optional
        Location of the scale bar ('lower right', 'lower left', 'upper right', 'upper left')
    
    Returns:
    --------
    tuple
        The scale distance in degrees and kilometers
    """
    
    # Convert DataArray to float if needed
    if hasattr(lon_range, 'item'):
        lon_range = lon_range.item()
    if hasattr(lat_range, 'item'):
        lat_range = lat_range.item()
    
    # Calculate a nice round distance for scale bar (approx 10% of the plot width)
    raw_scale_distance = lon_range * 0.1
    
    # Convert to approximate km (111 km per degree) -> CK: ??? - what does this have to do in the bar ? or is it the scale metre ?
    scale_km_raw = raw_scale_distance * 111
    
    # Round to nearest 5 or 10 km based on size
    if scale_km_raw < 50:
        scale_km = np.round(scale_km_raw / 5) * 5
        if scale_km < 1: 
            scale_km = 1
    else:
        scale_km = np.round(scale_km_raw / 10) * 10
    
    scale_distance = scale_km / 111
    
    # Get current axes limits
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    
    # Calculate margins to keep scale bar within bounds
    x_margin = lon_range * 0.10 
    y_margin = lat_range * 0.10  
    
    # Position for scale bar based on location
    if location.lower() == 'lower right':
        x_start = x_max - scale_distance - x_margin
        y_pos = y_min + y_margin
    elif location.lower() == 'lower left':
        x_start = x_min + x_margin
        y_pos = y_min + y_margin
    elif location.lower() == 'upper right':
        x_start = x_max - scale_distance - x_margin
        y_pos = y_max - y_margin
    elif location.lower() == 'upper left':
        x_start = x_min + x_margin
        y_pos = y_max - y_margin
    else:
        # Default to lower right
        x_start = x_max - scale_distance - x_margin
        y_pos = y_min + y_margin
    
    # Draw scale bar
    ax.plot([x_start, x_start + scale_distance], [y_pos, y_pos], 
            color='black', linewidth=2, solid_capstyle='butt')
    
    # Add text label
    if scale_km >= 1:
        label = f"{int(scale_km)} km" 
    else:
        label = f"{int(scale_km*1000)} m"
    
    # Position text label above the center of the scale bar
    ax.text(x_start + scale_distance/2, y_pos + y_margin*0.25, 
            label, ha='center', va='bottom', fontsize=8, 
            bbox=dict(facecolor='white', alpha=0.7, pad=2))
    
    return (scale_distance, scale_km)

#---------------------------------------------------------------------------
#Rose Function
#---------------------------------------------------------------------------

def add_compass_rose(ax, lon, lat, signed_dist, min_size=0.01, position=None, 
                     style='traditional', size_factor=0.04, color='black',
                     border_color='white', alpha=0.8, label_color='black',
                     shoreline_buffer=0.3):
    """
    Add a compass rose to a suitable land area on the map.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add the compass rose to
    lon : xarray.DataArray
        Longitude coordinates
    lat : xarray.DataArray
        Latitude coordinates
    signed_dist : numpy.ndarray
        Signed distance field from shoreline (positive on land)
    min_size : float, optional
        Minimum size for land areas to be considered for compass rose placement
        as a fraction of the map extent, default is 0.05 (5%)
    position : tuple or None, optional
        Optional (lon, lat) position to force placement, default is None (automatic)
    style : str, optional
        Style of compass rose: 'traditional', 'simple', or 'nautical', default is 'traditional'
    size_factor : float, optional
        Size of the compass rose as a fraction of the map extent, default is 0.1 (10%)
    color : str, optional
        Main color of the compass rose, default is 'black'
    border_color : str, optional
        Border/background color for compass rose, default is 'white'
    alpha : float, optional
        Transparency of the compass rose, default is 0.8
    label_color : str, optional
        Color for the cardinal direction labels, default is 'black'
    shoreline_buffer : float, optional
        Buffer distance from shoreline as a multiple of the rose size, default is 0.5
        Higher values ensure more distance from shorelines
    
    Returns:
    --------
    bool
        True if compass rose was added, False otherwise
    """
    
    # Calculate map extents for sizing
    lon_range = lon.max().item() - lon.min().item()
    lat_range = lat.max().item() - lat.min().item()
    avg_range = (lon_range + lat_range) / 2
    
    rose_size = avg_range * size_factor
    
    # Calculate minimum buffer distance in grid cells
    lon_cells = lon.size
    lat_cells = lat.size
    lon_cell_size = lon_range / lon_cells
    lat_cell_size = lat_range / lat_cells
    avg_cell_size = (lon_cell_size + lat_cell_size) / 2
    
    # Convert rose size to grid cells
    rose_size_cells = rose_size / avg_cell_size
    
    # Calculate required buffer in grid cells (rose radius + buffer)
    buffer_cells = int(rose_size_cells * (1 + shoreline_buffer))
    
    # Place at specified position if given
    if position is not None:
        rose_lon, rose_lat = position
        
        # Convert position to grid indices
        lon_idx = np.abs(lon - rose_lon).argmin().item()
        lat_idx = np.abs(lat - rose_lat).argmin().item()
        
        # Check if the provided position has enough buffer from shoreline
        land_mask = signed_dist > 0
        
        # Calculate distance from shoreline
        dist_from_shore = distance_transform_edt(land_mask)
        
        if dist_from_shore[lat_idx, lon_idx] < buffer_cells:
            print(f"Warning: Provided position is too close to shoreline (buffer={buffer_cells} cells required).")
            print(f"Distance from shore: {dist_from_shore[lat_idx, lon_idx]:.1f} cells.")
            print("Consider increasing position distance from shore or decreasing rose size.")
        
    # Otherwise, find a suitable land area
    else:

        # Create land mask (positive values in signed_dist are land)
        land_mask = signed_dist > 0
        
        # Calculate distance from shoreline for all land pixels
        dist_from_shore = distance_transform_edt(land_mask)
        
        # Create mask of areas with sufficient buffer
        buffer_mask = dist_from_shore >= buffer_cells
        
        # Dilation to connect nearby areas with sufficient buffer
        buffered_land = binary_dilation(buffer_mask, iterations=2)
        
        # Label connected regions
        labeled_mask, num_regions = label(buffered_land)
        
        if num_regions == 0:
            print("No land regions found for compass rose placement.")
            return False
        
        # Sizes, centers, and min distances of land regions
        region_sizes = []
        region_centers = []
        region_min_dists = [] 
        
        for i in range(1, num_regions + 1):
            region = labeled_mask == i
            region_size = np.sum(region)
            
            # Calculate position as center of mass
            r, c = center_of_mass(region)
            # Convert to lon/lat
            center_lat = lat[int(r)].item()
            center_lon = lon[int(c)].item()
            
            # Get minimum distance from shore in this region
            region_dist = dist_from_shore[region]
            min_dist = np.min(region_dist) if region_dist.size > 0 else 0
            
            # Store 
            region_sizes.append(region_size)
            region_centers.append((center_lon, center_lat))
            region_min_dists.append(min_dist)
        
        # Calculate minimum region size in pixels
        map_pixels = land_mask.size
        min_region_pixels = map_pixels * min_size
        
        # Find suitable regions (large enough and with sufficient buffer)
        suitable_regions = []
        for i, (size, center, min_dist) in enumerate(zip(region_sizes, region_centers, region_min_dists)):
            if size >= min_region_pixels and min_dist >= buffer_cells * 0.8:  # Allow slight tolerance
                suitable_regions.append((i, size, center, min_dist))
        
        if not suitable_regions:
            print(f"No land regions of sufficient size (>={min_size*100:.1f}% of map) and buffer found.")
            print("Trying regions with reduced buffer requirements...")
            
            # Try again with reduced buffer requirement
            suitable_regions = [(i, size, center, min_dist) 
                               for i, (size, center, min_dist) in 
                               enumerate(zip(region_sizes, region_centers, region_min_dists))
                               if size >= min_region_pixels]
        
        if not suitable_regions:
            print("No suitable land regions found for compass rose placement.")
            return False
        
        # Sort by combination of size and buffer distance (prioritizing both)
        suitable_regions.sort(key=lambda x: (x[1] * x[3]), reverse=True)
        
        
        _, _, (rose_lon, rose_lat), min_dist = suitable_regions[0]
        
        print(f"Found suitable region with size ranking {_} and distance from shore: {min_dist:.1f} cells (buffer required: {buffer_cells})")
    
    # Compass rose styles
    def draw_traditional_rose(center_x, center_y, size):
        """Draw a traditional 8-point compass rose"""
        # Main cross
        main_size = size
        secondary_size = size * 0.7
        
        # Create main cross (N-S, E-W)
        for angle in [0, 90, 180, 270]:
            # Main pointer
            x = center_x + main_size * np.sin(np.radians(angle))
            y = center_y + main_size * np.cos(np.radians(angle))
            ax.plot([center_x, x], [center_y, y], color=color, linewidth=2, 
                    solid_capstyle='round', alpha=alpha,
                    path_effects=[path_effects.withStroke(linewidth=3, foreground=border_color)])
            
            # Label
            if angle == 0:
                label = 'N'
                va, ha = 'bottom', 'center'
                yoffset = main_size * 0.1
            elif angle == 90:
                label = 'E'
                va, ha = 'center', 'left'
                yoffset = 0
            elif angle == 180:
                label = 'S'
                va, ha = 'top', 'center'
                yoffset = -main_size * 0.1
            else:  # 270
                label = 'W'
                va, ha = 'center', 'right'
                yoffset = 0
                
            label_x = center_x + (main_size * 1.2) * np.sin(np.radians(angle))
            label_y = center_y + (main_size * 1.2) * np.cos(np.radians(angle)) + yoffset
            
            text = ax.text(label_x, label_y, label, fontsize=12, color=label_color, 
                        va=va, ha=ha, fontweight='bold', alpha=alpha)
            text.set_path_effects([path_effects.withStroke(linewidth=3, foreground=border_color)])
        
        # Create intercardinal points (NE, SE, SW, NW)
        for angle in [45, 135, 225, 315]:
            # Secondary pointer
            x = center_x + secondary_size * np.sin(np.radians(angle))
            y = center_y + secondary_size * np.cos(np.radians(angle))
            ax.plot([center_x, x], [center_y, y], color=color, linewidth=1.5, 
                    solid_capstyle='round', alpha=alpha,
                    path_effects=[path_effects.withStroke(linewidth=2.5, foreground=border_color)])
        
        # Decorative circle at center
        inner_circle = Circle((center_x, center_y), size * 0.15, 
                             facecolor=border_color, edgecolor=color, 
                             linewidth=1.5, alpha=alpha, zorder=10)
        ax.add_patch(inner_circle)
        
        # Add star points
        star_size = size * 0.85
        for angle in range(0, 360, 15):
            if angle % 45 == 0:  # Skip cardinal and intercardinal points
                continue
                
            # Create alternating long and short points
            point_size = star_size * (0.7 if angle % 30 == 0 else 0.5)
            
            x = center_x + point_size * np.sin(np.radians(angle))
            y = center_y + point_size * np.cos(np.radians(angle))
            
            ax.plot([center_x, x], [center_y, y], color=color, linewidth=0.8, 
                    solid_capstyle='round', alpha=alpha * 0.8,
                    path_effects=[path_effects.withStroke(linewidth=1.8, foreground=border_color)])
    
    def draw_simple_rose(center_x, center_y, size):
        """Draw a simple 4-point compass rose"""
        # Draw outer circle
        outer_circle = Circle((center_x, center_y), size * 0.8, 
                             facecolor='none', edgecolor=color, 
                             linewidth=1.5, alpha=alpha)
        ax.add_patch(outer_circle)
        
        # Draw crosshairs
        for angle in [0, 90, 180, 270]:
            # Draw line from center to edge
            x = center_x + size * np.sin(np.radians(angle))
            y = center_y + size * np.cos(np.radians(angle))
            ax.plot([center_x, x], [center_y, y], color=color, linewidth=1.5, 
                    solid_capstyle='butt', alpha=alpha)
            
            # Add cardinal direction label
            if angle == 0:
                label = 'N'
                va, ha = 'bottom', 'center'
            elif angle == 90:
                label = 'E'
                va, ha = 'center', 'left'
            elif angle == 180:
                label = 'S'
                va, ha = 'top', 'center'
            else:  # 270
                label = 'W'
                va, ha = 'center', 'right'
                
            label_x = center_x + (size * 1.1) * np.sin(np.radians(angle))
            label_y = center_y + (size * 1.1) * np.cos(np.radians(angle))
            
            text = ax.text(label_x, label_y, label, fontsize=10, color=label_color, 
                        va=va, ha=ha, fontweight='bold', alpha=alpha)
            text.set_path_effects([path_effects.withStroke(linewidth=2, foreground=border_color)])
    
    def draw_nautical_rose(center_x, center_y, size):
        """Draw a nautical-style compass rose"""
        # Draw outer and inner circles
        outer_circle = Circle((center_x, center_y), size * 0.95, 
                             facecolor='none', edgecolor=color, 
                             linewidth=2, alpha=alpha,
                             path_effects=[path_effects.withStroke(linewidth=3, foreground=border_color)])
        ax.add_patch(outer_circle)
        
        inner_circle = Circle((center_x, center_y), size * 0.2, 
                             facecolor=border_color, edgecolor=color, 
                             linewidth=1.5, alpha=alpha, zorder=10)
        ax.add_patch(inner_circle)
        
        # Draw star pattern for compass points
        # Main cardinal points (N, E, S, W)
        for i, angle in enumerate([0, 90, 180, 270]):

            # Create triangle pointer
            tip_x = center_x + size * 0.95 * np.sin(np.radians(angle))
            tip_y = center_y + size * 0.95 * np.cos(np.radians(angle))
            
            # Calculate perpendicular points for the base of the triangle
            perp_angle1 = angle + 90
            perp_angle2 = angle - 90
            
            base_width = size * 0.15
            base_x1 = center_x + base_width * np.sin(np.radians(perp_angle1))
            base_y1 = center_y + base_width * np.cos(np.radians(perp_angle1))
            base_x2 = center_x + base_width * np.sin(np.radians(perp_angle2))
            base_y2 = center_y + base_width * np.cos(np.radians(perp_angle2))
            
            # Draw filled triangle
            triangle = Polygon(np.array([[base_x1, base_y1], [tip_x, tip_y], [base_x2, base_y2]]),
                            closed=True, facecolor=color, edgecolor='none', alpha=alpha,
                            path_effects=[path_effects.withStroke(linewidth=1, foreground=border_color)])
            ax.add_patch(triangle)
            
            # Cardinal direction label
            if angle == 0:
                label = 'N'
                va, ha = 'bottom', 'center'
                y_offset = size * 0.15
            elif angle == 90:
                label = 'E'
                va, ha = 'center', 'left'
                y_offset = 0
            elif angle == 180:
                label = 'S'
                va, ha = 'top', 'center'
                y_offset = -size * 0.15
            else:  # 270
                label = 'W'
                va, ha = 'center', 'right'
                y_offset = 0
                
            label_x = center_x + (size * 0.6) * np.sin(np.radians(angle))
            label_y = center_y + (size * 0.6) * np.cos(np.radians(angle)) + y_offset
            
            text = ax.text(label_x, label_y, label, fontsize=14, color=label_color, 
                        va=va, ha=ha, fontweight='bold', alpha=alpha)
            text.set_path_effects([path_effects.withStroke(linewidth=3, foreground=border_color)])
        
        # Add smaller tick marks at intercardinal and other points
        for angle in range(0, 360, 15):
            if angle % 90 == 0:  
                continue
                
            # Different lengths based on importance
            if angle % 45 == 0:  
                tick_length = size * 0.2
                width = 1.5
            else:  
                tick_length = size * 0.1
                width = 0.8
                
            inner_x = center_x + (size * 0.7) * np.sin(np.radians(angle))
            inner_y = center_y + (size * 0.7) * np.cos(np.radians(angle))
            
            outer_x = center_x + (size * 0.7 + tick_length) * np.sin(np.radians(angle))
            outer_y = center_y + (size * 0.7 + tick_length) * np.cos(np.radians(angle))
            
            ax.plot([inner_x, outer_x], [inner_y, outer_y], color=color, linewidth=width, 
                    solid_capstyle='butt', alpha=alpha,
                    path_effects=[path_effects.withStroke(linewidth=width+1, foreground=border_color)])
    
    # Draw the chosen style of compass rose
    if style == 'traditional':
        draw_traditional_rose(rose_lon, rose_lat, rose_size)
    elif style == 'simple':
        draw_simple_rose(rose_lon, rose_lat, rose_size)
    elif style == 'nautical':
        draw_nautical_rose(rose_lon, rose_lat, rose_size)
    else:
        print(f"Unknown compass rose style: {style}. Using traditional style.")
        draw_traditional_rose(rose_lon, rose_lat, rose_size)
    
    print(f"Compass rose ({style} style) added at position ({rose_lon:.4f}, {rose_lat:.4f})")
    return True

#---------------------------------------------------------------------------
# Hatch Generation
#---------------------------------------------------------------------------

def hatch(attribute1, attribute1_max=None, attribute1_min=None, attribute2=None, attribute3=None, 
          attribute4=None, attribute5=None, attribute6=None, signed_dist=None, lon=None, lat=None, 
          water_mask=None, elev=None, topography=None, normalized_attribute1=None,
          attribute5_norm=None, attribute5_cmap='RdBu_r', attribute1_label="", attribute2_label="", attribute3_label="",
          attribute4_label="", attribute5_label="", attribute6_label="", auto_scale=False,
          output_image="hatch_output.png", plot_title="Flow as Handdrawn Hatches with Direction Markers",
          use_attribute5_for_color=False, base_density=None, density_scale_factor=0.1, decay_factor=5):  # elev_modified=None (replaced by topography)
    """
    Generate hatches based on multiple attributes with automatic scaling based on geographic extent.
    
    Parameters:
    -----------
    attribute1 : xarray.DataArray
        Primary attribute (maps to hatch density)
    attribute1_max : float, optional
        Maximum value of attribute1, calculated automatically if None
    attribute1_min : float, optional
        Minimum value of attribute1, calculated automatically if None
    attribute2 : xarray.DataArray, optional
        Secondary attribute (maps to x-component of hatch direction)
    attribute3 : xarray.DataArray, optional
        Tertiary attribute (maps to y-component of hatch direction)
    attribute4 : xarray.DataArray, optional
        Fourth attribute (maps to hatch length)
        If None and attribute2/attribute3 provided, calculated as magnitude of velocity components
    attribute5 : numpy.ndarray, optional
        Fifth attribute (can map to hatch color)
        If None and attribute2/attribute3 provided, calculated as divergence
    attribute6 : xarray.DataArray, optional
        Sixth attribute (maps to hatch width)
        If None, width is determined by attribute4 (length)
    signed_dist : numpy.ndarray, optional
        Signed distance field from shoreline
    lon : xarray.DataArray
        Longitude coordinates
    lat : xarray.DataArray
        Latitude coordinates
    water_mask : numpy.ndarray, optional
        Boolean mask for water regions, calculated from signed_dist if None
    elev : xarray.DataArray, optional
        Elevation data
    topography : xarray.DataArray, optional
        Topography data (for display), created from elev if None
    normalized_attribute1 : numpy.ndarray, optional
        Normalized values of attribute1 (0-1 range), calculated automatically if None
    attribute5_norm : matplotlib.colors.Normalize, optional
        Normalization for attribute5 values for color mapping, created automatically if None
    attribute5_cmap : str or matplotlib.colors.Colormap, optional
        Colormap for attribute5, default is 'RdBu_r'
    attribute1_label: str, optional
        Label string of the density attribute. Default: empty string
    attribute2_label: str, optional
        Label string of the zonal velocity attribute. Default: empty string
    attribute3_label: str, optional
        Label string of the meridional velocity attribute. Default: empty string
    attribute4_label: str, optional
        Label string of the hatch length attribute. Default: empty string
    attribute5_label: str, optional
        Label string of the background colourmap attribute. Default: empty string
    attribute6_label: str, optional
        Label string of the hatch width attribute. Default: empty string
    auto_scale: boolean, optional
        Calculate the attribute limits based on percentiles (True) or by the min-max (False). Default: False
    output_image : str, optional
        Path to save the output image, default is "hatch_output.png"
    plot_title : str, optional
        Title for the plot, default is "Flow as Handdrawn Hatches with Direction Markers"
    log_attribute1_scale : bool, optional
        Whether to use logarithmic scale for attribute1, default is True
    attribute1_label : str, optional
        Label for attribute1 on the colorbar, default is "Temperature (°C)"
    use_attribute5_for_color : bool, optional
        Whether to use attribute5 for hatch color, default is False
    base_density : float, optional
        Base density control parameter (0-1), default is 0.1
    decay_factor : float, optional
        Controls how much temperature affects density, default is 5
    """
 
    # Calculate geographic extents for auto-scaling
    lon_range = lon.max().item() - lon.min().item()
    lat_range = lat.max().item() - lat.min().item()
    avg_range = (lon_range + lat_range) / 2
    
    # Auto-calculate base_density if not provided
    if base_density is None:
        # Use the resolution of the grid to determine appropriate density
        lon_res = lon[1].item() - lon[0].item()
        lat_res = lat[1].item() - lat[0].item()
        avg_res = (lon_res + lat_res) / 2
        
        # Adjust density based on data resolution and scale factor
        base_density = density_scale_factor * (avg_res / (avg_range * 0.01))
        
        print(f"Auto-calculated base_density: {base_density:.4f}")
    else:
        print(f"Using provided base_density: {base_density:.4f}")
    
    # Auto-calculate parameters based on coordinate scale
    
    # Hand-drawn effect parameters (auto-scaled)
    curve_intensity = avg_range * 0.002   # 0.1% of average range
    jitter_amount = avg_range * 0.002     # 0.2% of average range
    alpha_variance = 0.1                  # Fixed (percentage of opacity)
    
    # Shoreline buffer (distance to keep hatches away from shore)
    shoreline_buffer = avg_range * 0.01   # 1% of average range
    
    # Non-overlap control parameters (auto-scaled)
    min_distance = avg_range * 0.005      # 0.5% of average range
    buffer_factor = 1.05                  # Fixed (multiplier)
    max_iterations = 3                    # Fixed (number of attempts)
    
    # Stroke properties (auto-scaled)
    max_stroke_length = avg_range * 0.05    # 5% of average range
    min_strokewidth = 0.5                   # Fixed (points)
    max_strokewidth = 1.5                   # Fixed (points)
    width_variance = 0.0                  # Fixed (percentage)
    angle_variance = 10                   # Fixed (degrees)
    length_variance = 0.2                 # Fixed (percentage)
    stroke_color = "black"
    
    # Tuft parameters (auto-scaled)
    toggle_tufts = True                   # Fixed (boolean)
    tuft_size_factor = avg_range * 0.005  # 0.5% of average range
    tuft_color = "black"                  # Fixed (color)
    tuft_min_distance = avg_range * 0.01  # 1% of average range
    
    # Tapering effect parameters (fixed ratios)
    taper_probability = 1.0               # Fixed (probability)
    gap_probability = 0.0                 # Fixed (probability)
    min_taper_width_ratio = 0.8           # Fixed (ratio)
    max_taper_width_ratio = 0.9           # Fixed (ratio)
    taper_points = 10                     # Fixed (count)
    taper_zone = 0.15                     # Fixed (percentage)
    
    # Print auto-scaling parameters
    print(f"Auto-scaling parameters based on geographic extent:")
    print(f"  - Coordinate range: lon={lon_range:.6f}, lat={lat_range:.6f}, avg={avg_range:.6f}")
    print(f"  - Max dash length: {max_stroke_length:.6f}")
    print(f"  - Curve intensity: {curve_intensity:.6f}")
    print(f"  - Jitter amount: {jitter_amount:.6f}")
    print(f"  - Min distance: {min_distance:.6f}")
    print(f"  - Tuft size factor: {tuft_size_factor:.6f}")
    print(f"  - Tuft min distance: {tuft_min_distance:.6f}")

    # legend-related mapping containers
    val_varmod = np.random.uniform(-length_variance, length_variance)
    
    # ==== Create prerequisite data if not provided ==== #
    # Calculate attribute ranges
    attribute1_norm = None
    if attribute1_min is None:
        attribute1_min = np.nanmin(attribute1)
    if attribute1_max is None:
        attribute1_max = np.nanmax(attribute1)
    # Create normalisation for attribute1 if not provided
    if auto_scale:
        # Use robust percentile-based normalisation
        valid_values = attribute1[~np.isnan(attribute1)]
        if len(valid_values) > 0:
            robust_min = np.percentile(valid_values, 2)
            robust_max = np.percentile(valid_values, 98)
            attribute1_norm = Normalize(vmin=robust_min, vmax=robust_max)
        else:
            attribute1_norm = Normalize(vmin=attribute1_min, vmax=attribute1_max)
    else:
        attribute1_norm = plt.Normalize(vmin=attribute1_min, vmax=attribute1_max)

    # Handle normalized_attribute1 if not provided
    if normalized_attribute1 is None:
        attribute1_range = attribute1_max - attribute1_min
        if attribute1_range > 0:
            if isinstance(attribute1.values, np.ndarray):
                normalized_attribute1 = (attribute1.values - attribute1_min) / attribute1_range
            else:
                normalized_attribute1 = (attribute1 - attribute1_min) / attribute1_range
        else:
            normalized_attribute1 = np.zeros_like(attribute1.values)

    # Set up normalisation for zonal velocity
    attribute2_min = None
    attribute2_max = None
    attribute2_norm = None
    if attribute2 is not None:
        if attribute2_min is None:
            attribute2_min = np.nanmin(attribute2)
        if attribute2_max is None:
            attribute2_max = np.nanmax(attribute2)
        # Create normalisation for attribute2 if not provided
        if auto_scale:
            # Use robust percentile-based normalisation
            valid_values = attribute2[~np.isnan(attribute2)]
            if len(valid_values) > 0:
                robust_min = np.percentile(valid_values, 2)
                robust_max = np.percentile(valid_values, 98)
                attribute2_norm = Normalize(vmin=robust_min, vmax=robust_max)
                attribute2_min = robust_min
                attribute2_max = robust_max
            else:
                attribute2_norm = Normalize(vmin=attribute2_min, vmax=attribute2_max)
        else:
            attribute2_norm = plt.Normalize(vmin=attribute2_min, vmax=attribute2_max)

    # Set up normalisation for meridional velocity
    attribute3_min = None
    attribute3_max = None
    attribute3_norm = None
    if attribute3 is not None:
        # Use attribute3 for background coloration
        if attribute3_min is None:
            attribute3_min = np.nanmin(attribute3)
        if attribute3_max is None:
            attribute3_max = np.nanmax(attribute3)
        if auto_scale:
            # Use robust percentile-based normalisation
            valid_values = attribute3[~np.isnan(attribute3)]
            if len(valid_values) > 0:
                robust_min = np.percentile(valid_values, 2)
                robust_max = np.percentile(valid_values, 98)
                attribute3_norm = Normalize(vmin=robust_min, vmax=robust_max)
            else:
                attribute3_norm = Normalize(vmin=attribute3_min, vmax=attribute3_max)
        else:
            attribute3_norm = Normalize(vmin=attribute3_min, vmax=attribute3_max)

    # Set up normalisation for hatch length
    attribute4_min = None
    attribute4_max = None
    attribute4_norm = None
    if attribute4 is not None:
        # Use attribute4 for background coloration
        if attribute4_min is None:
            attribute4_min = np.nanmin(attribute4)
        if attribute4_max is None:
            attribute4_max = np.nanmax(attribute4)
        if auto_scale:
            # Use robust percentile-based normalisation
            valid_values = attribute4[~np.isnan(attribute4)]
            if len(valid_values) > 0:
                robust_min = np.percentile(valid_values, 2)
                robust_max = np.percentile(valid_values, 98)
                attribute4_norm = Normalize(vmin=robust_min, vmax=robust_max)
            else:
                attribute4_norm = Normalize(vmin=attribute4_min, vmax=attribute4_max)
        else:
            attribute4_norm = Normalize(vmin=attribute4_min, vmax=attribute4_max)

    # Set up normalisation for background colourmap
    attribute5_min = None
    attribute5_max = None
    attribute5_norm = None
    if attribute5 is not None:
        # Use attribute5 for background coloration
        if attribute5_min is None:
            attribute5_min = np.nanmin(attribute5)
        if attribute5_max is None:
            attribute5_max = np.nanmax(attribute5)
        if auto_scale:
            # Use robust percentile-based normalisation
            valid_values = attribute5[~np.isnan(attribute5)]
            if len(valid_values) > 0:
                robust_min = np.percentile(valid_values, 2)
                robust_max = np.percentile(valid_values, 98)
                attribute5_norm = Normalize(vmin=robust_min, vmax=robust_max)
            else:
                attribute5_norm = Normalize(vmin=attribute5_min, vmax=attribute5_max)
        else:
            attribute5_norm = Normalize(vmin=attribute5_min, vmax=attribute5_max)

    # Set up normalisation for hatch width
    attribute6_min = None
    attribute6_max = None
    attribute6_norm = None
    if attribute6 is not None:
        # Use attribute6 for background coloration
        if attribute6_min is None:
            attribute6_min = np.nanmin(attribute6)
        if attribute6_max is None:
            attribute6_max = np.nanmax(attribute6)
        if auto_scale:
            # Use robust percentile-based normalisation
            valid_values = attribute6[~np.isnan(attribute6)]
            if len(valid_values) > 0:
                robust_min = np.percentile(valid_values, 2)
                robust_max = np.percentile(valid_values, 98)
                attribute6_norm = Normalize(vmin=robust_min, vmax=robust_max)
            else:
                attribute6_norm = Normalize(vmin=attribute6_min, vmax=attribute6_max)
        else:
            attribute6_norm = Normalize(vmin=attribute6_min, vmax=attribute6_max)

    # create legend labels
    legend_hatchlengths = []
    legend_hatchlengths_labels = []
    legend_maxlen = 6
    litems = [.0, 0.5, 1.0]
    fstelem = -1.0
    uniform_len = False
    if attribute4 is not None:
        fstelem = attribute4.flatten()[0]
        uniform_len = np.allclose(attribute4, fstelem, rtol=np.finfo(np.float32).eps)
    if (attribute4_min is not None) and (attribute4_max is not None) and not uniform_len:
        for factor in litems:
            # legendvalue = factor * max_stroke_length * 0.5 * (1 + val_varmod)
            hatchlen = 1 + (factor * legend_maxlen)
            hatchgap = 1
            attrval = attribute4_min + factor * (attribute4_max - attribute4_min)
            # legend_contourstyles.append((0, (base_dash, base_gap)))
            legend_hatchlengths.append((0, (hatchlen, hatchgap)))
            legend_hatchlengths_labels.append(attrval)
    else:
        legend_hatchlengths.append((0, (legend_maxlen, 1)))
        legend_hatchlengths_labels.append(1.0)
    legend_hatchwidths = []
    legend_hatchwidths_labels = []
    fstelem = -1.0
    uniform_width = False
    if attribute6 is not None:
        fstelem = attribute6.flatten()[0]
        uniform_width = np.allclose(attribute6, fstelem, rtol=np.finfo(np.float32).eps)
    if (attribute6_min is not None) and (attribute6_max is not None) and not uniform_width:
        for factor in [.0, 0.5, 1.0]:
            stroke_width = min_strokewidth + factor * (max_strokewidth - min_strokewidth)
            attrval = attribute6_min + factor * (attribute6_max - attribute6_min)
            legend_hatchwidths.append(stroke_width)
            legend_hatchwidths_labels.append(attrval)
    else:
        legend_hatchwidths.append(1.0)
        legend_hatchwidths_labels.append(1.0)

    # legend_tuftsizes = []  # -> unused
    # cbar_tuft_colours = []  # -> unused
    
    # Create water mask from signed distance if not provided
    if water_mask is None and signed_dist is not None:
        water_mask = (signed_dist < 0)
    elif water_mask is None and elev is not None:
        water_mask = (elev < 0)
    elif water_mask is None:
        # Create a default water mask (all True)
        water_mask = np.ones_like(attribute1, dtype=bool)

    # TODO: what is 'elev_modified' ? are we even using it ? I think: no, we do not.
    # ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== #
    # Create elev_modified from elev if not provided    #
    # ==== ==== ==== ==== ==== ==== ==== ==== ==== ==== #
    # if topography is None and elev is not None:
    #     topography = elev.where(elev <= 0, 0)
    # elif topography is None:
    #     # Create a placeholder elev_modified (all zeros)
    #     topography = np.zeros(attribute1.shape, dtype=np.float32)

    # ==== ==== ==== ==== ==== ==== ==== ==== ==== #
    # Calculate tuft density based on attribute1   #
    # ==== ==== ==== ==== ==== ==== ==== ==== ==== #
    attribute1_factor = np.exp(decay_factor * normalized_attribute1) - 1
    attribute1_factor_normalized = attribute1_factor / np.max(attribute1_factor) if np.max(attribute1_factor) > 0 else attribute1_factor
    prob_matrix = np.where(water_mask, base_density * attribute1_factor_normalized, 0)
    
    # Random point selection (density based on attribute1)
    np.random.seed(8)
    rand_matrix = np.random.rand(*prob_matrix.shape)
    selected_points = (rand_matrix < prob_matrix)
    rows, cols = np.where(selected_points)

    if attribute4 is not None:
        # Apply shoreline buffer if signed_dist is provided
        if signed_dist is not None:
            # Create a buffer zone around the shoreline
            # Exclude points that are too close to land (within shoreline_buffer distance)
            buffer_mask = np.abs(signed_dist) >= shoreline_buffer
            # selected_points = (rand_matrix < prob_matrix) & (np.abs(attribute4.values) >= 0) & buffer_mask
            selected_points = (rand_matrix < prob_matrix) & (np.abs(attribute4) >= 0) & buffer_mask
        else:
            # selected_points = (rand_matrix < prob_matrix) & (np.abs(attribute4.values) >= 0)
            selected_points = (rand_matrix < prob_matrix) & (np.abs(attribute4) >= 0)
    else:
        if signed_dist is not None:
            buffer_mask = np.abs(signed_dist) >= shoreline_buffer
            # TODO: this should be deterministic, otherwise you're visualizing a pattern not backed-up by data
            selected_points = (rand_matrix < prob_matrix) & buffer_mask  # No attribute4 dependency
            rows, cols = np.where(selected_points)

    # Create KDTree for spatial indexing to check for overlaps
    placed_centers = []
    
    # Create a separate KDTree for tuft circles
    placed_tufts = []
    tuft_kdtree = None
    
    # Mix the order to avoid prioritizing certain regions
    indices = np.arange(len(rows))
    np.random.shuffle(indices)
    rows = rows[indices]
    cols = cols[indices]

    # TODO: this is already done during the loading.
    # attribute1_values = attribute1.values.copy()
    # attribute1_values = np.nan_to_num(attribute1_values, nan=attribute1_min)
    
    # Prepare bezier curves
    all_segments = []
    all_linewidths = []
    all_alphas = []
    color = stroke_color
    # all_colors = []
    all_colors = color
    # flow_values = []  # Store attribute4 (flow magnitude) for each segment
    # divergence_values = []  # Store attribute5 (divergence) values for each segment
    
    # Lists for storing tuft circles
    tuft_centers = []
    tuft_sizes = []
    tuft_alphas = []

    for r, c in zip(rows, cols):
        # Get position and attribute values
        lon_center = lon[c].item()
        lat_center = lat[r].item()
        
        # Safely get attribute values
        # if attribute2 is not None and hasattr(attribute2, 'values'):
        if attribute2 is not None and isinstance(attribute3, np.ndarray):
            # attribute2_val = attribute2.values[r, c]
            attribute2_val = attribute2[r, c]
        else:
            attribute2_val = 1.0
            
        # if attribute3 is not None and hasattr(attribute3, 'values'):
        if attribute3 is not None and isinstance(attribute3, np.ndarray):
            # attribute3_val = attribute3.values[r, c]
            attribute3_val = attribute3[r, c]
        else:
            attribute3_val = 0.0
        
        # if attribute4 is not None and hasattr(attribute4, 'values'):
        if attribute4 is not None and isinstance(attribute4, np.ndarray):
            # attribute4_val = attribute4.values[r, c]
            attribute4_val = attribute4[r, c]
            # special case where divergence is used for length, use absolute value  # TODO: use absolute all the time - can't modulate a negative length ...
            # if "divergence (length)" in plot_title.lower():
            #     attribute4_val = np.abs(attribute4_val)
            attribute4_val = np.abs(attribute4_val)
        else:
            attribute4_val = 1.0  # Default value for uniform length
            
        if attribute5 is not None and isinstance(attribute5, np.ndarray):
            attribute5_val = attribute5[r, c]
        else:
            attribute5_val = attribute5
            
        # if attribute6 is not None and hasattr(attribute6, 'values'):
        if attribute6 is not None and isinstance(attribute6, np.ndarray):
            attribute6_val = attribute6[r, c]
        else:
            # attribute6_val = attribute4_val
            attribute6_val = 1.0
        
        # Skip some lines randomly to create gaps
        if np.random.random() < gap_probability:
            continue

        # TODO: change to arctan2(wedge([0,1], [a,b]) => [DONE]
        # Calculate angle from vector direction with random variance
        avec = np.array([.0, .1], dtype=np.float32)
        bvec = np.array([attribute2_val, attribute3_val], dtype=np.float32)
        wedge_ab = np.absolute(avec[0]*bvec[1]-avec[1]*bvec[0])
        dot_ab = avec[0]*bvec[0] + avec[1]*bvec[1]
        # base_angle = np.arctan2(attribute3_val, attribute2_val)
        base_angle = np.arctan2(wedge_ab, dot_ab)
        angle_rad_variance = np.radians(angle_variance)  # why ? if the atan2 is done correctly, this returns radians
        angle = base_angle + np.random.uniform(-angle_rad_variance, angle_rad_variance)
        
        # Dynamic dash length based on attribute4  with random variance
        base_stroke_length = -1.0
        if attribute4 is not None and isinstance(attribute4, np.ndarray):
            # Variable length based on attribute4
            # attr4_max = np.nanmax(attribute4.values)

            # base_stroke_length = (attribute4_val / attr4_max if attr4_max > 0 else 0.5) * max_stroke_length
            base_stroke_length = (np.abs(attribute4_val) / attribute4_max if attribute4_max > 0 else 0.5) * max_stroke_length
        else:
            # Uniform length when attribute4 is None
            base_stroke_length = max_stroke_length * 0.5
            
        stroke_length = base_stroke_length * (1 + np.random.uniform(-length_variance, length_variance))
        
        # Check for nearby centers to avoid overlaps
        overlap = False
        points_to_try = []
        
        # Try a few positions if there's an overlap
        for _ in range(max_iterations):
            # Add position jitter
            jitter_x = np.random.uniform(-jitter_amount, jitter_amount)
            jitter_y = np.random.uniform(-jitter_amount, jitter_amount)
            test_lon = lon_center + jitter_x
            test_lat = lat_center + jitter_y
            
            # Check if point is too close to existing hatches
            if placed_centers:
                kdtree = KDTree(placed_centers)
                dist, _ = kdtree.query((test_lon, test_lat), k=1)
                overlap = dist < (min_distance + stroke_length * buffer_factor)
            
            if not overlap:
                # Center is good, use it
                points_to_try.append((test_lon, test_lat))
                break
            
        # If all attempts resulted in overlaps, skip this hatch
        if not points_to_try:
            continue
            
        # Use the first valid position
        lon_center, lat_center = points_to_try[0]
        
        # Calculate start and end points
        dx = stroke_length * np.cos(angle)
        dy = stroke_length * np.sin(angle)
        start = (lon_center - dx/2, lat_center - dy/2)
        end = (lon_center + dx/2, lat_center + dy/2)
        
        # Calculate curve control points
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        
        # Add perpendicular curvature
        line_dx = end[0] - start[0]
        line_dy = end[1] - start[1]
        length = np.sqrt(line_dx**2 + line_dy**2)
        
        # Calculate perpendicular vector
        if length > 0:
            perp_x = -line_dy / length
            perp_y = line_dx / length
        else:
            perp_x, perp_y = 0, 0
        
        # Apply random curve intensity
        curve_factor = np.random.normal(0, curve_intensity)
        mid_x += perp_x * curve_factor
        mid_y += perp_y * curve_factor
        
        # Create segments for tapered line
        segments = []
        
        # Create segments with varying width
        t_points = np.linspace(0, 1, taper_points)
        points = []

        for t in t_points:
            # Quadratic Bezier formula for position
            x = (1-t)**2 * start[0] + 2*(1-t)*t * mid_x + t**2 * end[0]
            y = (1-t)**2 * start[1] + 2*(1-t)*t * mid_y + t**2 * end[1]
            points.append((x, y))

        # Calculate stroke width based on attribute6 (if provided) or use uniform width
        # TODO: well, it is 'in range' of attribute 6, but not based on it ...
        # if attribute6 is not None and hasattr(attribute6, 'values'):
        if attribute6 is not None and isinstance(attribute6, np.ndarray):
            # Normalize attribute6 (width) value
            # attr6_min = np.nanmin(attribute6.values)
            # attr6_max = np.nanmax(attribute6.values)
            
            if attribute6_min == attribute6_max:
                stroke_width = (min_strokewidth + max_strokewidth)/2.0
            else:
                normalized_width = (attribute6_val - attribute6_min) / (attribute6_max - attribute6_min)
                stroke_width = min_strokewidth + normalized_width * (max_strokewidth - min_strokewidth)
        else:
            # Use a uniform width when attribute6 is not provided
            stroke_width = (min_strokewidth + max_strokewidth)/2.0

        # Apply width variance to the base width
        stroke_width = stroke_width * (1.0 + np.random.uniform(-width_variance, width_variance))

        # Calculate tuft size
        # according to the current calculation, this is constant over all datapoints
        tuft_size = stroke_width * tuft_size_factor
        
        # Check if the tuft would overlap with existing tufts
        tuft_position = (points[0][0], points[0][1])
        tuft_overlap = False
        
        if placed_tufts:
            tuft_kdtree = KDTree(placed_tufts)
            tuft_dist, _ = tuft_kdtree.query(tuft_position, k=1)
            tuft_overlap = tuft_dist < (tuft_min_distance + tuft_size)

         # Check if tuft is too close to shoreline
        if signed_dist is not None:
            # Extract coordinates
            tuft_lon, tuft_lat = tuft_position
            
            # Find grid indices for the tuft position
            # lon_idx = np.argmin(np.abs(lon.values - tuft_lon))
            # lat_idx = np.argmin(np.abs(lat.values - tuft_lat))
            lon_idx = np.argmin(np.abs(lon - tuft_lon))
            lat_idx = np.argmin(np.abs(lat - tuft_lat))
            
            # Check signed distance at this position
            try:
                dist_to_shore = np.abs(signed_dist[lat_idx, lon_idx])
                if dist_to_shore < shoreline_buffer:
                    tuft_overlap = True  # Too close to shore
            except IndexError:
                # If out of bounds, skip this tuft
                tuft_overlap = True
        
        # Only proceed with the line if the tuft doesn't overlap
        if not tuft_overlap or not toggle_tufts:
            # Add the tuft position to our tracking if using tufts
            if toggle_tufts:
                placed_tufts.append(tuft_position)
                
                # Add the tuft circle
                tuft_centers.append(tuft_position)
                tuft_sizes.append(tuft_size)  # TODO: this should be a value, not an array ...
                
                # Use the same alpha as the line for consistency
                # TODO: check with constant alpha
                alpha = 0.8 + np.random.uniform(-alpha_variance, alpha_variance)
                alpha = max(0.4, min(1.0, alpha))  # Keep alpha within reasonable bounds
                tuft_alphas.append(alpha)  # TODO: this should be a value, not an array ...
            else:
                alpha = 0.8 + np.random.uniform(-alpha_variance, alpha_variance)
                alpha = max(0.4, min(1.0, alpha))  # Keep alpha within reasonable bounds
            
            # Create line segments
            for i in range(len(points) - 1):
                segments.append([points[i], points[i+1]])
                
                # Calculate position along the curve (0 at start, 1 at end)
                rel_position = i / (len(points) - 2)
                
                # Define taper zones
                taper_zone_val = taper_zone  # Controls how much of each end gets tapered (15% from each end)
                
                # Determine width factor (full width in middle, tapered at ends)
                taper_ratio = np.random.uniform(min_taper_width_ratio, max_taper_width_ratio)
                
                if rel_position < taper_zone_val:  # Start taper
                    width_factor = taper_ratio + (1.0 - taper_ratio) * (rel_position / taper_zone_val)
                elif rel_position > (1.0 - taper_zone_val):  # End taper
                    normalized_end_pos = (1.0 - rel_position) / taper_zone_val
                    width_factor = taper_ratio + (1.0 - taper_ratio) * normalized_end_pos
                else:  # Middle section (full width)
                    width_factor = 1.0
                
                # Apply width factor to base width
                segment_width = stroke_width * width_factor
                all_linewidths.append(segment_width)
                
                # Store segment
                all_segments.append([points[i], points[i+1]])
                
                # Variable alpha for each segment
                all_alphas.append(alpha)  # TODO: is this necessary ? test without ...

                # Add the color to the list, ensuring it's in the right format
                if not isinstance(color, str) and not isinstance(all_colors, str):
                    all_colors.append(color if isinstance(color, str) else tuple(color))  # TODO: interpolate on a cmap using an input attribute
                
                # flow_values.append(attribute4_val)  # unused
                # divergence_values.append(attribute5_val)  # unused
            
            # Add center to the list of placed centers
            placed_centers.append((lon_center, lat_center))

    # ---------------- #
    # Plot the results #
    # ---------------- #
    # fig, ax = plt.subplots(figsize=(10, 8))
    datafig = plt.figure(figsize=(12, 8))
    dataaxis = datafig.add_axes([0.15, 0.11, 0.73, 0.78])
    dataaxis.set_xlim([lon.min(), lon.max()])
    dataaxis.set_ylim([lat.min(), lat.max()])

    # Add parchment texture background if available
    try:
        img = plt.imread(parchment_file)
        dataaxis.imshow(img, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                        aspect='auto', alpha=1.0, zorder=0)
    except FileNotFoundError as error:
        print("Parchment texture file not found. Proceeding without it.")
        # traceback.format_exc()
        logger.exception(error)
    
    # variables for colorbar control
    cs_u = None 
    background_shown = False

    # TODO: pretty sure that if you wanna work with multiple semi-transparent
    # layers, that you then need to control the zorder
    
    if attribute5 is not None and use_attribute5_for_color:
        attribute5_absmax = np.maximum(np.absolute(attribute5_min), np.absolute(attribute5_max))
        attribute5_absnorm = np.abs(attribute5) / attribute5_absmax
        # attribute5_clipabs = np.maximum(np.minimum((attribute3_absnorm - 0.1) / 0.2, 1.0), .0)
        # Use attribute5 for background coloration
        cs_attr5 = dataaxis.pcolormesh(
            lon,
            lat,
            attribute5,
            shading='gouraud',
            cmap=attribute5_cmap,
            norm=attribute5_norm,
            alpha=attribute5_absnorm,  # means we still have a bit of parchment showing through
            zorder=1,
            # edgecolor=None,
            # rasterize=True
        )
        background_shown = True
    
    all_segments = [
        seg for seg in all_segments
        if np.isfinite(seg[0]).all() and np.isfinite(seg[1]).all()
    ]
    clean_segments = []
    clean_linewidths = []
    clean_alphas = []
    for i in range(len(all_segments)):
        seg = all_segments[i]
        if np.isfinite(seg[0]).all() and np.isfinite(seg[1]).all():
            clean_segments.append(all_segments[i])
            clean_linewidths.append(all_linewidths[i])
            clean_alphas.append(all_alphas[i])
    all_segments = clean_segments
    all_linewidths = clean_linewidths
    all_alphas = clean_alphas

    # Create line collection for the hatches
    if len(all_segments) > 0:
        lc = LineCollection(all_segments, linewidths=all_linewidths, colors=all_colors, alpha=all_alphas, zorder=2)
        # lc = LineCollection(all_segments, linewidths=all_linewidths, colors=all_colors, zorder=2)
        dataaxis.add_collection(lc)
    
    # Add tuft circles to indicate flow direction if enabled
    if toggle_tufts and tuft_centers:
        for center, size, alpha in zip(tuft_centers, tuft_sizes, tuft_alphas):
            circle = Circle(center, size/2, color=tuft_color, alpha=alpha, zorder=3)
            dataaxis.add_patch(circle)
    
    # Always add shoreline contour if signed_dist is provided
    if signed_dist is not None:
        shoreline = dataaxis.contour(lon, lat, signed_dist, levels=[0], colors="k", linewidths=1, linestyles='solid', zorder=4)
    
    # Add grid lines for scale reference
    grid_alpha = 0.5 
    grid_color = 'gray'
    grid_linewidth = 0.5
    grid_linestyle = '--'
    
    # Calculate appropriate grid spacing based on data bounds
    lon_range = lon.max() - lon.min()
    lat_range = lat.max() - lat.min()
    
    # Define grid spacing 
    lon_spacing = lon_range / 8 
    lat_spacing = lat_range / 8  
    
    # Round to nice values for the spacing
    lon_spacing = round_to_nice(lon_spacing)
    lat_spacing = round_to_nice(lat_spacing)
    
    # Calculate grid line positions
    lon_start = np.floor(lon.min() / lon_spacing) * lon_spacing
    lon_end = np.ceil(lon.max() / lon_spacing) * lon_spacing
    lat_start = np.floor(lat.min() / lat_spacing) * lat_spacing
    lat_end = np.ceil(lat.max() / lat_spacing) * lat_spacing
    
    lon_grid = np.arange(lon_start, lon_end + lon_spacing/2, lon_spacing)
    lat_grid = np.arange(lat_start, lat_end + lat_spacing/2, lat_spacing)
    
    # Draw grid lines
    dataaxis.grid(False)  # Disable default grid
    
    # Add custom grid lines
    for x in lon_grid:
        dataaxis.axvline(x=x, color=grid_color, linestyle=grid_linestyle,
                  linewidth=grid_linewidth, alpha=grid_alpha, zorder=0)
    
    for y in lat_grid:
        dataaxis.axhline(y=y, color=grid_color, linestyle=grid_linestyle,
                  linewidth=grid_linewidth, alpha=grid_alpha, zorder=0)
    
    # Add ticks at grid line positions
    dataaxis.set_xticks(lon_grid)
    dataaxis.set_yticks(lat_grid)
    
    # Format tick labels to reduce clutter
    dataaxis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    dataaxis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
     
    # Add compass rose if signed_dist is provided
    if signed_dist is not None:
        compass_position = add_compass_rose(
            dataaxis, lon, lat, signed_dist,
            size_factor=0.06,  # Adjust size relative to plot
            style='traditional',
            color='black',
            border_color='white',
            alpha=1.0
        )
        print(f"Added compass rose at position {compass_position}")

    # Set title and labels
    dataaxis.set_title(plot_title)  # => figure
    dataaxis.set_xlabel('lon [°]')
    dataaxis.set_ylabel('lat [°]')
    # dataaxis.set_xlim(lon.min(), lon.max())
    # dataaxis.set_ylim(lat.min(), lat.max())

    # Legend / colourbar for density
    ax_cbar_tuft_density = datafig.add_axes([0.06, 0.1, 0.02, 0.8])
    img = plt.imread("densitybar_vertical.png")
    ax_cbar_tuft_density.imshow(img, aspect='auto')
    ax_cbar_tuft_density.set_xlim(0, 1)
    ax_cbar_tuft_density.set_ylim(0, 1)
    # ax_cbar_tuft_density.tight_layout()
    ax_cbar_tuft_density.xaxis.set_major_locator(ticker.NullLocator())
    ax_cbar_tuft_density.yaxis.set_major_locator(ticker.MaxNLocator(5))
    if hasattr(attribute5, 'name'):
        ax_cbar_tuft_density.set_label(f'{attribute1.name}')
    else:
        density_cbar_label = ""
        # colorbar units
        # if background_label == "Depth":
        if attribute1_label == "Topography":
            density_cbar_label = "Elevation (m)"
        elif attribute1_label == "Bathymetry":
            density_cbar_label += "Depth (m)"
        elif attribute1_label == "Temperature":
            density_cbar_label = "Temperature (°C)"
        elif attribute1_label == "VelocityMagnitude":
            density_cbar_label = "Velocity Magnitude (m/s)"
        elif attribute5_label == "Divergence":
            density_cbar_label = "Divergence (1/s)"
        ax_cbar_tuft_density.set_label(density_cbar_label)
        # ax_cbar_tuft_density.xaxis.set_ticks_position('top')
        # ax_cbar_tuft_density.xaxis.set_label_position('bottom')
        ax_cbar_tuft_density.yaxis.set_ticks_position('right')
        ax_cbar_tuft_density.yaxis.set_label_position('left')

    # Legend / colourbar for the background colourmap
    ax_cbar_bg_colourmap = datafig.add_axes([0.92, 0.1, 0.02, 0.8])
    ax_cbar_bg_colourmap.set_facecolor("white")
    ax_cbar_bg_colourmap.yaxis.set_ticks_position('right')
    ax_cbar_bg_colourmap.yaxis.set_label_position('left')
    if use_attribute5_for_color and attribute5 is not None:
        cmap_bar = None
        # if attribute5_norm is not None:
        if not background_shown:
            # Create a ScalarMappable for the colorbar
            sm = cm.ScalarMappable(cmap=attribute5_cmap, norm=attribute5_norm)
            sm.set_array([])

            # Add colorbar for attribute5
            # cmap_bar = plt.colorbar(sm, ax=ax)
            cmap_bar = plt.colorbar(sm, cax=ax_cbar_bg_colourmap, ticks=ticker.AutoLocator(), orientation='vertical')
        # elif background_shown and cs_attr5 is not None:
        else:
            # Use the colormesh for the colorbar if we have one
            # cbar = fig.colorbar(cs_u, ax=ax)
            cmap_bar = plt.colorbar(cs_attr5, cax=ax_cbar_bg_colourmap, ticks=ticker.AutoLocator(), orientation='vertical')
        if hasattr(attribute5, 'name'):
            cmap_bar.set_label(f'{attribute5.name}')
        else:
            cbar_label = ""
            if attribute5_label == "Topography":
                cbar_label = "Elevation (m)"
            elif attribute5_label == "Bathymetry":
                cbar_label += "Depth (m)"
            elif attribute5_label == "Temperature":
                cbar_label = "Temperature (°C)"
            elif attribute5_label == "VelocityMagnitude":
                cbar_label = "Velocity Magnitude (m/s)"
            elif attribute5_label == "Divergence":
                cbar_label = "Divergence (1/s)"
            cmap_bar.set_label(cbar_label)
            # ax_cbar_bg_colourmap.xaxis.set_ticks_position('top')
            # ax_cbar_bg_colourmap.xaxis.set_label_position('bottom')
            ax_cbar_bg_colourmap.yaxis.set_ticks_position('left')
            ax_cbar_bg_colourmap.yaxis.set_label_position('right')

    legend_elements = []
    # Legend / colourbar for line colour
    if not isinstance(all_colors, str):
        # for filling this, we need the base colour map for it
        pass

    # Legend / colourbar for alpha difficult to do - leave for now

    # Legend / colourbar for hatch length
    if attribute4 is not None:
        unit_base = ""
        if attribute4_label == "Topography":
            unit_base = "m"
        elif attribute4_label == "Bathymetry":
            unit_base = "m"
        elif attribute4_label == "Temperature":
            unit_base = "°C"
        elif attribute4_label == "VelocityMagnitude":
            unit_base = "m/s"
        elif attribute4_label == "Divergence":
            unit_base = "1/s"
        for i in range(len(legend_hatchlengths)):
            lengthitem = legend_hatchlengths[i]
            lengthlabel = legend_hatchlengths_labels[i]
            lengthitemlabel = "{:.2f} {}".format(lengthlabel, unit_base)
            legend_elements.append(Line2D([0], [0], color='black', linestyle=lengthitem, lw=1, label=lengthitemlabel))

    # Legend / colourbar for hatch width
    # if not isinstance(all_linewidths, str):
    if attribute6 is not None:
        unit_base = ""
        if attribute6_label == "Topography":
            # label_base = "Elevation (m)"
            unit_base = "m"
        elif attribute6_label == "Bathymetry":
            # label_base = "Depth (m)"
            unit_base = "m"
        elif attribute6_label == "Temperature":
            # label_base = "Temperature (°C)"
            unit_base = "°C"
        elif attribute6_label == "VelocityMagnitude":
            # label_base = "Velocity Magnitude ()"
            unit_base = "m/s"
        elif attribute6_label == "Divergence":
            # label_base = "Divergence (1/s)"
            unit_base = "1/s"
        for i in range(len(legend_hatchwidths)):  # TODO: pregenerate that field of contour width [DONE]
            widthitem = legend_hatchwidths[i]
            widthlabel = legend_hatchwidths_labels[i]
            widthitemlabel = "{:.2f} {}".format(widthlabel, unit_base)
            legend_elements.append(Line2D([0], [0], color='black', lw=widthitem, label=widthitemlabel))

    # Finalize the in-plot legend
    if len(legend_elements) > 0:
        dataaxis.legend(handles=legend_elements, loc='upper right')
    # Add scale bar
    add_scale_bar(dataaxis, lon_range, lat_range)
    # plt.tight_layout()
    
    # Save the figure
    plt.savefig(output_image, dpi=plot_dpi, bbox_inches='tight')
    print(f"Figure saved to {output_image}")
    
    # Optionally show the figure
    # plt.show()
    plt.close(datafig)
    
    # Return generated data for potential further use
    # return {
    #     'segments': all_segments,
    #     'linewidths': all_linewidths,
    #     'alphas': all_alphas,
    #     'colors': all_colors,
    #     'tuft_centers': tuft_centers,
    #     'tuft_sizes': tuft_sizes,
    #     'tuft_alphas': tuft_alphas,
    #     'parameters': {
    #         'max_dash_length': max_dash_length,
    #         'curve_intensity': curve_intensity,
    #         'jitter_amount': jitter_amount,
    #         'min_distance': min_distance,
    #         'tuft_size_factor': tuft_size_factor,
    #         'tuft_min_distance': tuft_min_distance
    #     }
    # }
    return True

#--------------------------------------------------------------------------
# Stipple Generation
#--------------------------------------------------------------------------

def stipple(lon, lat, attribute1, attribute2=None, attribute3=None, signed_dist=None, output_image="stipple_output.png", 
           attribute1_norm=None, attribute1_min=None, attribute1_max=None,
           attribute2_norm=None, attribute2_min=None, attribute2_max=None,
           attribute3_norm=None, attribute3_min=None, attribute3_max=None,
           attribute1_label="Contour Lines", attribute2_label="Point Density", attribute3_label="Background Color",
           plot_title="Stipple Visualisation", attribute3_cmap="viridis", 
           num_contours=6, marker_size=12, marker_color="k", marker_shape="o",
           min_gap=None, max_gap=None, flow_scaling=0.7, auto_scale=False):
    """
    Generate stipple visualisation based on attribute values and save the figure.
    
    Parameters:
    -----------
    lon : xarray.DataArray
        Longitude coordinates
    lat : xarray.DataArray
        Latitude coordinates
    attribute1 : xarray.DataArray
        First attribute (maps to contour lines)
    attribute2 : xarray.DataArray, optional
        Second attribute (maps to stipple density/spacing)
    attribute3 : xarray.DataArray, optional
        Third attribute (maps to background coloration)
    signed_dist : numpy.ndarray, optional
        Signed distance field from shoreline
    output_image : str, optional
        Path to save the output image, default is "stipple_output.png"
    attribute1_norm : matplotlib.colors.Normalize, optional
        Normalization for attribute1 values
    attribute1_min : float, optional
        Minimum value for attribute1 (for contour levels)
    attribute1_max : float, optional
        Maximum value for attribute1 (for contour levels)
    attribute2_norm : matplotlib.colors.Normalize, optional
        Normalization for attribute2 values
    attribute2_min : float, optional
        Minimum value for attribute2
    attribute2_max : float, optional
        Maximum value for attribute2
    attribute3_norm : matplotlib.colors.Normalize, optional
        Normalization for attribute3 values for color mapping
    attribute3_min : float, optional
        Minimum value for attribute3
    attribute3_max : float, optional
        Maximum value for attribute3
    attribute1_label : str, optional
        Label for attribute1, default is "Contour Lines"
    attribute2_label : str, optional
        Label for attribute2 (density), default is "Point Density"
    attribute3_label : str, optional
        Label for attribute3 on the colorbar, default is "Background Color"
    plot_title : str, optional
        Title for the plot, default is "Stipple Visualisation"
    attribute3_cmap : str or matplotlib.colors.Colormap, optional
        Colormap for attribute3, default is "viridis"
    num_contours : int, optional
        Number of contour levels to use, default is 8
    marker_size : int, optional
        Size of the stipple markers, default is 12
    marker_color : str, optional
        Color of the stipple markers, default is "k" (black)
    marker_shape : str, optional
        Shape of the stipple markers, default is "o" (circle)
    min_gap : float, optional
        Minimum gap between markers at maximum attribute2 value, default is 0.01
    max_gap : float, optional
        Maximum gap between markers at minimum attribute2 value, default is 0.05
    flow_scaling : float, optional
        Exponent for non-linear scaling of attribute2 to marker spacing, default is 0.7
    auto_scale: boolean, optiona,
        Rescales colourmap to percentiles (True) instead of min-max (False)
    """

    # Calculate geographic extents for auto-scaling
    lon_range = lon.max().item() - lon.min().item()
    lat_range = lat.max().item() - lat.min().item()
    avg_range = (lon_range + lat_range) / 2
    
    # Exponent for scaling the gap size
    scaling_exponent = 1.3 
    if min_gap is None:
        min_gap = 0.01 * (avg_range ** scaling_exponent)
    if max_gap is None:
        max_gap = 0.05 * (avg_range ** scaling_exponent)

    # Ensure attribute1_min and attribute1_max are set if not provided (for contour lines)
    if attribute1_min is None:
        attribute1_min = np.nanmin(attribute1)
    if attribute1_max is None:
        attribute1_max = np.nanmax(attribute1)
    
    # Create normalisation for attribute1 if not provided
    if attribute1_norm is None:
        attribute1_norm = plt.Normalize(vmin=attribute1_min, vmax=attribute1_max)
    
    # For attribute2 (density) use attribute1 if not provided; abs to avoid negative values
    density_attribute = attribute2 if attribute2 is not None else attribute1
    if attribute2 is not None:
        attribute2 = np.abs(attribute2) 
        # eps = np.info(np.float32)  # 1e-6
        # attribute2 = np.where(attribute2 == 0, eps, attribute2)  # float == 0 ? really ?
    
    # Ensure attribute2_min and attribute2_max are set if not provided (for density)
    if attribute2 is not None:
        if attribute2_min is None:
            attribute2_min = np.nanmin(attribute2)
        if attribute2_max is None:
            attribute2_max = np.nanmax(attribute2)
        if attribute2_norm is None:
            attribute2_norm = plt.Normalize(vmin=attribute2_min, vmax=attribute2_max)
    else:
        attribute2_min = attribute1_min
        attribute2_max = attribute1_max
        attribute2_norm = attribute1_norm

    # Set up normalisation for background
    if attribute3 is not None:
        # Use attribute3 for background coloration
        if attribute3_min is None:
            attribute3_min = np.nanmin(attribute3)
        if attribute3_max is None:
            attribute3_max = np.nanmax(attribute3)
        if attribute3_norm is None:
            if auto_scale:
                # Use robust percentile-based normalisation
                valid_values = attribute3[~np.isnan(attribute3)]
                if len(valid_values) > 0:
                    robust_min = np.percentile(valid_values, 2)
                    robust_max = np.percentile(valid_values, 98)
                    attribute3_norm = Normalize(vmin=robust_min, vmax=robust_max)
                else:
                    attribute3_norm = Normalize(vmin=attribute3_min, vmax=attribute3_max)
            else:
                attribute3_norm = Normalize(vmin=attribute3_min, vmax=attribute3_max)
    #     background_norm = attribute3_norm
    # elif attribute2 is not None:
    #     # Use attribute2 for background if attribute3 is not provided
    #     background_norm = attribute2_norm
    # else:
    #     # Use attribute1 for background if neither attribute2 nor attribute3 is provided
    #     background_norm = attribute1_norm

    # Determine which attribute to use for background coloration
    # background_data = attribute3 if attribute3 is not None else (attribute2 if attribute2 is not None else attribute1)
    # background_cmap = attribute3_cmap
    # background_label = attribute3_label if attribute3 is not None else (attribute2_label if attribute2 is not None else attribute1_label)

    legend_stippledensity = []
    legend_stippledensity_labels = []
    if attribute2 is not None:
        mnv = np.nanmin(attribute2)
        mxv = np.nanmax(attribute2)
        legend_gapmin = 1
        legend_gapmax = 10
        for svalue in [.0, 0.5, 1.0]:
            attrib_value = mnv + svalue * (mxv - mnv)
            # ratio = svalue ** flow_scaling
            gap = legend_gapmax - (svalue * (legend_gapmax - legend_gapmin))
            gap = np.clip(gap, legend_gapmin, legend_gapmax)
            # print("Gap for svalue = {}: {}".format(svalue, gap))
            legend_stippledensity.append((0, (1, gap)))
            legend_stippledensity_labels.append(attrib_value)

    # ---------------- #
    # Plot the results #
    # ---------------- #
    # fig, ax = plt.subplots(figsize=(12, 10))
    datafig = plt.figure(figsize=(12, 9))
    dataaxis = datafig.add_axes([0.15, 0.11, 0.73, 0.78])
    dataaxis.set_xlim([lon.min(), lon.max()])
    dataaxis.set_ylim([lat.min(), lat.max()])
    
    # Add parchment texture background if available
    try:
        # img = plt.imread('parchment_texture.jpg')
        img = plt.imread(parchment_file)
        dataaxis.imshow(img, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                aspect='auto', alpha=1.0, zorder=0)
    except FileNotFoundError as error:
        print("Parchment texture file not found. Proceeding without it.")
        # traceback.format_exc()
        logger.exception(error)

    if attribute3 is not None:
        # do for pcolormesh alpha
        attribute3_absmax = np.maximum(np.absolute(attribute3_min), np.absolute(attribute3_max))
        attribute3_absnorm = np.abs(attribute3) / attribute3_absmax
        # attribute3_clipabs = np.maximum(np.minimum((attribute3_absnorm - 0.1) / 0.2, 1.0), .0)
        # Plot background coloration
        cs_attr3 = dataaxis.pcolormesh(lon, lat, attribute3,
                                       cmap=attribute3_cmap,
                                       norm=attribute3_norm,
                                       shading= "gouraud",
                                       alpha=attribute3_absnorm,
                                       zorder=1)
        # plt.colorbar(im, ax=ax, label=background_label)
    
    # Draw invisible contour lines where we will sample points
    def fmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        # return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"
        return f"{s}"

    contour_levels = np.linspace(attribute1_min, attribute1_max, num_contours)
    CS = dataaxis.contour(lon, lat, attribute1,
                    levels=contour_levels,
                    colors="none", linewidths=0.8, zorder=2)
    
    # Create an interpolator for density attribute
    density_interpolator = RegularGridInterpolator(
        (lat, lon), density_attribute, bounds_error=False, fill_value=np.nan
    )
    
    # Collect all marker positions (lon, lat)
    marker_points = []
    
    # Go through each contour level
    for level_idx, level_val in enumerate(CS.levels):
        # Each level can have multiple segments 
        for seg in CS.allsegs[level_idx]:
            if len(seg) < 2:
                continue
            
            # seg  shape  [ [lon0, lat0], [lon1, lat1], ... ]
            # For density_interpolator, we need (lat, lon)
            pts_for_interp = seg[:, ::-1]  # swap to (lat, lon)
            density_vals = density_interpolator(pts_for_interp)
            
            # Skip if everything is NaN
            valid = ~np.isnan(density_vals)
            if not valid.any():
                continue
                
            # Average density along this  segment
            local_density = np.nanmean(density_vals)
            
            # Compute the gap based on density value
            ratio = (local_density / attribute2_max) ** flow_scaling
            gap = max_gap - (ratio * (max_gap - min_gap))
            gap = np.clip(gap, min_gap, max_gap)
            
            # Parametrise the segment by its cumulative distance
            x = seg[:, 0]  # lon
            y = seg[:, 1]  # lat
            
            # Distances between consecutive contour points
            step_dist = np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2)
            distances = np.cumsum(step_dist)
            total_length = distances[-1]
            
            # Interpolator to get (lon, lat) for any distance s along the segment
            if total_length <= 0:
                continue  
                
            f = interp1d(distances,
                         np.column_stack([x, y]),
                         axis=0,
                         kind='linear')
            
            # Step along the segment 
            dist_pos = 0.0
            while dist_pos <= total_length:
                # Evaluate (lon, lat) 
                xy = f(dist_pos)  
                lon_pt, lat_pt = xy
                
                # Check if density value is NaN at this point (land or out-of-bounds)
                test_density = density_interpolator((lat_pt, lon_pt))
                if not np.isnan(test_density):
                    marker_points.append(xy)
                    
                dist_pos += gap
    
    # Convert to numpy array (N, 2)
    marker_points = np.array(marker_points)
    
    if marker_points.size > 0:
        # marker_points[:, 0] => lon
        # marker_points[:, 1] => lat
        dataaxis.scatter(
            marker_points[:, 0],
            marker_points[:, 1],
            s=marker_size,
            color=marker_color,
            marker=marker_shape,
            zorder=3,
        )
    
    # Add shoreline (where elevation=0)
    if signed_dist is not None:
        shoreline = dataaxis.contour(lon, lat, signed_dist, levels=[0], colors="k", linewidths=1, linestyles='solid', zorder=4)
    
    # Add grid lines for scale reference
    grid_alpha = 0.8  # Transparency of grid lines
    grid_color = 'gray'
    grid_linewidth = 0.5
    grid_linestyle = '--'
    
    # Calculate appropriate grid spacing based on data bounds
    lon_range = lon.max() - lon.min()
    lat_range = lat.max() - lat.min()
    
    # Define grid spacing 
    lon_spacing = lon_range / 8  
    lat_spacing = lat_range / 8  
    
    # Round to nice values for the spacing
    lon_spacing = round_to_nice(lon_spacing)
    lat_spacing = round_to_nice(lat_spacing)
    
    # Calculate grid line positions
    lon_start = np.floor(lon.min() / lon_spacing) * lon_spacing
    lon_end = np.ceil(lon.max() / lon_spacing) * lon_spacing
    lat_start = np.floor(lat.min() / lat_spacing) * lat_spacing
    lat_end = np.ceil(lat.max() / lat_spacing) * lat_spacing
    
    lon_grid = np.arange(lon_start, lon_end + lon_spacing/2, lon_spacing)
    lat_grid = np.arange(lat_start, lat_end + lat_spacing/2, lat_spacing)
    
    # Draw grid lines
    dataaxis.grid(False)  # Disable default grid
    
    # Add custom grid lines
    for x in lon_grid:
        dataaxis.axvline(x=x, color=grid_color, linestyle=grid_linestyle,
                  linewidth=grid_linewidth, alpha=grid_alpha, zorder=0)
    
    for y in lat_grid:
        dataaxis.axhline(y=y, color=grid_color, linestyle=grid_linestyle,
                  linewidth=grid_linewidth, alpha=grid_alpha, zorder=0)
    
    # Add ticks at grid line positions
    dataaxis.set_xticks(lon_grid)
    dataaxis.set_yticks(lat_grid)
    
    # Format tick labels to reduce clutter
    dataaxis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    dataaxis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    
    # Add compass rose if signed_dist is provided
    if signed_dist is not None:
        compass_position = add_compass_rose(
            dataaxis, lon, lat, signed_dist,
            size_factor=0.06,  # Adjust size relative to plot
            style='traditional',
            color='black',
            border_color='white',
            alpha=1.0
        )
        print(f"Added compass rose at position {compass_position}")

    # Set title and labels
    dataaxis.set_title(plot_title)
    dataaxis.set_xlabel('lon [°]')
    dataaxis.set_ylabel('lat [°]')
    # dataaxis.set_xlim(lon.min(), lon.max())
    # dataaxis.set_ylim(lat.min(), lat.max())

    # TODO: missing color bars / legend
    # ------------------------------------------------------------------------------ #
    # LEGEND & COLOUR BARS
    # ------------------------------------------------------------------------------ #

    # Legend / colourbar for the contour density itself (see previous contour plots)
    ax_cbar_contourdensity = datafig.add_axes([0.06, 0.1, 0.02, 0.8])
    ax_cbar_contourdensity.set_facecolor("white")
    if attribute1 is not None:
        sm = cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list("Blacks", [(1,1,1), (0,0,0)]), norm=attribute1_norm)
        sm.set_array(contour_levels)
        cbar_attr1 = datafig.colorbar(sm, cax=ax_cbar_contourdensity, ticks=contour_levels, orientation='vertical')  # , extend='both', format='%.0e'), ticker.MaxNLocator(num_contours)
        cbar_label = "Stippling edge levels: "
        if attribute1_label == "Topography":
            cbar_label += "Elevation (m)"
        elif attribute1_label == "Bathymetry":
            cbar_label += "Depth (m)"
        elif attribute1_label == "Temperature":
            cbar_label += "Temperature (°C)"
        elif attribute1_label == "VelocityMagnitude":
            cbar_label += "Velocity Magnitude (m/s)"
        elif attribute1_label == "Divergence":
            cbar_label += "Divergence (1/s)"
        cbar_attr1.set_label(cbar_label)
        # ax_cbar_contourdensity.xaxis.set_ticks_position('bottom')
        # ax_cbar_contourdensity.xaxis.set_label_position('top')
        ax_cbar_contourdensity.yaxis.set_ticks_position('right')
        ax_cbar_contourdensity.yaxis.set_label_position('left')

    # Legend / colourbar for the background colourmap
    ax_cbar_bg_colourmap = datafig.add_axes([0.92, 0.1, 0.02, 0.8])
    ax_cbar_bg_colourmap.set_facecolor("white")
    # ax_cbar_bg_colourmap.yaxis.set_ticks_position('right')
    # ax_cbar_bg_colourmap.yaxis.set_label_position('left')
    if attribute3 is not None:
        cbar_attr3 = datafig.colorbar(cs_attr3, cax=ax_cbar_bg_colourmap, ticks=ticker.AutoLocator(), orientation='vertical')
        cbar_label = ""
        if attribute3_label == "Topography":
            cbar_label = "Elevation (m)"
        elif attribute3_label == "Bathymetry":
            cbar_label += "Depth (m)"
        elif attribute3_label == "Temperature":
            cbar_label = "Temperature (°C)"
        elif attribute3_label == "VelocityMagnitude":
            cbar_label = "Velocity Magnitude (m/s)"
        elif attribute3_label == "Divergence":
            cbar_label = "Divergence (1/s)"
        cbar_attr3.set_label(cbar_label)
        # ax_cbar_bg_colourmap.xaxis.set_ticks_position('top')
        # ax_cbar_bg_colourmap.xaxis.set_label_position('bottom')
        ax_cbar_bg_colourmap.yaxis.set_ticks_position('left')
        ax_cbar_bg_colourmap.yaxis.set_label_position('right')

    legend_elements = []
    # Legend / colourbar for contour dash pattern
    if attribute2 is not None:
        unit_base = ""
        if attribute2_label == "Topography":
            unit_base = "m"
        elif attribute2_label == "Bathymetry":
            unit_base = "m"
        elif attribute2_label == "Temperature":
            unit_base = "°C"
        elif attribute2_label == "VelocityMagnitude":
            unit_base = "m/s"
        elif attribute2_label == "Divergence":
            unit_base = "1/s"
        for i in range(len(legend_stippledensity)):
            itemlabel = "{:.2f} {}".format(legend_stippledensity_labels[i], unit_base)
            legend_elements.append(Line2D([0], [0], color='black', lw=1, linestyle=legend_stippledensity[i], label=itemlabel))

    # Finalize the in-plot legend
    if len(legend_elements) > 0:
        dataaxis.legend(handles=legend_elements, loc='upper right')
    # Add scale bar
    add_scale_bar(dataaxis, lon_range, lat_range)
    # plt.tight_layout()
    plt.savefig(output_image, dpi=plot_dpi, bbox_inches='tight')
    print(f"Figure saved to {output_image}")
    plt.close()
    # Return marker points for potential further use
    # return marker_points
    return True

#--------------------------------------------------------------------------
# Flow Contours
#--------------------------------------------------------------------------

def flow(attribute1, lon, lat, attribute2, attribute3=None, attribute4=None, signed_dist=None, output_image="flow_output.png", attribute1_label="Depth",
         attribute2_max=None, attribute2_min=None, attribute2_label="", attribute3_max=None, attribute3_min=None, attribute4_max=None, attribute4_min=None,
         attribute4_label="", attribute3_norm=None, attribute3_cmap='viridis', attribute3_label="Topography",
         plot_title="Flow as Dashed Contours", num_contours=6,
         min_linewidth=None, max_linewidth=None, scale_factor=None,
         MIN_DASH=None, MAX_DASH=None, chunk_size=40, DASH_JITTER=0.05, MAX_SUBSEG_LEN=0.02,
         use_texture=True, auto_scale=True):
    """
    Generate flow visualisation with dashed contours based on multiple attributes.
    
    Parameters:
    -----------
    attribute1 : xarray.DataArray
        Primary attribute for contour generation (e.g., flow magnitude)
    lon : xarray.DataArray
        Longitude coordinates
    lat : xarray.DataArray
        Latitude coordinates
    attribute2 : xarray.DataArray
        Secondary attribute for dash pattern (e.g., temperature)
    attribute3 : xarray.DataArray, optional
        Tertiary attribute for background coloration (e.g., elevation/depth)
    attribute4 : xarray.DataArray, optional
        Fourth attribute for line width (e.g., divergence)
        If None, a uniform width is used
    signed_dist : numpy.ndarray, optional
        Signed distance field from shoreline
    output_image : str, optional
        Path to save the output image, default is "flow_output.png"
    attribute1_label : str, optional
        Label for attribute1 on the colorbar, default is "Depth"
    attribute2_max : float, optional
        Maximum value for attribute2
    attribute2_min : float, optional
        Minimum value for attribute2
    attribute2_label : str, optional
        Label for attribute2
    attribute3_max : float, optional
        Maximum value for attribute3
    attribute3_min : float, optional
        Minimum value for attribute3
    attribute3_label : str, optional
        Label for attribute3 on the colorbar, default is "Topography"
    attribute4_max : float, optional
        Maximum value for attribute4
    attribute4_min : float, optional
        Minimum value for attribute4
    attribute4_label : str, optional
        Label for attribute4
    attribute3_norm : matplotlib.colors.Normalize, optional
        Normalization for attribute3 values for color mapping
    attribute3_cmap : str or matplotlib.colors.Colormap, optional
        Colormap for attribute3, default is "viridis"
    plot_title : str, optional
        Title for the plot, default is "Flow as Dashed Contours"
    num_contours : int, optional
        Number of contour levels to use, default is 8
    min_linewidth : float, optional
        Minimum line width for contours, default is auto-scaled if None
    max_linewidth : float, optional
        Maximum line width for contours, default is auto-scaled if None
    scale_factor : float, optional
        Scale factor for dash length, default is auto-scaled if None
    MIN_DASH : float, optional
        Minimum dash length, default is auto-scaled if None
    MAX_DASH : float, optional
        Maximum dash length, default is auto-scaled if None
    chunk_size : float, optional
        Size for chunking line segments, default is 40
    DASH_JITTER : float, optional
        Jitter amount for dash pattern (0-1), default is 0.05
    MAX_SUBSEG_LEN : float, optional
        Maximum subsegment length for breaking contours, default is 0.02
    use_texture : bool, optional
        Whether to use background texture, default is True
    auto_scale : bool, optional
        Whether to automatically scale parameters based on data, default is True
    """
    # Calculate data extent
    lon_range = np.abs(lon.max() - lon.min())
    lat_range = np.abs(lat.max() - lat.min())
    data_extent = np.sqrt(lon_range**2 + lat_range**2)

    # Auto-scale parameters based on data characteristics if requested
    if auto_scale:
        # Auto-scale dash lengths based on data extent
        if MIN_DASH is None:
            MIN_DASH = max(1.0, data_extent * 0.004) 
        
        if MAX_DASH is None:
            MAX_DASH = max(25.0, data_extent * 0.2) 
        
        # Auto-scale line widths based on data complexity and extent
        if min_linewidth is None:
            if attribute1.size > 100000: 
                min_linewidth = 0.3
            elif attribute1.size > 10000:  
                min_linewidth = 0.5
            else:  
                min_linewidth = 0.8
        
        if max_linewidth is None:
            if attribute1.size > 100000: 
                max_linewidth = 1.0
            elif attribute1.size > 10000:  
                max_linewidth = 1.5
            else:  
                max_linewidth = 2.0
        
        if scale_factor is None:
                scale_factor = 1.0
        
        # Adjust chunk_size based on data density
        density = attribute1.size / (lon_range * lat_range)
        if density > 1000:  
            chunk_size = min(chunk_size, 20) 
        elif density < 100:  
            chunk_size = max(chunk_size, 60)  
    else:
        if MIN_DASH is None:
            MIN_DASH = 2.0
        if MAX_DASH is None:
            MAX_DASH = 50.0
        if min_linewidth is None:
            min_linewidth = 0.5
        if max_linewidth is None:
            max_linewidth = 1.5
        if scale_factor is None:
            scale_factor = 1.0
    
    # Ensure MAX_DASH is valid relative to chunk_size
    MAX_DASH = min(chunk_size / 2, MAX_DASH)

    # Calculate attribute2 min/max for dash pattern
    if attribute2 is not None:
        if attribute2_min is None:
            attribute2_min = np.nanmin(attribute2)
        if attribute2_max is None:
            attribute2_max = np.nanmax(attribute2)
        # Create normalisation for attribute2 if not provided
        if auto_scale:
            # Get percentile-based min/max
            # valid_values = attribute2.values[~np.isnan(attribute2.values)]
            valid_values = attribute2[~np.isnan(attribute2)]
            robust_min = np.percentile(valid_values, 5)
            robust_max = np.percentile(valid_values, 95)
            attribute2_norm = Normalize(vmin=robust_min, vmax=robust_max)
            attribute2_min = robust_min
            attribute2_max = robust_max
        else:
            attribute2_norm = Normalize(vmin=attribute2_min, vmax=attribute2_max)

    if attribute3 is not None:
        # Ensure attribute3_min and attribute3_max are set if not provided
        if attribute3_min is None:
            attribute3_min = np.nanmin(attribute3)
        if attribute3_max is None:
            attribute3_max = np.nanmax(attribute3)
        # Create normalisation for attribute3 if not provided
        if attribute3_norm is None:
            if auto_scale:
                # Use robust percentile-based normalisation
                # valid_values = attribute3.values[~np.isnan(attribute3.values)]
                valid_values = attribute3[~np.isnan(attribute3)]
                if len(valid_values) > 0:
                    robust_min = np.percentile(valid_values, 2)
                    robust_max = np.percentile(valid_values, 98)
                    attribute3_norm = Normalize(vmin=robust_min, vmax=robust_max)
                    attribute3_min = robust_min
                    attribute3_max = robust_max
                else:
                    attribute3_norm = Normalize(vmin=attribute3_min, vmax=attribute3_max)
            else:
                attribute3_norm = Normalize(vmin=attribute3_min, vmax=attribute3_max)

    # Calculate attribute4 min/max if provided (for line width)
    if attribute4 is not None:
        if attribute4_min is None:
            attribute4_min = np.nanmin(attribute4)
        if attribute4_max is None:
            attribute4_max = np.nanmax(attribute4)
        if auto_scale:
            # Get percentile-based min/max
            # valid_values = attribute4.values[~np.isnan(attribute4.values)]
            valid_values = attribute4[~np.isnan(attribute4)]
            robust_min = np.percentile(valid_values, 5)
            robust_max = np.percentile(valid_values, 95)
            attribute4_norm = Normalize(vmin=robust_min, vmax=robust_max)
            attribute4_min = robust_min
            attribute4_max = robust_max
        else:
            attribute4_norm = Normalize(vmin=attribute4_min, vmax=attribute4_max)

    # prepare legend arrays
    legend_contourwidths = []
    legend_contourwidths_labels = []
    if attribute4 is not None:
        for wvalue in [.0, 0.5, 1.0]:
            width_value = min_linewidth + wvalue * (max_linewidth - min_linewidth)
            width_attr_value = wvalue * (attribute4_max - attribute4_min) + attribute4_min
            legend_contourwidths.append(width_value)
            legend_contourwidths_labels.append(width_attr_value)
    legend_contourstyles = []
    legend_contourstyles_labels = []
    if attribute2 is not None:
        for svalue in [.0, 0.5, 1.0]:
            mnv = np.min(attribute2)
            mxv = np.max(attribute2)
            attrib_value = mnv + svalue * (mxv - mnv)
            dash_range = MAX_DASH - MIN_DASH
            base_dash = MAX_DASH - (svalue * dash_range * 0.8)
            base_gap = base_dash * 0.6  # Gap shorter than dash
            base_dash *= scale_factor
            base_gap *= scale_factor
            legend_contourstyles.append((0, (base_dash, base_gap)))
            legend_contourstyles_labels.append(attrib_value)


    # ----------------------------------------- #
    # ------ Prepare data for plotting -------- #
    # ----------------------------------------- #
    # Define contour levels based on attribute1
    # valid_data = attribute1.values[~np.isnan(attribute1.values)]
    valid_data = attribute1[~np.isnan(attribute1)]

    if len(valid_data) == 0:
        print("Warning: No valid data found in attribute1. Skipping visualisation.")
        return None

    min_attr1 = np.min(valid_data)
    max_attr1 = np.max(valid_data)

    # Check if we have sufficient variation in the data
    data_range = max_attr1 - min_attr1

    # Make sure we have usable contour levels
    if auto_scale and len(valid_data) > 100:
        # Use percentiles for more robust contour levels
        percentiles = np.linspace(10, 90, num_contours)  # Use 10-90% range to avoid extreme outliers
        contour_levels = np.percentile(valid_data, percentiles)
    else:
        contour_levels = np.linspace(min_attr1, max_attr1, num_contours)

    # remove duplicates
    contour_levels = np.unique(contour_levels)
    contour_levels = contour_levels[contour_levels > min_attr1]

    if len(contour_levels) < 2:
        # Fallback linear levels
        contour_levels = np.linspace(min_attr1 + data_range * 0.1,
                                    max_attr1 - data_range * 0.1,
                                    max(2, num_contours // 2))

    # check that levels are strictly increasing
    contour_levels = np.sort(contour_levels)
    if len(contour_levels) < 2 or not np.all(np.diff(contour_levels) > 0):
        print(f"Warning: Could not generate valid contour levels for attribute1. Data range: {min_attr1:.6f} to {max_attr1:.6f}")
        return None

    print(f"Generated {len(contour_levels)} contour levels ranging from {contour_levels[0]:.6f} to {contour_levels[-1]:.6f}")

    # ---------------- #
    # Plot the results #
    # ---------------- #
    # fig, ax = plt.subplots(figsize=(10, 8))
    datafig = plt.figure(figsize=(12, 9))
    dataaxis = datafig.add_axes([0.15, 0.11, 0.73, 0.78])
    dataaxis.set_xlim([lon.min(), lon.max()])
    dataaxis.set_ylim([lat.min(), lat.max()])

    # Add texture background
    if use_texture:
        try:
            img = plt.imread(parchment_file)
            dataaxis.imshow(img, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                    aspect='auto', alpha=1.0, zorder=0)
        except FileNotFoundError as error:
            print(f"Parchment texture file {parchment_file} not found. Proceeding without it.")
            # traceback.format_exc()
            logger.exception(error)

    # Generate contours based on attribute1
    def fmt(x):
        s = f"{x:.1f}"
        if s.endswith("0"):
            s = f"{x:.0f}"
        # return rf"{s} \%" if plt.rcParams["text.usetex"] else f"{s} %"
        return f"{s}"

    try:
        CS = dataaxis.contour(lon, lat, attribute1, levels=contour_levels, colors="none", linewidths=0.8, zorder=1)  # , colors="none"
        dataaxis.clabel(CS, CS.levels, fmt=fmt, fontsize=9, colors="k")
    except ValueError as e:
        print(f"Error generating contours: {e}")
        print(f"Contour levels: {contour_levels}")
        print(f"Data shape: {attribute1.shape}, range: [{min_attr1:.6f}, {max_attr1:.6f}]")
        # traceback.format_exc()
        logger.exception(e)
        return None
    
    # 2D interpolator for attribute2  dash pattern
    attribute2_interpolator = RegularGridInterpolator(
        (lat, lon),           
        attribute2,  # attribute2.values
        method='linear',      
        bounds_error=False,    
        fill_value=np.nan      
    )
    
    # 2D interpolator for attribute4  (line width)
    if attribute4 is not None:
        attribute4_interpolator = RegularGridInterpolator(
            (lat, lon),           
            attribute4,  # attribute4.values
            method='linear',      
            bounds_error=False,    
            fill_value=np.nan      
        )

    def attr2_to_dash_len(attr2_val, attr2_min, attr2_max, scale_factor):
        """
        Map attribute2 value to dash length using proper min/max normalisation.
        
        Parameters:
        -----------
        attr2_val : float
            The attribute2 value to map
        attr2_min : float  
            Minimum value of attribute2 range
        attr2_max : float
            Maximum value of attribute2 range
        scale_factor : float
            Scale factor for dash lengths
            
        Returns:
        --------
        tuple : (dash_length, gap_length)
        
        """
        if np.isnan(attr2_val) or np.isnan(attr2_min) or np.isnan(attr2_max):
            return (MIN_DASH, MIN_DASH)
        
        # Handle edge case where min == max
        if attr2_max == attr2_min:
            return ((MIN_DASH + MAX_DASH) / 2, (MIN_DASH + MAX_DASH) / 2)
        
        # Normalize to [0,1] range 
        normalized = (attr2_val - attr2_min) / (attr2_max - attr2_min)
        
        # Clamp to [0,1] 
        normalized = max(0, min(1, normalized))
        
        # Map to dash length: 
        # normalized=0 (attr2_min) → longer dashes (MAX_DASH end)
        # normalized=1 (attr2_max) → shorter dashes (MIN_DASH end)
        #  0.8 factor to avoid extreme values and maintain visual variety
        dash_range = MAX_DASH - MIN_DASH
        base_dash = MAX_DASH - (normalized * dash_range * 0.8)
        base_gap = base_dash * 0.6  # Gap shorter than dash 
        
        # Apply scale factor
        base_dash *= scale_factor
        base_gap *= scale_factor
        
        # Add jitter to prevent too-regular patterns (preserve minimums)
        dash_jit = base_dash * (1 + DASH_JITTER * (np.random.rand() - 0.5))
        gap_jit = base_gap * (1 + DASH_JITTER * (np.random.rand() - 0.5))
        
        return (
            max(MIN_DASH, min(MAX_DASH, dash_jit)), 
            max(MIN_DASH, min(MAX_DASH, gap_jit))
        )

    # Split a line (array of [y, x]) into sub-segments with each sub-segment up to ~<= max_length in total distance.
    def chunk_by_distance(points, max_length):
        if len(points) < 2:
            return [points]  # Return original if too short
            
        sub_segments = []
        start_idx = 0
        accumulated_dist = 0.0

        for i in range(1, len(points)):
            dx = points[i, 1] - points[i-1, 1]
            dy = points[i, 0] - points[i-1, 0]
            dist = np.hypot(dx, dy)

            if accumulated_dist + dist > max_length:
                # Include current point in current segment
                sub_segments.append(points[start_idx:i+1])
                # Start new segment at current point
                start_idx = i
                accumulated_dist = 0.0
            else:
                accumulated_dist += dist

        # Add any remaining points
        if start_idx < len(points) - 1:
            sub_segments.append(points[start_idx:])

        return sub_segments

    all_segments = []
    all_styles = []  
    all_widths = [] 

    # Process each contour level
    for i, level_val in enumerate(CS.levels):
        segs_for_level = CS.allsegs[i]
        if not segs_for_level:
            continue

        for seg in segs_for_level:
            if len(seg) < 2:
                continue

            # Break into geometry-based sub-segments
            sub_segs = chunk_by_distance(seg, MAX_SUBSEG_LEN)

            cumulative_offset = 0.0

            for xy_sub in sub_segs:
                # Skip if sub-segment is too short
                if len(xy_sub) < 2:
                    continue

                # Get local attribute2 values for this segment (for dash pattern)
                points_for_interp = np.column_stack((xy_sub[:, 1], xy_sub[:, 0]))
                attr2_vals = []
                
                for point in points_for_interp:
                    try:
                        val = attribute2_interpolator(point)
                        if not np.isnan(val):
                            attr2_vals.append(val)
                    except:
                        pass
                
                # Skip if no valid  values
                if not attr2_vals:
                    continue

                local_attr2 = np.mean(attr2_vals)
                
                dash_len, gap_len = attr2_to_dash_len(local_attr2, attribute2_min, attribute2_max, scale_factor)
                
                # Create line style with appropriate offset
                style = (cumulative_offset, (dash_len, gap_len))
                
                # Get local attribute4 values for this segment (for line width) if provided
                if attribute4 is not None:
                    attr4_vals = []
                    
                    for point in points_for_interp:
                        try:
                            val = attribute4_interpolator(point)
                            if not np.isnan(val):
                                attr4_vals.append(val)
                        except:
                            # traceback.format_exc()
                            pass
                    
                    # Calculate linewidth based on attribute4
                    if attr4_vals and attribute4_max != attribute4_min:
                        local_attr4 = np.mean(attr4_vals)
                        norm_attr4 = (local_attr4 - attribute4_min) / (attribute4_max - attribute4_min)
                        width = min_linewidth + norm_attr4 * (max_linewidth - min_linewidth)
                    else:
                        width = (min_linewidth + max_linewidth) / 2
                else:
                    # uniform width when attribute4 is not provided
                    width = (min_linewidth + max_linewidth) / 2
                
                # Add positional jitter to coordinates
                jitter_scale = 0.0003  #adjust based on  coordinate system
                xy_jittered = xy_sub + np.random.normal(0, jitter_scale, xy_sub.shape)
                
                # Add width jitter (preserve min/max bounds)
                width_jitter = width * (1 + 0.2 * (np.random.rand() - 0.5))
                final_width = np.clip(width_jitter, min_linewidth, max_linewidth)
                
                all_segments.append(xy_jittered)  
                all_styles.append(style)
                all_widths.append(final_width)

                # Measure the sub-segment length for offset calculation
                seg_length = 0.0
                for i in range(len(xy_sub) - 1):
                    dx = xy_sub[i+1, 1] - xy_sub[i, 1]
                    dy = xy_sub[i+1, 0] - xy_sub[i, 0]
                    seg_length += np.hypot(dx, dy)

                # Calculate pattern length and update offset
                pattern_len = dash_len + gap_len
                if pattern_len > 0:
                    cumulative_offset = (cumulative_offset + seg_length) % pattern_len

    # Create line collection with calculated properties
    if all_segments:  # Only create if we have segments
        lc = LineCollection(
            all_segments,
            linestyles=all_styles,
            colors="k",
            linewidths=all_widths,
            zorder=3
        )
        dataaxis.add_collection(lc)
    else:
        print("No valid segments to plot.")


    print(f"Attribute2 range used for dash patterns: [{attribute2_min:.4f}, {attribute2_max:.4f}]")
    if attribute4 is not None:
        print(f"Attribute4 range used for line widths: [{attribute4_min:.4f}, {attribute4_max:.4f}]")

    # plot the background attribute as pcolormesh
    if attribute3 is not None:
        attribute3_absmax = np.maximum(np.absolute(attribute3_min), np.absolute(attribute3_max))
        attribute3_absnorm = np.abs(attribute3) / attribute3_absmax
        # attribute3_clipabs = np.maximum(np.minimum((attribute3_absnorm - 0.1) / 0.2, 1.0), .0)

        # Plot attribute3 as background
        cs_attr3 = dataaxis.pcolormesh(
            lon,
            lat,
            attribute3,
            shading='gouraud',
            cmap=attribute3_cmap,
            norm=attribute3_norm,
            # alpha=0.6,
            alpha=attribute3_absnorm,
            zorder=2
        )
        # TODO: isn't rendered = where are they ? those need to be axis elements, not figure elements, if I'm not mistaken ...
        # TODO: [DONE]
        # cbar_attr3 = fig.colorbar(cs_attr3, ax=ax, label=attribute3_label)
    
    # Plot shoreline if signed_dist is provided
    if signed_dist is not None:
        shoreline = dataaxis.contour(lon, lat, signed_dist, levels=[0],
                             colors='black', linewidths=0.8, linestyles='solid', zorder=4)


    # Add grid lines for scale reference
    grid_alpha = 0.5  # Transparency of grid lines
    grid_color = 'gray'
    grid_linewidth = 0.5
    grid_linestyle = '--'
    
    # Calculate appropriate grid spacing based on data bounds
    lon_range = lon.max() - lon.min()
    lat_range = lat.max() - lat.min()
    
    # Define grid spacing
    lon_spacing = lon_range / 8
    lat_spacing = lat_range / 8  
    
    # Round to nice values for the spacing
    lon_spacing = round_to_nice(lon_spacing)
    lat_spacing = round_to_nice(lat_spacing)
    
    # Calculate grid line positions
    lon_start = np.floor(lon.min() / lon_spacing) * lon_spacing
    lon_end = np.ceil(lon.max() / lon_spacing) * lon_spacing
    lat_start = np.floor(lat.min() / lat_spacing) * lat_spacing
    lat_end = np.ceil(lat.max() / lat_spacing) * lat_spacing
    
    lon_grid = np.arange(lon_start, lon_end + lon_spacing/2, lon_spacing)
    lat_grid = np.arange(lat_start, lat_end + lat_spacing/2, lat_spacing)
    
    # Draw grid lines
    dataaxis.grid(False)  # Disable default grid
    
    # Add custom grid lines
    for x in lon_grid:
        dataaxis.axvline(x=x, color=grid_color, linestyle=grid_linestyle,
                  linewidth=grid_linewidth, alpha=grid_alpha, zorder=0)
    
    for y in lat_grid:
        dataaxis.axhline(y=y, color=grid_color, linestyle=grid_linestyle,
                  linewidth=grid_linewidth, alpha=grid_alpha, zorder=0)
    
    # Add ticks at grid line positions
    dataaxis.set_xticks(lon_grid)
    dataaxis.set_yticks(lat_grid)
    
    # Format tick labels to reduce clutter
    dataaxis.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))
    dataaxis.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.1f'))

    # Add compass rose if signed_dist is provided
    if signed_dist is not None:
        compass_position = add_compass_rose(
            dataaxis, lon, lat, signed_dist,
            size_factor=0.06, 
            style='traditional',
            color='black',
            border_color='white',
            alpha=1.0,
        )
        print(f"Added compass rose at position {compass_position}")

    # Set title and labels
    dataaxis.set_title(plot_title)  # => figure
    dataaxis.set_xlabel('lon [°]')
    dataaxis.set_ylabel('lat [°]')
    # dataaxis.set_xlim(lon.min(), lon.max())
    # dataaxis.set_ylim(lat.min(), lat.max())

    # ------------------------------------------------------------------------------ #
    # LEGEND & COLOUR BARS
    # ------------------------------------------------------------------------------ #

    # Legend / colourbar for the contour density itself (see previous contour plots)
    # ax_cbar_contourdensity = datafig.add_axes([0.055, 0.1, 0.02, 0.8])
    ax_cbar_contourdensity = datafig.add_axes([0.06, 0.1, 0.02, 0.8])
    ax_cbar_contourdensity.set_facecolor("white")
    if attribute1 is not None:
        sm = cm.ScalarMappable(cmap=LinearSegmentedColormap.from_list("Blacks", [(1,1,1), (0,0,0)]), norm=Normalize(vmin=min_attr1, vmax=max_attr1))
        # sm.set_array(np.linspace(min_attr1, max_attr1, num_contours))
        sm.set_array(contour_levels)
        cbar_attr1 = datafig.colorbar(sm, cax=ax_cbar_contourdensity, ticks=contour_levels, orientation='vertical')  # , extend='both', format='%.0e'), ticker.MaxNLocator(num_contours)
        ctrbar_label = "Contour levels: "
        if attribute1_label == "Topography":
            ctrbar_label += "Elevation (m)"
        elif attribute1_label == "Bathymetry":
            ctrbar_label += "Depth (m)"
        elif attribute1_label == "Temperature":
            ctrbar_label += "Temperature (°C)"
        elif attribute1_label == "VelocityMagnitude":
            ctrbar_label += "Velocity Magnitude (m/s)"
        elif attribute1_label == "Divergence":
            ctrbar_label += "Divergence (1/s)"
        cbar_attr1.set_label(ctrbar_label)
        # ax_cbar_contourdensity.xaxis.set_ticks_position('bottom')
        # ax_cbar_contourdensity.xaxis.set_label_position('top')
        ax_cbar_contourdensity.yaxis.set_ticks_position('right')
        ax_cbar_contourdensity.yaxis.set_label_position('left')

    # Legend / colourbar for the background colourmap
    ax_cbar_bg_colourmap = datafig.add_axes([0.92, 0.1, 0.02, 0.8])
    ax_cbar_bg_colourmap.set_facecolor("white")
    # ax_cbar_bg_colourmap.yaxis.set_ticks_position('right')
    # ax_cbar_bg_colourmap.yaxis.set_label_position('left')
    if attribute3 is not None:
        cbar_attr3 = datafig.colorbar(cs_attr3, cax=ax_cbar_bg_colourmap, ticks=ticker.AutoLocator(), orientation='vertical')
        cbar_label = ""
        if attribute3_label == "Topography":
            cbar_label = "Elevation (m)"
        elif attribute3_label == "Bathymetry":
            cbar_label += "Depth (m)"
        elif attribute3_label == "Temperature":
            cbar_label = "Temperature (°C)"
        elif attribute3_label == "VelocityMagnitude":
            cbar_label = "Velocity Magnitude (m/s)"
        elif attribute3_label == "Divergence":
            cbar_label = "Divergence (1/s)"
        cbar_attr3.set_label(cbar_label)
        # ax_cbar_bg_colourmap.xaxis.set_ticks_position('top')
        # ax_cbar_bg_colourmap.xaxis.set_label_position('bottom')
        ax_cbar_bg_colourmap.yaxis.set_ticks_position('left')
        ax_cbar_bg_colourmap.yaxis.set_label_position('right')

    legend_elements = []
    # Legend / colourbar for contour dash pattern
    if not isinstance(all_styles, str) and len(all_styles) > 1:
        unit_base = ""
        if attribute2_label == "Topography":
            # label_base = "Elevation (m)"
            unit_base = "m"
        elif attribute2_label == "Bathymetry":
            # label_base = "Depth (m)"
            unit_base = "m"
        elif attribute2_label == "Temperature":
            # label_base = "Temperature (°C)"
            unit_base = "°C"
        elif attribute2_label == "VelocityMagnitude":
            # label_base = "Velocity Magnitude ()"
            unit_base = "m/s"
        elif attribute2_label == "Divergence":
            # label_base = "Divergence (1/s)"
            unit_base = "1/s"
        for i in range(len(legend_contourstyles)):
            itemlabel = "{:.3f} {}".format(legend_contourstyles_labels[i], unit_base)
            legend_elements.append(Line2D([0], [0], color='black', lw=1, linestyle=legend_contourstyles[i], label=itemlabel))

    # Legend / colourbar for contour width
    if not isinstance(all_widths, str):
        unit_base = ""
        if attribute4_label == "Topography":
            unit_base = "m"
        elif attribute4_label == "Bathymetry":
            unit_base = "m"
        elif attribute4_label == "Temperature":
            unit_base = "°C"
        elif attribute4_label == "VelocityMagnitude":
            unit_base = "m/s"
        elif attribute4_label == "Divergence":
            unit_base = "1/s"
        for i in range(len(legend_contourwidths)):  # TODO: pregenerate that field of contour width [DONE]
            widthitem = legend_contourwidths[i]
            widthlabel = legend_contourwidths_labels[i]
            itemlabel = "{:.3f} {}".format(widthlabel, unit_base)
            legend_elements.append(Line2D([0], [0], color='black', lw=widthitem, label=itemlabel))

    # Finalize the in-plot legend
    dataaxis.legend(handles=legend_elements, loc='upper right')  # , bbox_to_anchor=(0.5, -0.05),
    # Add scale bar
    add_scale_bar(dataaxis, lon_range, lat_range)
    # plt.tight_layout()

    # Save the figure
    plt.savefig(output_image, dpi=plot_dpi, bbox_inches='tight') # , bbox_inches='tight'
    print(f"Figure saved to {output_image}")
    plt.close()
    
    # Return generated elements and auto-scaled parameters for potential further use
    # return {
    #     'segments': all_segments,
    #     'styles': all_styles,
    #     'widths': all_widths,
    #     'auto_scaled_params': {
    #         'MIN_DASH': MIN_DASH,
    #         'MAX_DASH': MAX_DASH,
    #         'min_linewidth': min_linewidth,
    #         'max_linewidth': max_linewidth,
    #         'scale_factor': scale_factor,
    #         'chunk_size': chunk_size,
    #         'attribute2_range': (attribute2_min, attribute2_max),
    #         'attribute4_range': (attribute4_min, attribute4_max) if attribute4 is not None else None
    #     }
    # }
    return True

def load_sample_data():
    """
    Load and prepare oceanographic data for visualisation.
    """
    # Load dataset for elevation
    ds = xr.open_dataset(os.path.join(bathytopograhy_dir,bathytopograhy_file))
    lat = ds["lat"].data        # 1D array of latitudes
    lon = ds["lon"].data        # 1D array of longitudes
    elevation = ds["elevation"].data # 2D array [lat, lon]
    # TODO: this clamping to a squared area (|lon| = |lat|) is something I don't fully understand the purpose for ...
    min_size = min(len(lon), len(lat))
    lon = lon[:min_size]
    lat = lat[:min_size]
    elevation = elevation[:min_size, :min_size]

    # Load dataset for flow data & replace NaNs with small values
    ds2 = xr.open_dataset(os.path.join(currents_dir, currents_file), decode_cf=False, engine='netcdf4')
    # u = ds2["uo"].isel(time=0)  # Eastward velocity (m/s)
    # v = ds2["vo"].isel(time=0)  # Northward velocity (m/s)
    # u = np.squeeze(ds2["uo"].data[0])
    u = np.squeeze(ds2["uo"].data)[0]
    u_attrs = ds2["uo"].attrs
    # print("u-attrs: {}".format(u_attrs))
    if "scale_factor" in u_attrs.keys():
        scalefactor = u_attrs["scale_factor"]
        print("U has a scale-factor of {}. Rescaling values ...".format(scalefactor))
        u = u * scalefactor
    if "add_offset" in u_attrs.keys():
        offset = u_attrs["add_offset"]
        print("U has an active offset of {}. Applying offset ...".format(offset))
        u = u + offset
    # u = u.fillna(.0)
    u = np.nan_to_num(u, copy=False, nan=.0, posinf=.0, neginf=.0)
    # print(u.shape)
    v = np.squeeze(ds2["vo"].data)[0]
    v_attrs = ds2["vo"].attrs
    # print("v-attrs: {}".format(v_attrs))
    if "scale_factor" in v_attrs.keys():
        scalefactor = v_attrs["scale_factor"]
        print("V has a scale-factor of {}. Rescaling values ...".format(scalefactor))
        v = v * scalefactor
    if "add_offset" in v_attrs.keys():
        offset = v_attrs["add_offset"]
        print("V has an active offset of {}. Applying offset ...".format(offset))
        v = v + offset
    # v = v.fillna(.0)
    v = np.nan_to_num(v, copy=False, nan=.0, posinf=.0, neginf=.0)
    in_lat = ds2["latitude"].data
    in_lon = ds2["longitude"].data
    # print("in_lat {}; in_lon {}".format(in_lat.shape, in_lon.shape))
    
    # Load dataset for temperature data
    ds3 = xr.open_dataset(os.path.join(temperature_dir, temperature_file))
    # temp = ds3["thetao"].isel(time=0)  # Temperature (°C)
    temp = np.squeeze(ds3["thetao"].data)[0]  # Temperature (°C)
    # temp = temp.fillna(.0)
    temp = np.nan_to_num(temp, copy=False, nan=.0, posinf=.0, neginf=.0)

    # Identify the shoreline
    shoreline_mask = np.where(elevation >= 0, 1, 0).astype(np.uint32)
    erosion = ndi.binary_erosion(shoreline_mask, structure=np.ones((3, 3)))
    shoreline_mask = shoreline_mask & ~erosion

    # Compute the distance transform 
    dist_to_shore = distance_transform_edt(np.where(shoreline_mask == 0, 1, 0), sampling=[np.abs(lat[1] - lat[0]), np.abs(lon[1] - lon[0])])
    
    # Assign sign based on whether elevation is above or below zero
    # TODO: usually, the sign here should be switched: positive for outside, negative for inside
    signed_dist = dist_to_shore.copy()
    signed_dist[elevation < 0] *= -1
    # Create mask for water regions
    water_mask = (signed_dist <= 0)

    # Interpolate flow data to match elevation grid
    mesh_lat, mesh_lon = np.meshgrid(lat, lon, sparse=False, indexing='ij')
    # == elevation_ij = (mesh_lat.flatten(), mesh_lon.flatten()) == #
    # u = u.interp(latitude=lat, longitude=lon, method="linear")  # method="cubic"
    interp_u = RegularGridInterpolator((in_lat, in_lon), u, bounds_error=False, fill_value=.0)
    u = interp_u((mesh_lat, mesh_lon))  # method="cubic"
    # v = v.interp(latitude=lat, longitude=lon, method="linear")  # method="cubic"
    interp_v = RegularGridInterpolator((in_lat, in_lon), v, bounds_error=False, fill_value=.0)
    v = interp_v((mesh_lat, mesh_lon))  # method="cubic"
    # temp = temp.interp(latitude=lat, longitude=lon, method="linear")  # method="cubic"
    interp_temp = RegularGridInterpolator((in_lat, in_lon), temp, bounds_error=False, fill_value=.0)
    temp = interp_temp((mesh_lat, mesh_lon))  # method="cubic"

    # TODO: again: why are we clamping the fields to a squared area ?
    u = u[:min_size, :min_size]
    v = v[:min_size, :min_size]
    temp = temp[:min_size, :min_size]

    u = np.where(water_mask, u, .0)  # u.where(water_mask)
    v = np.where(water_mask, v, .0)  # v.where(water_mask)
    temp = np.where(water_mask, temp, .0)  # temp.where(water_mask)

    # Compute the magnitude of the flow field
    velmag = u**2 + v**2
    velmag = np.where(velmag > .0, np.sqrt(velmag), .0)
    velmag_invalid = np.isclose(velmag, .0) & np.isclose(velmag, -.0)
    velmag_min = np.finfo(velmag.dtype).eps if velmag_invalid.all() else np.nanmin(velmag[~velmag_invalid])
    velmag[velmag_invalid] = velmag_min


    # Calculate flow divergence
    lon_spacing = float(np.abs(lon[1] - lon[0]))
    lat_spacing = float(np.abs(lat[1] - lat[0]))
    
    # Calculate gradients
    # u_vals = u.values.copy()
    dudx = np.zeros_like(u)
    dudx[:, 1:-1] = (u[:, 2:] - u[:, :-2]) / (2 * lon_spacing)
    dudx[:, 0] = (u[:, 1] - u[:, 0]) / lon_spacing
    dudx[:, -1] = (u[:, -1] - u[:, -2]) / lon_spacing
    
    # v_vals = v.values.copy()
    dvdy = np.zeros_like(v)
    dvdy[1:-1, :] = (v[2:, :] - v[:-2, :]) / (2 * lat_spacing)
    dvdy[0, :] = (v[1, :] - v[0, :]) / lat_spacing
    dvdy[-1, :] = (v[-1, :] - v[-2, :]) / lat_spacing
    
    # Compute divergence
    flow_divergence = dudx + dvdy
    
    # Create divergence dataarray
    # divergence = xr.DataArray(
    #     data=flow_divergence,
    #     dims=("lat", "lon"),
    #     coords={"lat": lat, "lon": lon}
    # )
    
    # Mask out non-water areas in the divergence field
    # divergence = divergence.where(water_mask)
    flow_divergence = np.where(water_mask, flow_divergence, np.NaN)
    
    # Replace NaNs and handle infinite values
    flow_divergence = np.nan_to_num(flow_divergence, nan=0, posinf=0, neginf=0)
    # divergence.values = flow_divergence
    

    # TODO: again: why are we clamping the fields to a squared area ?
    # divergence = divergence[:min_size, :min_size]
    flow_divergence = flow_divergence[:min_size, :min_size]
    velmag = velmag[:min_size, :min_size]
    temp = temp[:min_size, :min_size]

    # Create divergence norm
    # divergence_min = np.nanpercentile(flow_divergence[water_mask], 5)
    # divergence_max = np.nanpercentile(flow_divergence[water_mask], 95)
    divergence_min = np.nanpercentile(flow_divergence, 5)
    divergence_max = np.nanpercentile(flow_divergence, 95)
    divergence_abs_max = max(abs(divergence_min), abs(divergence_max))
    divergence_norm = TwoSlopeNorm(vmin=-divergence_abs_max, vcenter=0, vmax=divergence_abs_max)
    
    # Normalize velocity components
    abs_max_u = max(np.abs(np.nanmin(u)), np.abs(np.nanmax(u)))
    normu = TwoSlopeNorm(vmin=-abs_max_u, vcenter=0, vmax=abs_max_u)

    abs_max_v = max(np.abs(np.nanmin(v)), np.abs(np.nanmax(v)))
    normv = TwoSlopeNorm(vmin=-abs_max_v, vcenter=0, vmax=abs_max_v)

    # Norm for flow magnitude
    velmag_max = np.nanmax(velmag)
    velmag_min = 0
    norm_speed = Normalize(vmin=velmag_min, vmax=velmag_max)

    # Modified elevation for topography -> TODO: call it topography !!!! [DONE]
    topography = np.where(elevation <= 0, .0, elevation)

    # Modified elevation for bathymetry
    bathymetry = np.abs(np.where(elevation < .0, elevation, .0))
    
    # Normalize temperature data
    temp_min = np.nanmin(temp)
    temp_max = np.nanmax(temp)
    # temp_values = temp.values.copy()
    temp = np.nan_to_num(temp, nan=temp_min, posinf=temp_max, neginf=temp_min)
    
    # Create a log norm for temperature
    temp_log_min = 0.1  # Minimum value for log scale
    temp_min_safe = max(temp_min, temp_log_min)
    temp_log_norm = LogNorm(vmin=temp_min_safe, vmax=temp_max)
    
    # Calculate normalized temperature (0-1 range)
    temp_range = temp_max - temp_min
    if temp_range > 0:
        normalized_temp = (temp - temp_min) / temp_range
    else:
        normalized_temp = np.zeros_like(temp)


    return {
        'lat': lat, 
        'lon': lon, 
        'elevation': elevation,
        'topography': topography,
        'bathymetry': bathymetry,
        'u': u, 
        'v': v, 
        'temp': temp,
        'velmag': velmag,  # formerly: flow_mag
        'flow_divergence': flow_divergence,  # TODO: change to divergence ? what's the difference ?
        # 'divergence': divergence,  # removed as it is just the xarray duplicate of 'flow_divergence'
        'signed_dist': signed_dist,
        'water_mask': water_mask,
        'normalized_temp': normalized_temp,  # TODO: normalized to what ?
        'divergence_norm': divergence_norm,
        'norm_speed': norm_speed,
        'normu': normu,
        'normv': normv,
        'temp_log_norm': temp_log_norm,
        'velmag_min': velmag_min,
        'velmag_max': velmag_max,
        'temp_min': temp_min,
        'temp_max': temp_max
    }

def generate_all_possible_mappings(data, output_dir="visualisation_outputs"):
    """
    Generate all possible variable-to-visual-property mappings for the three
    visualisation techniques: hatches, stipples, and flow contours.
    
    Parameters:
    -----------
    data : dict
        Dictionary containing the loaded oceanographic data
    output_dir : str, optional
        Directory to save the output visualisations, default is "visualisation_outputs"
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define available variables and their descriptive names
    variables = {
        'temp': 'Temperature',
        'velmag': 'VelocityMagnitude',
        'flow_divergence': 'Divergence',
        'topography': 'Topography',
        'bathymetry': 'Bathymetry'
        # 'elev_modified': 'Depth'
    }
    
    # Define flow direction variables
    flow_direction = {
        'u': data['u'],
        'v': data['v']
    }
    
    print("Generating visualisations for all possible mappings...")
    
    # ==== Generate all combinations for hatch visualisations ==== #
    generate_all_hatch_combinations(data, variables, flow_direction, output_dir)
    
    # ==== Generate all combinations for stipple visualisations ==== #
    # generate_all_stipple_combinations(data, variables, output_dir)
    
    # ==== Generate all combinations for flow contour visualisations ==== #
    # generate_all_flow_combinations(data, variables, output_dir)
    
    print(f"All possible mappings generated and saved to {output_dir}")

def generate_all_hatch_combinations(data, variables, flow_direction, output_dir):
    """
    Generate all possible combinations for hatch visualisations.
    
    For hatches, we need:
    - attribute1: density
    - attribute2/3: direction (fixed to u/v)
    - attribute4: line length (variable or uniform)
    - attribute5: background color (optional)
    - attribute6: line width (variable or uniform)
    """
    print("Generating all hatch combinations...")
    
    # Create a list of all combinations
    # For density (attribute1)
    density_vars = list(variables.keys())
    
    # For line length (attribute4)
    length_vars = list(variables.keys()) + [None]  # None means uniform length
    
    # For color (attribute5)
    color_vars = list(variables.keys()) + [None]  # None means no background colour
    
    # For line width (attribute6)
    width_vars = list(variables.keys()) + [None]  # None means uniform width
    
    # Generate all combinations
    combinations = list(itertools.product(density_vars, length_vars, color_vars, width_vars))
    
    # Filter out invalid combinations (same variable used for multiple properties)
    valid_combinations = []
    for combo in combinations:
        density_var, length_var, color_var, width_var = combo
        
        # Check that each non-None variable is used at most once
        used_vars = [var for var in [density_var, length_var, color_var, width_var] if var is not None]
        if len(used_vars) == len(set(used_vars)):
            valid_combinations.append(combo)
    
    print(f"Found {len(valid_combinations)} valid hatch combinations")

    # -------------------------------------------------- #
    # Generate visualisations for each valid combination #
    # -------------------------------------------------- #
    for i, combo in enumerate(valid_combinations):
        density_var, length_var, color_var, width_var = combo  # Unpack all four variables
        
        # Skip some combinations to make the total number manageable
        if i % 10 != 0 and len(valid_combinations) > 50:
            continue
            
        # Create descriptive filename components
        density_part = f"{variables[density_var]}-Density"
        orient_part = "VelocityDirection-Orientation"
        
        if length_var is None:
            length_part = "Uniform-Length"
        else:
            length_part = f"{variables[length_var]}-Length"
            
        if width_var is None:
            width_part = "Uniform-Width"
        else:
            width_part = f"{variables[width_var]}-Width"
            
        if color_var is None:
            color_part = "No-BgColor"
        else:
            color_part = f"_{variables[color_var]}-BgColor"
        
        # Combine filename components
        filename = f"Hatch_{density_part}_{orient_part}_{length_part}_{width_part}{color_part}.png"
        
        output_path = os.path.join(output_dir, filename)
        
        print(f"Generating hatch visualisation {i+1}/{len(valid_combinations)}: {filename}")
        
        # Get the appropriate variables
        attribute1 = data[density_var]
        attribute1_min = np.nanmin(attribute1)
        attribute1_max = np.nanmax(attribute1)
        attribute2 = flow_direction['u']
        attribute3 = flow_direction['v']
        
        # Set up attribute4 (length)
        if length_var is None:
        # For UniformLength, create a dummy array instead of using None
            uniform_length_value = 0.5  # Medium value
            attribute4 = np.ones(attribute1.shape, dtype=np.float32) * uniform_length_value
            # attribute4 = xr.DataArray(
            #     data=np.ones_like(attribute1.values) * uniform_length_value,
            #     dims=attribute1.dims,
            #     coords=attribute1.coords,
            #     name="uniform_length"
            # )
            print(f"Created uniform length attribute4 with shape {attribute4.shape}")
        else:
            attribute4 = data[length_var]
        
        # Set up attribute5 (color) and related parameters
        attribute5 = data[color_var] if color_var else None
        use_attribute5_for_color = (color_var is not None)
        
        # Set up attribute6 (width)
        attribute6 = data[width_var] if width_var else None
        
        # Normalize attribute1 (density controlling variable)
        attribute1_range = attribute1_max - attribute1_min
        if attribute1_range > 0:
            # # if isinstance(attribute1.values, np.ndarray):
            # if isinstance(attribute1, np.ndarray):
            #     # normalized_attribute1 = (attribute1.values - attribute1_min) / attribute1_range
            #     normalized_attribute1 = (attribute1 - attribute1_min) / attribute1_range
            # else:
            #     normalized_attribute1 = (attribute1 - attribute1_min) / attribute1_range
            normalized_attribute1 = (attribute1 - attribute1_min) / attribute1_range
        else:
            normalized_attribute1 = np.zeros_like(attribute1.values)
        
        # Determine appropriate colormap and normalisation for attribute5
        if color_var is None:
            attribute5 = None
            attribute5_cmap = None
            attribute5_norm = None
            use_attribute5_for_color = False
        else:
            attribute5 = data[color_var]
            # attribute5 = attribute5.rename(variables[color_var])
            use_attribute5_for_color = True

            #  Build an appropriate normalisation
            if color_var in ["flow_divergence", "temp"]:
                # Diverging: centre on zero
                vmax = np.nanpercentile(np.abs(attribute5), 95)
                attribute5_norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
                # attribute5_cmap = plt.cm.get_cmap("RdBu_r")
                attribute5_cmap = plt.colormaps.get_cmap("bwr")
            else:
                # Sequential: simple linear 5-th→95-th percentile stretch
                vmin, vmax = np.nanpercentile(attribute5, (5, 95))
                attribute5_norm = plt.Normalize(vmin=vmin, vmax=vmax)
                if color_var == "velmag":
                    attribute5_cmap = plt.colormaps.get_cmap("PuBu")
                elif color_var == "topography":  # elev_modified
                    # attribute5_cmap = plt.cm.get_cmap("BrBG_r")
                    attribute5_cmap = plt.colormaps.get_cmap("YlOrBr")
                elif color_var == "bathymetry":  # elev_modified
                    # attribute5_cmap = plt.cm.get_cmap("BrBG_r")
                    attribute5_cmap = plt.colormaps.get_cmap("Greys")
                # elif color_var == "temp":
                #     attribute5_cmap = plt.cm.get_cmap("bwr")
                else:                            
                    # attribute5_cmap = plt.cm.get_cmap("viridis")
                    attribute5_cmap = plt.colormaps.get_cmap("Greys")
        
        # Create title based on mappings
        title_parts = [
            f"{variables[density_var]} (Stroke Density)",
            "Veloctity Orientation (Stroke Orientation)"
        ]
        
        if length_var is None:
            title_parts.append("Uniform Stroke Length")
        else:
            title_parts.append(f"{variables[length_var]} (Stroke Length)")
            
        if width_var is None:
            title_parts.append("Uniform Stroke Width")
        else:
            title_parts.append(f"{variables[width_var]} (Stroke Width)")
            
        if color_var:
            title_parts.append(f"{variables[color_var]} (Background Color)")
            
        plot_title = f"Hatch: {', '.join(title_parts)}"
        if skip_title_plotting:
            plot_title = ""
        
        # Call  hatch function with the configured parameters
        try:
            hatch(
                attribute1=attribute1,
                attribute1_max=attribute1_max,
                attribute1_min=attribute1_min,
                attribute2=attribute2,
                attribute3=attribute3,
                attribute4=attribute4,  # Can be None for uniform length
                attribute5=attribute5 if color_var else None,
                attribute6=attribute6,  # Can be None for uniform width
                signed_dist=data['signed_dist'],
                lon=data['lon'],
                lat=data['lat'],
                water_mask=data['water_mask'],
                elev=data['elevation'],
                topography=data["topography"],  # elev_modified
                normalized_attribute1=normalized_attribute1,
                attribute5_norm=attribute5_norm,
                attribute5_cmap=attribute5_cmap,
                output_image=output_path,
                plot_title=plot_title,
                attribute1_label=f"{variables[density_var]}",
                attribute2_label="U-velocity",
                attribute3_label="V-velocity",
                attribute4_label=f"{variables[length_var]}" if length_var else "",
                attribute5_label=f"{variables[color_var]}" if color_var else "",
                attribute6_label=f"{variables[width_var]}" if width_var else "",
                use_attribute5_for_color=use_attribute5_for_color,
                base_density=0.08,  # Adjust for better visualisation
                auto_scale = True
            )  # TODO: to be modfied by plotting title
        except Exception as e:
            print(f"Error generating {filename}: {e}")
            # traceback.format_exc()
            logger.exception(e)
    

def generate_all_stipple_combinations(data, variables, output_dir):
    """
    Generate all possible combinations for stipple visualisations with three attributes.
   
    For stipples, we need:
    - attribute1: contour lines
    - attribute2: point density/spacing
    - attribute3: background coloration (optional)
    
    Ensures the same attribute is never used for multiple visual encodings.
    """
    print("Generating all stipple combinations...")
   
    # Create lists of variables for each attribute
    contour_vars = list(variables.keys())
    density_vars = list(variables.keys())
    background_vars = list(variables.keys()) + [None]  
   
    # Generate all combinations
    combinations = list(itertools.product(contour_vars, density_vars, background_vars))
   
    # Filter out invalid combinations
    valid_combinations = []
    for combo in combinations:
        contour_var, density_var, background_var = combo
        
        # Check if all used variables are different (when not None)
        used_vars = [contour_var, density_var]
        if background_var is not None:
            used_vars.append(background_var)
            
        if len(set(used_vars)) == len(used_vars): 
            valid_combinations.append(combo)
   
    print(f"Found {len(valid_combinations)} valid stipple combinations")
   
    # Generate visualisations for each valid combination
    for i, combo in enumerate(valid_combinations):
        contour_var, density_var, background_var = combo
       
        # Skip some combinations to make the total number manageable
        if i % 2 != 0 and len(valid_combinations) > 20:
            continue
           
        # Create descriptive filename
        filename = f"Stipple_{variables[contour_var]}-Contour_{variables[density_var]}-Density"
        if background_var:
            filename += f"_{variables[background_var]}-Background"
        filename += ".png"
       
        output_path = os.path.join(output_dir, filename)
       
        print(f"Generating stipple visualisation {i+1}/{len(valid_combinations)}: {filename}")
       
        # Get the appropriate variables and normalizations
        attribute1 = data[contour_var]  # For contours
        attribute1_min = np.nanmin(attribute1)
        attribute1_max = np.nanmax(attribute1)
        
        attribute2 = data[density_var]  # For density
        attribute2_min = np.nanmin(attribute2)
        attribute2_max = np.nanmax(attribute2)
       
        # Set up attribute3 (background) if specified
        attribute3 = data[background_var] if background_var else None
       
        # Determine appropriate colormap based on the background variable
        
        
        if background_var == 'flow_divergence':
            attribute3_cmap = 'RdBu_r' 
            attribute3_min = np.nanmin(data['flow_divergence'])
            attribute3_max = np.nanmax(data['flow_divergence'])
            attribute3_norm = TwoSlopeNorm(vmin=-abs(attribute3_max), vcenter=0, vmax=abs(attribute3_max))
        else:
            #normalize background variable
            attribute3_min = np.nanmin(attribute3)
            attribute3_max = np.nanmax(attribute3)
            attribute3_norm = Normalize(vmin=attribute3_min, vmax=attribute3_max)
            if background_var == 'temp':
                attribute3_cmap = plt.colormaps.get_cmap('bwr')
            elif background_var == 'velmag':
                attribute3_cmap = plt.colormaps.get_cmap('PuBu')
            elif background_var == "topography":  # elev_modified
                # attribute3_cmap = 'BrBG_r'
                attribute3_cmap = plt.colormaps.get_cmap("YlOrBr")
            elif background_var == "bathymetry":
                attribute3_cmap = plt.colormaps.get_cmap('Greys')
            else:
                attribute3_cmap = plt.colormaps.get_cmap('Greys')

        plot_title = f"Stipple: {variables[contour_var]} (Contours), {variables[density_var]} (Stipple Density)" + (f", {variables[background_var]} (Background)" if background_var else "")
        if skip_title_plotting:
            plot_title = ""

        # Call the stipple function with the configured parameters
        try:
            stipple(
                lon=data['lon'],
                lat=data['lat'],
                attribute1=attribute1,  # For contours
                attribute2=attribute2,  # For density
                attribute3=attribute3,  # For background (optional)
                attribute3_norm=attribute3_norm,
                signed_dist=data['signed_dist'],
                attribute1_min=attribute1_min,
                attribute1_max=attribute1_max,
                attribute2_min=attribute2_min,
                attribute2_max=attribute2_max,
                attribute1_label=f"{variables[contour_var]}",
                attribute2_label=f"{variables[density_var]}",
                attribute3_label=f"{variables[background_var]}" if background_var else None,
                plot_title=plot_title,  # TODO: to be modfied [DONE]
                output_image=output_path,
                attribute3_cmap=attribute3_cmap,
                num_contours=6,
                marker_size=8,
                flow_scaling=0.7,
                auto_scale=True
            )
        except Exception as e:
            print(f"Error generating {filename}: {e}")
            # traceback.format_exc()
            logger.exception(e)

def generate_all_flow_combinations(data, variables, output_dir):
    """
    Generate all possible combinations for flow contour visualisations.
    
    For flow contours, we need:
    - attribute1: contour lines
    - attribute2: dash pattern
    - attribute3: background coloration (optional)
    - attribute4: line width (optional, uniform if None)
    """
    print("Generating all flow contour combinations...")
    
    # Create a list of all combinations
    # For contours (attribute1)
    contour_vars = list(variables.keys())
    
    # For dash pattern (attribute2)
    dash_vars = list(variables.keys())
    
    # For background (attribute3)
    background_vars = list(variables.keys()) + [None] 
    
    # For line width (attribute4)
    width_vars = list(variables.keys()) + [None]  # None means uniform width
    
    # Generate all combinations
    combinations = list(itertools.product(contour_vars, dash_vars, background_vars, width_vars))
    
    # Filter out invalid combinations (same variable used for multiple properties)
    valid_combinations = []
    for combo in combinations:
        contour_var, dash_var, background_var, width_var = combo
        # Ensure each variable is used at most once
        used_vars = [var for var in [contour_var, dash_var, background_var, width_var] if var is not None]
        if len(used_vars) == len(set(used_vars)):
            valid_combinations.append(combo)
    
    print(f"Found {len(valid_combinations)} valid flow contour combinations")
    
    # Generate visualisations for each valid combination
    for i, combo in enumerate(valid_combinations):
        contour_var, dash_var, background_var, width_var = combo
        
        # Skip some combinations to make the total number manageable
        if i % 4 != 0 and len(valid_combinations) > 40:
            continue
        
        # Create filename components
        contour_part = f"{variables[contour_var]}-ContourLevels"
        dash_part = f"{variables[dash_var]}-DashPattern"
        
        if width_var is None:
            width_part = "UniformLineWidth"
        else:
            width_part = f"{variables[width_var]}-LineWidth"
            
        if background_var is None:
            bg_part = ""
        else:
            bg_part = f"_{variables[background_var]}-Background"
            
        # Combine filename components
        filename = f"FlowContour_{contour_part}_{dash_part}_{width_part}{bg_part}.png"
        
        output_path = os.path.join(output_dir, filename)
        
        print(f"Generating flow contour visualisation {i+1}/{len(valid_combinations)}: {filename}")
        
        # Get the appropriate variables
        attribute1 = data[contour_var]
        attribute2 = data[dash_var]
        attribute3 = data[background_var] if background_var else None
        attribute4 = data[width_var] if width_var else None
        
        # Determine appropriate colormap for background
        
        if background_var == 'flow_divergence':
            # attribute3_cmap = plt.cm.get_cmap("RdBu_r")  # Blue for divergence, Red for convergence
            attribute3_cmap = plt.colormaps.get_cmap("bwr")
            attribute3_min = np.nanmin(data['flow_divergence'])
            attribute3_max = np.nanmax(data['flow_divergence'])
            attribute3_norm = TwoSlopeNorm(vmin=-abs(attribute3_max), vcenter=0, vmax=abs(attribute3_max))
        else:
            #normalize background variable
            attribute3_min = np.nanmin(attribute3)
            attribute3_max = np.nanmax(attribute3)
            attribute3_norm = Normalize(vmin=attribute3_min, vmax=attribute3_max)
            if background_var == 'velmag':
                attribute3_cmap = 'PuBu'
            elif background_var == "topography":  # elev_modified
                # attribute3_cmap = plt.cm.get_cmap("BrBG_r")
                attribute3_cmap = plt.colormaps.get_cmap("YlOrBr")
            elif background_var == "bathymetry":
                attribute3_cmap = plt.colormaps.get_cmap("Greys")
            elif background_var == 'temp':
                attribute3_cmap = 'bwr'
            else:
                # attribute3_cmap = plt.cm.get_cmap("viridis")
                attribute3_cmap = plt.colormaps.get_cmap("Greys")  # Greys
        
        # Create title components
        title_parts = [
            f"{variables[contour_var]} (Contours Levels)",
            f"{variables[dash_var]} (Dash Pattern)"
        ]
        
        if width_var is None:
            title_parts.append("Uniform Line Width")
        else:
            title_parts.append(f"{variables[width_var]} (Line Width)")
            
        if background_var:
            title_parts.append(f"{variables[background_var]} (Background)")
            
        plot_title = f"Flow Contour: {', '.join(title_parts)}"  # TODO: to be modified [DONE]
        if skip_title_plotting:
            plot_title = ""
        
        # Call the flow function with the configured parameters
        try:
            flow(
                attribute1=attribute1,
                lon=data['lon'],
                lat=data['lat'],
                attribute2=attribute2,
                attribute3=attribute3,
                attribute4=attribute4,  # New parameter for width
                attribute1_label = f"{variables[contour_var]}",
                attribute2_label = f"{variables[dash_var]}" if dash_var else None,
                attribute3_label=f"{variables[background_var]}" if background_var else None,
                attribute4_label = f"{variables[width_var]}" if width_var else None,
                signed_dist=data['signed_dist'],
                plot_title=plot_title,
                output_image=output_path,
                attribute3_cmap=attribute3_cmap,
                num_contours=5,  # formerly: 4
                auto_scale=True,
                attribute3_norm= attribute3_norm
            )
        except Exception as e:
            print(f"Error generating {filename}: {e}")
            # traceback.format_exc()
            logger.exception(e)

def main():
    """
    Main function to load data and generate all visualisation mappings.
    """
    print("Loading oceanographic data...")
    data = load_sample_data()
    print("Data loaded successfully.")
    
    generate_all_possible_mappings(data, output_dir=output_dir)
    
    print("All visualisation mappings completed successfully!")

if __name__ == "__main__":

    main()
