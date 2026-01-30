import os
import sys
import h5py
import math
from scipy.io import netcdf
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import colors
from matplotlib.colors import colorConverter
from matplotlib.animation import FuncAnimation, writers
from datetime import timedelta
from argparse import ArgumentParser
from glob import glob
import fnmatch

#from code import DecayLine
from colorpals import colour_scales
#from code import TransparentCircles, TransparentEllipses

ccode = "blue"

def time_index_value(tx, _ft):
    # Expect ft to be forward-linear
    ft = _ft
    if isinstance(_ft, xr.DataArray):
        ft = ft.data

    # time difference between first two steps
    f_dt = ft[1] - ft[0]

    # Convert numpy.timedelta64 -> float seconds
    if isinstance(f_dt, np.timedelta64):
        f_dt = float(f_dt / np.timedelta64(1, 's'))
    elif not isinstance(f_dt, (float, int)):
        # maybe it's already a Python timedelta?
        f_dt = f_dt.total_seconds()

    # now f_dt is just a float in seconds
    f_interp = tx / f_dt
    ti = int(math.floor(f_interp))
    return ti


def time_partion_value(tx, _ft):
    ft = _ft
    if isinstance(_ft, xr.DataArray):
        ft = ft.data

    f_dt = ft[1] - ft[0]

    if isinstance(f_dt, np.timedelta64):
        f_dt = float(f_dt / np.timedelta64(1, 's'))
    elif not isinstance(f_dt, (float, int)):
        f_dt = f_dt.total_seconds()

    f_interp = tx / f_dt
    f_t = f_interp - math.floor(f_interp)
    return f_t

if __name__ =='__main__':

# ================================================================================================== Parsing and loading data ================================================================================================== #

    #Parse arguments
    parser = ArgumentParser(description="Plotting UV (2D) flow fields of a given time index with colour maps.")
    
    parser.add_argument("-d", "--filedir", dest="filedir", type=str, default=os.path.dirname(os.path.abspath(__file__)),
                        help="head directory containing all input data and also are the store target for output files")
    
    parser.add_argument("-o", "--outdir", dest="outdir", type=str, default="None", help="head output directory")
    
    parser.add_argument("-t", "--time", dest="time", type=str, default="-1.0",
                        help="timestamp (in seconds) at which to be plotted")
    
    parser.add_argument("-s", "--signed", dest="signed", action='store_true', default=False, help="plot signed version of velocity magnitude")
    
    parser.add_argument("-C", "--ccode", dest="ccode", choices=("red", "green", "blue", "bred", "bgreen", "bblue"), default="blue",
                        help='chosen colour code baseline (default: blue)')
    
    args = parser.parse_args()

    filedir = args.filedir

    outdir = args.outdir
    outdir = eval(outdir) if outdir == "None" else outdir
    if outdir is None:
        outdir = filedir
    if outdir is None:
        outdir = filedir

    timestamp = float(eval(args.time))
    alpha = 0.3
    zorder = 1
    signed_plot = args.signed
    ccode = args.ccode

    # Checking in data for netCDF
    fU_nc = None
    fV_nc = None
    fX_nc = None
    fY_nc = None
    fT_nc = None
    speed_nc = None
    fU_ext_nc = None
    fV_ext_nc = None
    f_velmag_ext_nc = None
    extents_nc = None
    uvel_fpath_nc = glob(os.path.join(filedir, 'metoffice_foam1_amm7_NWS_SSC_hi20220101.nc'))
    vvel_fpath_nc = glob(os.path.join(filedir, 'metoffice_foam1_amm7_NWS_SSC_hi20220101.nc'))

    if len(uvel_fpath_nc) > 0 and os.path.exists(uvel_fpath_nc[0]):
        uvel_fpath_nc = uvel_fpath_nc[0]
        f_u = xr.open_dataset(uvel_fpath_nc, decode_cf=True, engine='netcdf4')
        fT_nc = f_u.variables['time'].data
        fX_nc = f_u.variables['longitude'].data
        fY_nc = f_u.variables['latitude'].data
        extents_nc = (fX_nc.min(), fX_nc.max(), fY_nc.min(), fY_nc.max())
        fU_nc = f_u.variables['uo'].data
        # fU_nc = np.transpose(fU_nc, [0,1,3,2])
        max_u_value = np.maximum(np.abs(fU_nc.min()), np.abs(fU_nc.max()))
        fU_ext_nc = (-max_u_value, +max_u_value)
        f_u.close()
        del f_u

    if len(vvel_fpath_nc) > 0 and os.path.exists(vvel_fpath_nc[0]):
        vvel_fpath_nc = vvel_fpath_nc[0]
        f_v = xr.open_dataset(vvel_fpath_nc, decode_cf=True, engine='netcdf4')
        fV_nc = f_v.variables['vo'].data
        max_v_value = np.maximum(np.abs(fV_nc.min()), np.abs(fV_nc.max()))
        fV_ext_nc = (-max_v_value, +max_v_value)
        f_v.close()
        del f_v

    if fU_nc is not None and fV_nc is not None:
        speed_nc = fU_nc ** 2 + fV_nc ** 2
        speed_nc = np.where(speed_nc > 0, np.sqrt(speed_nc), 0)
        f_velmag_ext_nc = (speed_nc.min(), speed_nc.max())

    # Checking data for HDF5
    fU_h5 = None
    fV_h5 = None
    fX_h5 = None
    fY_h5 = None
    fT_h5 = None
    speed_h5 = None
    fU_ext_h5 = None
    fV_ext_h5 = None
    f_velmag_ext_h5 = None
    extents_h5 = None
    uvel_fpath_h5 = glob(os.path.join(filedir, '*U.h5'))
    vvel_fpath_h5 = glob(os.path.join(filedir, '*V.h5'))
    grid_fpath_h5 = os.path.join(filedir, 'grid.h5')

    if len(uvel_fpath_h5) > 0 and os.path.exists(uvel_fpath_h5[0]):
        uvel_fpath_h5 = uvel_fpath_h5[0]
        f_u = h5py.File(uvel_fpath_h5, "r")
        fU_h5 = f_u['uo'][()]
        max_u_value = np.maximum(np.abs(fU_h5.min()), np.abs(fU_h5.max()))
        fU_ext_h5 = (-max_u_value, +max_u_value)
        f_u.close()
        del f_u

    if len(vvel_fpath_h5) > 0 and os.path.exists(vvel_fpath_h5[0]):
        vvel_fpath_h5 = vvel_fpath_h5[0]
        f_v = h5py.File(vvel_fpath_h5, "r")
        fV_h5 = f_v['vo'][()]
        max_v_value = np.maximum(np.abs(fV_h5.min()), np.abs(fV_h5.max()))
        fV_ext_h5 = (-max_v_value, +max_v_value)
        f_v.close()
        del f_v

    if os.path.exists(grid_fpath_h5):
        f_grid = h5py.File(grid_fpath_h5, "r")
        fX_h5 = f_grid['longitude'][()]
        fY_h5 = f_grid['latitude'][()]
        fT_h5 = f_grid['times'][()]
        extents_h5 = (fX_h5.min(), fX_h5.max(), fY_h5.min(), fY_h5.max())
        f_grid.close()
        del f_grid

    if fU_h5 is not None and fV_h5 is not None:
        speed_h5 = fU_h5 ** 2 + fV_h5 ** 2
        speed_h5 = np.where(speed_h5 > 0, np.sqrt(speed_h5), 0)
        f_velmag_ext_h5 = (speed_h5.min(), speed_h5.max())

    #Loading in data
    fU = None
    fV = None
    fX = None
    fY = None
    fT = None
    speed = None
    fU_ext = None
    fV_ext = None
    f_velmag_ext = None
    extents = None

    if fU_nc is not None and fV_nc is not None:
        fU = fU_nc.squeeze()
        fV = fV_nc.squeeze()
        fX = fX_nc
        fY = fY_nc
        fT = fT_nc
        # speed = speed_nc.squeeze()
        fU_ext = fU_ext_nc
        fV_ext = fV_ext_nc
        # f_velmag_ext = f_velmag_ext_nc
        extents = extents_nc
        speed = fU** 2 + fV ** 2
        speed = np.where(speed > 0, np.sqrt(speed), 0)
        f_velmag_ext = (speed.min(), speed.max())
    elif fU_h5 is not None and fV_h5 is not None:
        fU = fU_h5
        fV = fV_h5
        fX = fX_h5
        fY = fY_h5
        fT = fT_h5
        speed = speed_h5
        fU_ext = fU_ext_h5
        fV_ext = fV_ext_h5
        f_velmag_ext = f_velmag_ext_h5
        extents = extents_h5
        # speed = fU** 2 + fV ** 2
        # speed = np.where(speed > 0, np.sqrt(speed), 0)
        # f_velmag_ext = (speed.min(), speed.max())
    else:
        exit()



    fX_ext = (np.nanmin(fX), np.nanmax(fX))
    fY_ext = (np.nanmin(fY), np.nanmax(fY))
    fT_ext = (np.nanmin(fT), np.nanmax(fT))
    sX = fX_ext[1]-fX_ext[0]
    sY = fY_ext[1] - fY_ext[0]
    sT = fT_ext[1] - fT_ext[0]
    dt = fT[1] - fT[0]

    print(("fU ext.: {}".format(fU_ext)))
    print(("fV ext.: {}".format(fV_ext)))
    print(("f VelMag ext.: {}".format(f_velmag_ext)))
    print("fX - shape: {}; |fX|: {}".format(fX.shape, len(fX)))
    print("fY - shape: {}; |fY|: {}".format(fY.shape, len(fY)))

    #=========================================================================Interpolation=========================================================================#
    
    np.set_printoptions(linewidth=160)
    ns_per_sec = np.timedelta64(1, 's')  # nanoseconds in an sec
    sec_per_day = 86400.0
    
    if timestamp >= 0:
        
        tx = timestamp
        tx = math.fmod(tx, fT[-1])
        ti0 = time_index_value(tx, fT)
        tt = time_partion_value(tx, fT)
        ti1 = 0
        if ti0 < (len(fT)-1):
            ti1 = ti0+1
        else:
            ti1 = 0
        print("time: i_0 = {}; i_1 = {}, t = {}".format(ti0, ti1, tt))

        s0 = fU[ti0] ** 2 + fV[ti0] ** 2
        s0 = np.where(s0 > 0, np.sqrt(s0), 0)
        s0_invalid = np.isclose(s0, 0) & np.isclose(s0,-0)
        s0_min = np.finfo(s0.dtype).eps if s0_invalid.all() else np.min(s0[~s0_invalid])
        s0[s0_invalid] = s0_min

        s1 = fU[ti1] ** 2 + fV[ti1] ** 2
        s1 = np.where(s1 > 0, np.sqrt(s1), 0)
        s1_invalid = np.isclose(s1, 0) & np.isclose(s1,-0)
        s1_min = np.finfo(s1.dtype).eps if s1_invalid.all() else np.min(s1[~s1_invalid])
        s1[s1_invalid] = s1_min

        fu_show = (1.0 - tt) * fU[ti0] + tt * fU[ti1]
        fu_show_alpha = np.abs(fu_show)
        fu_a_min = np.min(fu_show_alpha)
        fu_a_max = np.max(fu_show_alpha)
        # fu_show_alpha = fu_show_alpha / np.linalg.norm(fu_show_alpha, 'fro')
        fu_show_alpha = fu_show_alpha / (fu_a_max-fu_a_min)
        fu_sign = np.sign(fu_show)

        fv_show = (1.0 - tt) * fV[ti0] + tt * fV[ti1]
        fv_show_alpha = np.abs(fv_show)
        fv_a_min = np.min(fv_show_alpha)
        fv_a_max = np.max(fv_show_alpha)
        # fv_show_alpha = fv_show_alpha / np.linalg.norm(fv_show_alpha, 'fro')
        fv_show_alpha = fv_show_alpha / (fv_a_max-fv_a_min)
        fv_sign = np.sign(fv_show)

        fs_show = (1.0 - tt) * s0 + tt * s1
        fs_show_alpha = np.abs(fs_show)
        fs_a_min = np.min(fs_show_alpha)
        fs_a_max = np.max(fs_show_alpha)
        # fs_show_alpha = fs_show_alpha / np.linalg.norm(fs_show_alpha, 'fro')
        fs_show_alpha = fs_show_alpha / (fs_a_max-fs_a_min)
        fsign = fu_sign + fv_sign
        fsign = np.sign(fsign)  # np.where(fsign < 0., -1.0, 1.0)
        if signed_plot:
            fs_show= fs_show * fsign

        print(("fu shape: {}; min: {}, max: {}".format(fu_show.shape, np.min(fu_show), np.max(fu_show))))
        print(("fu_a shape: {}; min: {}, max: {}".format(fu_show_alpha.shape, np.min(fu_show_alpha), np.max(fu_show_alpha))))
        print(("fv shape: {}; min: {}, max: {}".format(fv_show.shape, np.min(fv_show), np.max(fv_show))))
        print(("fv_a shape: {}; min: {}, max: {}".format(fv_show_alpha.shape, np.min(fv_show_alpha), np.max(fv_show_alpha))))
        print(("fs shape: {}; min: {}, max: {}".format(fs_show.shape, np.min(fs_show), np.max(fs_show))))
        print(("fs_a shape: {}; min: {}, max: {}".format(fs_show_alpha.shape, np.min(fs_show_alpha), np.max(fs_show_alpha))))

        # ==== ==== PLOT ==== ============================================================================================================================ #

        # datafig, dataaxis = plt.subplots(1, 1, figsize=(10, 8))
        datafig = plt.figure(figsize=(12, 8))
        #dataaxis = plt.axes()
        dataaxis = datafig.add_axes([0.14, 0.11, 0.75, 0.78])
        dataaxis.set_xlim([extents[0], extents[1]])
        dataaxis.set_ylim([extents[2], extents[3]])

        # Switching to black background
        if ccode in ["bred", "bgreen", "bblue"]:
            dataaxis.set_facecolor("black")
        else:
            # dataaxis.set_facecolor("lightgray")
            dataaxis.set_facecolor("white")

        def register_patch(patch):
            dataaxis.add_patch(patch)

        speed_min = fs_show.min()
        speed_max = fs_show.max()
        levels = np.linspace(speed_min, speed_max, 20)

        csf = dataaxis.contourf(
            fX, fY, fs_show,
            levels=levels,
            cmap='viridis'
        )

        cs_lines = dataaxis.contour(
            fX, fY, fs_show,
            levels=levels,
            colors='black',
            linewidths=0.5
        )
        dataaxis.clabel(cs_lines, inline=True, fontsize=8)


        ax_cbar = datafig.add_axes([0.91, 0.1, 0.02, 0.8])
        cbar = plt.colorbar(csf, cax=ax_cbar)
        cbar.set_label("Flow Speed [m/s]")

        dataaxis.set_title("Simulation - Speed contours at t = %.1f s" % timestamp)
        # ...
        plt.savefig(os.path.join(outdir, "flow_speed_filled_contour.png"), dpi=300)
        plt.show()

        #====================================================================================Title and Save====================================================================================#
        ofname = "hydrodynamics_perceptual_UVW"
        ofname += "_"#+ccode

        dataaxis.set_title("Simulation - NetCDF data - t = %5.1f d" % (timestamp / sec_per_day))
        plt.savefig(os.path.join(outdir, ofname+".png"), dpi=300)




