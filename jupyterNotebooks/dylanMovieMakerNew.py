"""
Some quick plots for the isolated disk setup.
"""
import numpy as np
import h5py
import glob
from os.path import isfile, isdir
from os import mkdir

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from sphMap import sphMap

#path = '/vera/ptmp/gc/dnelson/public/xeno_diskIC/run_diskIC_wdm/output/'
path = '/u/dnelson/data.xboecker/run/4_galaxy/xeno_diskIC/run_diskIC_wdm/output_1000xRes_cooling_SFR_xenoSN/'

def units():
    """ Load/derive the unit system. """

    # get a snapshot
    snap = glob.glob(path + 'snap*.hdf5')[0]

    with h5py.File(snap,'r') as f:
        attrs = dict(f['Header'].attrs)

    # unit system
    u = {k:v for k,v in attrs.items() if 'Unit' in k}

    u['BoxSize'] = attrs['BoxSize']

    # derived units
    u['UnitTime_in_s']       = u['UnitLength_in_cm'] / u['UnitVelocity_in_cm_per_s']
    u['UnitDensity_in_cgs']  = u['UnitMass_in_g'] / u['UnitLength_in_cm']**3
    u['UnitPressure_in_cgs'] = u['UnitMass_in_g'] / u['UnitLength_in_cm'] / u['UnitTime_in_s']**2
    u['UnitEnergy_in_cgs']   = u['UnitMass_in_g'] * u['UnitLength_in_cm']**2 / u['UnitTime_in_s']**2
    u['UnitTemp_in_cgs']     = u['UnitEnergy_in_cgs'] / u['UnitMass_in_g']

    # helpful conversions
    sec_in_year = 3.155693e7
    Msun_in_g = 1.98892e33
    kpc_in_cm = 3.085680e21

    u['UnitTime_in_Gyr'] = u['UnitTime_in_s'] / sec_in_year / 1e9
    u['UnitMass_in_Msun'] = u['UnitMass_in_g'] / Msun_in_g
    u['UnitLength_in_kpc'] = u['UnitLength_in_cm'] / kpc_in_cm

    u['UnitDensity_in_Msun_kpc3'] = u['UnitDensity_in_cgs'] * (u['UnitMass_in_Msun']/u['UnitLength_in_kpc']**3)

    return u

def numpart_vs_time():
    """ Plot number of gas/stars/DM, versus time. """
    snaps = sorted(glob.glob(path + 'snap*.hdf5'))

    # load
    times = []
    numgas = []
    numdm = []
    numstars = []

    for snap in snaps:
        with h5py.File(snap,'r') as f:
            attrs = dict(f['Header'].attrs)

        times.append(attrs['Time'])
        numgas.append(attrs['NumPart_Total'][0])
        numdm.append(attrs['NumPart_Total'][1])
        numstars.append(attrs['NumPart_Total'][4])

    # convert times to Gyr
    u = units()

    times = np.array(times) * u['UnitTime_in_Gyr']

    # plot
    fig, ax = plt.subplots(figsize=(11,8))

    ax.set_xlabel('Time [ Gyr ]')
    ax.set_ylabel('Number of Gas Cells')
    ax.set_yscale('log')

    ax.plot(times, numgas, '-', lw=2.5, label='Gas')
    ax.plot(times, numstars, '-', lw=2.5, label='Stars')
    ax.plot(times, numdm, '-', lw=2.5, label='DM')

    ax.legend(loc='best')
    fig.savefig('numpart_vs_time.png')
    plt.close(fig)

def visualize_frames():
    """ Create individual images frames, to be combined into a movie, visualizing the gas. 
    Encode with:

    ffmpeg -f image2 -start_number 0 -i frame_%03d.png -vcodec libx264 -pix_fmt yuv420p -crf 19 -an -threads 0 out.mp4

    """
    nbins = 500
    vmm = [4.0, 8.0] # log msun/kpc^2

    size_faceon = [70,130] # kpc, i.e. center of box
    size_edgeon = [85,115] # kpc, i.e. narrow for edge-on

    if not isdir('frames'):
        mkdir('frames')

    # get list of snapshots, and units
    snaps = sorted(glob.glob(path + 'snap*.hdf5'))
    u = units()

    # loop over each snapshot
    for i, snap in enumerate(snaps):
        # output file
        filename = 'frames/frame_%03d.png' % i
        print(filename)

        # frame already exists?
        if isfile(filename):
            continue

        # load
        with h5py.File(snap,'r') as f:
            pos = f['PartType0']['Coordinates'][()]
            mass = f['PartType0']['Masses'][()]
            dens = f['PartType0']['Density'][()]
            volume = mass / dens

        # convert masses from code units to msun, positions to kpc
        pos *= u['UnitLength_in_kpc']
        mass *= u['UnitMass_in_Msun']

        if 0:
            # histogram face-on, and edge-on
            extent = [[0,u['BoxSize']],[0,u['BoxSize']]]

            h2d_faceon, _, _ = np.histogram2d(pos[:,0], pos[:,1], weights=mass, bins=nbins, range=extent)
            h2d_edgeon, _, _ = np.histogram2d(pos[:,0], pos[:,2], weights=mass, bins=nbins, range=extent)

        # new: use sphMap to generate projection images
        cell_radius = (volume * 3.0 / (4*np.pi))**(1.0/3.0) # assume spherical
        size = 2.5 * cell_radius # kpc, factor of 2.5 means slightly larger than cell diameter

        boxCen = [u['BoxSize']/2,u['BoxSize']/2,u['BoxSize']/2]
        boxSizeImg = [u['BoxSize'],u['BoxSize'],u['BoxSize']]

        h2d_faceon = sphMap(pos, size, mass, quant=None, axes=[0,1], boxSizeImg=boxSizeImg, 
                            boxSizeSim=0, boxCen=boxCen, nPixels=[nbins,nbins], ndims=3)

        h2d_edgeon = sphMap(pos, size, mass, quant=None, axes=[2,0], boxSizeImg=boxSizeImg, 
                            boxSizeSim=0, boxCen=boxCen, nPixels=[nbins,nbins], ndims=3)

        # normalize by pixel area: [msun per pixel] -> [msun/kpc^2]
        px_area = (boxSizeImg[0]/nbins)**2 # kpc^2

        h2d_faceon /= px_area
        h2d_edgeon /= px_area

        h2d_faceon[h2d_faceon > 0] = np.log10(h2d_faceon[h2d_faceon > 0])
        h2d_edgeon[h2d_edgeon > 0] = np.log10(h2d_edgeon[h2d_edgeon > 0])

        # plot
        fig, (ax_left, ax_right) = plt.subplots(nrows=1, ncols=2, width_ratios=[0.63,0.37], figsize=(13,8))

        ax_left.set_xlabel('x [kpc]')
        ax_left.set_ylabel('y [kpc]')
        ax_left.set_xlim(size_faceon)
        ax_left.set_ylim(size_faceon)

        extent = [0,u['BoxSize'],0,u['BoxSize']]
        im_left = ax_left.imshow(h2d_faceon, cmap='inferno', extent=extent, aspect=1.0, vmin=vmm[0], vmax=vmm[1])

        ax_right.set_xlabel('x [kpc]')
        ax_right.set_ylabel('z [kpc]')
        ax_right.set_xlim(size_edgeon)
        ax_right.set_ylim(size_faceon) # should match above

        im_right = ax_right.imshow(h2d_edgeon, cmap='inferno', extent=extent, aspect=1.0, vmin=vmm[0], vmax=vmm[1])

        # colorbar and finish plot
        cax = make_axes_locatable(ax_right).append_axes('right', size='10%', pad=0.1)
        cb = plt.colorbar(im_right, cax=cax)
        cb.ax.set_ylabel('Gas Surface Mass Density [ $\\rm{M_\odot kpc^{-2}}$ ]')

        fig.savefig(filename)
        plt.close(fig)
