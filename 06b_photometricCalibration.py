# -*- coding: utf-8 -*-
"""
Calibrates the Stokes I images to Jy units using the AstroImage classes
"""
# Core imports
import os
import copy
import sys
from datetime import datetime

# Scipy/numpy imports
import numpy as np

# Astropy imports
from astropy.table import Table
import astropy.units as u
from astropy.stats import sigma_clipped_stats

# Import astroimage
import astroimage as ai
ai.set_instrument('Mimir')

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\'

# These are the directories where polarimetry data are stored
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
stokesDir      = os.path.join(polarimetryDir, 'stokesImages')
if (not os.path.isdir(stokesDir)):
    os.mkdir(stokesDir, 0o755)

# Specify which (target, filter) pairs to process
targetFilterDict = {
    'NGC2023':['H', 'Ks'],
    'NGC7023':['H', 'Ks'],
    'M78':['H', 'Ks']
}

# Initalize a dictionary to store all the IPPA images for this target
# Loop through each target-filter pairing
for thisTarget, filters in targetFilterDict.items():
    # Quickly loop through filters and check if this target has already been done
    stokesIdict = {}
    for thisFilter in filters:
        # Construct the expected output names filenames for this
        # (target, filter) pair
        stokesIfilename = '_'.join([thisTarget, thisFilter, 'I']) + '.fits'
        stokesIfilename = os.path.join(stokesDir, stokesIfilename)

        # Store the output Stokes filenames in the list and dictionary
        stokesIdict[thisFilter] = ai.reduced.ReducedScience.read(
            stokesIfilename
        )

    # Construct a calibration object
    photCalibrator = ai.utilitywrappers.PhotometryCalibrator(
        stokesIdict
    )

    # Run the calibraion method
    calImgDict = photCalibrator.calibrate_photometry()

    # Write the calibrated images to disk
    for key, img in calImgDict.items():
        # Determine if this is an intensity image
        keyParts = key.split('_')
        if len(keyParts) > 1:
            filename  = os.path.join(stokesDir, '_'.join([thisTarget, keyParts[0], 'I', 'cal']) + '.fits')
        else:
            filename  = os.path.join(stokesDir, '_'.join([thisTarget, key, 'cal']) + '.fits')

        img.write(filename, clobber=True)

print('Done!')
