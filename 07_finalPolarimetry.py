# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 09:34:43 2015

@author: jordan
"""

import os
import sys
import numpy as np
from astropy.table import Table as Table
import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
import image_tools
from AstroImage import AstroImage

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data'

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

polAngDir = os.path.join(polarimetryDir, 'polAngImgs')
if (not os.path.isdir(polAngDir)):
    os.mkdir(polAngDir, 0o755)

stokesDir = os.path.join(polarimetryDir, 'stokesImgs')
if (not os.path.isdir(stokesDir)):
    os.mkdir(stokesDir, 0o755)

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

print('\nReading calibration constants from disk')
calTableFile = os.path.join(pyPol_data, 'calData.csv')
calTable = Table.read(calTableFile, format='csv')

# Determine which parts of the fileIndex pertain to science images
useFiles = np.logical_and((fileIndex['Use'] == 1), (fileIndex['Dither'] == 'ABBA'))

# Cull the file index to only include files selected for use
fileIndex = fileIndex[np.where(useFiles)]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither'])

# Define any required conversion constants
rad2deg = (180.0/np.pi)
deg2rad = (np.pi/180.0)
# wavebands = ('R', 'V')
# deltaPA = (16.0*deg2rad, 19.0*deg2rad) #degrees converted into radians
# deltaPA = dict(zip(wavebands, deltaPA))

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisRow      = np.where(calTable['Waveband'] == thisWaveband)
    numImgs      = len(group)
    print('\nProcessing images for'.format(numImgs))
    print('\tTarget   : {0}'.format(thisTarget))
    print('\tWaveband : {0}'.format(thisWaveband))

    # Read in the polAng image
    fileCheck  = True
    polAngImgs = []
    polAngs    = []
    for polAng in range(0,601,200):
        fileName = os.path.join(polAngDir,
            '_'.join([thisTarget, thisWaveband, str(polAng)]) + '.fits')

        # If the file exists, then read it into a dictionary
        if os.path.isfile(fileName):
            polAngs.append(polAng)
            polAngImgs.append(AstroImage(fileName))
        else:
            fileCheck = False

    # Check that all the necessarry files are present
    if not fileCheck:
        print("\tSome required Polaroid Angle files are missing.")
        continue

    # Use the "align_stack" method to align the newly created image list
    print('\nAligning images\n')
    polAngImgs = image_tools.align_images(polAngImgs,
        mode='cross_correlate', subPixel=True, padding=np.nan)

    # Convert the list into a dictionary
    polAngImgs = dict(zip(polAngs, polAngImgs)) #aligned

    #**********************************************************************
    # Stokes I
    #**********************************************************************
    # Average the images to get stokes I
    stokesI = image_tools.combine_images([polAngImgs[0],
                                          polAngImgs[200],
                                          polAngImgs[400],
                                          polAngImgs[600]])
    stokesI = 2 * stokesI

    # Perform astrometry to apply to the headers of all the other images...
    stokesI, success = image_tools.astrometry(stokesI)

    # Check if astrometry solution was successful
    if not success: pdb.set_trace()

    #**********************************************************************
    # Stokes Q
    #**********************************************************************
    # Subtract the images to get stokes Q
    A = polAngImgs[0] - polAngImgs[400]
    B = polAngImgs[0] + polAngImgs[400]

    # Divide the difference images
    stokesQ = A/B

    # Update the header to include the new astrometry
    stokesQ.header = stokesI.header

    # Mask out masked values
    Qmask = np.logical_or(polAngImgs[0].arr   == 0,
                          polAngImgs[400].arr == 0)

    #**********************************************************************
    # Stokes U
    #**********************************************************************
    # Subtact the images to get stokes Q
    A = polAngImgs[200] - polAngImgs[600]
    B = polAngImgs[200] + polAngImgs[600]

    # Divide difference images
    stokesU = A/B

    # Update the header to include the new astrometry
    stokesU.header = stokesI.header

    # Mask out zero values
    Umask   = np.logical_or(polAngImgs[200].arr == 0,
                            polAngImgs[600].arr == 0)

    #**********************************************************************
    # Calibration
    #**********************************************************************
    # Construct images to hold PE, sig_PE, deltaPA, and sig_deltePA
    PE       = stokesI.copy()
    PE.arr   = calTable[thisRow]['PE'].data[0]*np.ones_like(stokesI.arr)
    PE.sigma = calTable[thisRow]['s_PE'].data[0]*np.ones_like(stokesI.arr)

    # Grab the calibration data
    PAsign  = calTable[thisRow]['PAsign'].data[0]
    deltaPA = calTable[thisRow]['dPA'].data[0]
    deltaPArad = np.deg2rad(deltaPA)
    s_dPA   = calTable[thisRow]['s_dPA'].data[0]

    # Normalize the U and Q values by the polarization efficiency
    # and correct for the instrumental rotation direction
    stokesQ = 1.0    * stokesQ / PE
    stokesU = PAsign * stokesU / PE

    # Store the deltaPA values in the stokesQ and stokesQ headers
    stokesQ.header['DELTAPA'] = deltaPA
    stokesU.header['DELTAPA'] = deltaPA
    stokesQ.header['S_DPA'] = s_dPA
    stokesU.header['S_DPA'] = s_dPA

    #**********************************************************************
    # Build the polarization maps
    #**********************************************************************
    Pmap, PAmap = image_tools.build_pol_maps(stokesQ, stokesU)

    #**********************************************************************
    # Final masking and writing to disk
    #**********************************************************************
    # Generate a complete mask of all bad elements
    fullMask = np.logical_or(Qmask, Umask)

    stokesU.arr[np.where(Umask)]  = np.NaN
    stokesQ.arr[np.where(Qmask)]  = np.NaN
    Pmap.arr[np.where(fullMask)]  = np.NaN
    PAmap.arr[np.where(fullMask)] = np.NaN

    # Generate the output filenames
    Ifile = os.path.join(stokesDir,
        '_'.join([thisTarget, thisWaveband, 'I']) + '.fits')
    Ufile = os.path.join(stokesDir,
        '_'.join([thisTarget, thisWaveband, 'U']) + '.fits')
    Qfile = os.path.join(stokesDir,
        '_'.join([thisTarget, thisWaveband, 'Q']) + '.fits')
    Pfile = os.path.join(polarimetryDir,
        '_'.join([thisTarget, thisWaveband, 'P']) + '.fits')
    PAfile = os.path.join(polarimetryDir,
        '_'.join([thisTarget, thisWaveband, 'PA']) + '.fits')

    # Write to disk
    stokesI.write(Ifile)
    stokesU.write(Ufile)
    stokesQ.write(Qfile)
    Pmap.write(Pfile)
    PAmap.write(PAfile)
