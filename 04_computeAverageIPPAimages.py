# -*- coding: utf-8 -*-
"""
Combines all the images for a given (TARGET, FILTER, HWP) combination to
produce a single, average image.

Estimates the sky background level of the on-target position at the time of the
on-target observation using a bracketing pair of off-target observations through
the same HWP polaroid rotation value. Subtracts this background level from
each on-target image to produce background free images. Applies an airmass
correction to each image, and combines these final image to produce a background
free, airmass corrected, average image.
"""

# Core imports
import os
import sys
import copy
import warnings

# Import scipy/numpy packages
import numpy as np
from scipy import ndimage

# Import astropy packages
from astropy.table import Table
import astropy.units as u
from astropy.convolution import Gaussian2DKernel
from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from photutils import (make_source_mask,
    MedianBackground, SigmaClip, Background2D)

# Import plotting utilities
from matplotlib import pyplot as plt

# Add the AstroImage class
import astroimage as ai

# Add the header handler to the BaseImage class
from Mimir_header_handler import Mimir_header_handler
ai.reduced.ReducedScience.set_header_handler(Mimir_header_handler)
ai.set_instrument('mimir')

# Grad the assigned keyword for the airmass value
airmassKeyword = ai.reduced.ReducedScience.headerKeywordDict['AIRMASS']

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# This is where the associated PPOL directory is located
PPOL_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_Reduced\\201611'

# The user must specify the *name* of the PPOL meta-groups associated with each
# target-filter combination. This will allow the script to locate the computed
# scaling factors for these images.
metagroupDict = {
    'NGC2023':{
        'H': 'NGC2023_H_meta',
        'Ks': 'NGC2023_K_meta'
    },
    'NGC7023':{
        'H': 'NGC7023_H_meta',
        'Ks': 'NGC7023_K_meta'
    },
    'M78':{
        'H': 'M78_H_meta',
        'Ks': 'M78_K_meta'
    }
}

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611'

# This is the location of the previously generated masks (step 3b)
maskDir         = os.path.join(pyPol_data, 'Masks')
artifactMaskDir = os.path.join(maskDir, 'artifactMasks')

# This is the location where the background free on-target images are stored
bkgFreeDir = os.path.join(pyPol_data, 'bkgFreeHWPimages')

# Build directories for storing final stages of polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(IPPAdir, 0o755)

IPPAdir = os.path.join(polarimetryDir, 'IPPAimages')
if (not os.path.isdir(IPPAdir)):
    os.mkdir(IPPAdir, 0o755)

###############################################################################
# Do some preprocessing for the scaling files... delete all lines with ';'
for target, filtDict in metagroupDict.items():
    for filt, metagroupName in filtDict.items():
        # Construct the path to the metagroup scaling factors
        scalingFactorFile = os.path.join(
            os.path.join(
                PPOL_dir, 'S6_Scaling'
            ), metagroupName + '_meta_scale.dat'
        )

        # Search for that file
        if not os.path.isfile(scalingFactorFile):
            raise ValueError('File {} does not exist'.format(
                os.path.basename(scalingFactorFile))
            )

        # If the file *does* exist, then generate a modified version
        scalingFactorModFile = os.path.join(
            os.path.dirname(scalingFactorFile), metagroupName + '_meta_scale_mod.dat'
        )

        # Test if this file has already been created, and skip it if it has been
        if os.path.isfile(scalingFactorModFile):
            print('File {} already exists... skipping'.format(
                os.path.basename(scalingFactorModFile))
            )
            continue

        # Open the file to read
        scalingFactorFile = open(scalingFactorFile, 'r')

        # open the file to write
        scalingFactorModFile = open(scalingFactorModFile, 'w')

        # Write a header for this modified file
        scalingFactorModFile.write(
            'FILENAME                      HWP   SCALE_FACTOR  XOFF    YOFF\n'
        )
        # Loop over every line in the scalingFactorFile
        for line in scalingFactorFile:
            # test if this line begins with a semicolon
            if line[0] == ';': continue

            scalingFactorModFile.write(line)

        # CLose down both files
        scalingFactorFile.close()
        scalingFactorModFile.close()
###############################################################################

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='ascii.csv')

# Determine which parts of the fileIndex pertain to HEX dither science images
useFiles    = np.logical_and(
    np.logical_and(
        fileIndex['USE'] == 1,
        fileIndex['DITHER_TYPE'] == 'ABBA'
    ), fileIndex['AB'] == 'A'
)
useFileRows = np.where(useFiles)

# Cull the file index to only include (on-target) files selected for use
fileIndex = fileIndex[useFileRows]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
fileIndexByTarget = fileIndex.group_by(['TARGET', 'FILTER'])

# Loop through each group
for group in fileIndexByTarget.groups:
    # Grab the current group information
    thisTarget     = str(np.unique(group['TARGET'].data)[0])
    thisFilter     = str(np.unique(group['FILTER'].data)[0])

    # Update the user on processing status
    print('\nProcessing images for')
    print('Target : {0}'.format(thisTarget))
    print('Filter : {0}'.format(thisFilter))

    # Read in the scaling factors for this target-filter combination
    metagroupName     = metagroupDict[thisTarget][thisFilter]
    scalingFactorFile = os.path.join(
        os.path.join(
            PPOL_dir, 'S6_Scaling'
        ), metagroupName + '_meta_scale_mod.dat'
    )
    scalingFactorTable = Table.read(
        scalingFactorFile,
        format='ascii.basic',
        delimiter=' '
    )

    # Reformat the file name
    reformattedFilenames = [
        '_'.join(f.split('_')[0:2]) + '.fits' for f in scalingFactorTable['FILENAME']
    ]
    scalingFactorTable['FILENAME'] = reformattedFilenames

    # Further divide this group by its constituent HWP values
    indexByPolAng = group.group_by(['IPPA'])

    # Loop over each of the HWP values, as these are independent from
    # eachother and should be treated entirely separately from eachother.
    for IPPAgroup in indexByPolAng.groups:
        # Grab the current HWP information
        thisIPPA = np.unique(IPPAgroup['IPPA'].data)[0]

        # Update the user on processing status
        print('\tIPPA : {0}'.format(thisIPPA))

        # Construct the path of the expected output file
        outFile = '_'.join([thisTarget, thisFilter, str(thisIPPA)])
        outFile = os.path.join(IPPAdir, outFile) + '.fits'

        # Test if this file has already been constructed and either skip
        # this subgroup or break out of the subgroup loop.
        if os.path.isfile(outFile):
            print('\t\tFile {0} already exists...'.format(os.path.basename(outFile)))
            continue

        # Read in all the images for this subgroup
        numFiles       = len(IPPAgroup)
        progressString = '\t\tLoading images: {0:2.0%} complete'
        imgList        = []
        tableFilenames = scalingFactorTable['FILENAME']
        for iFile, filename in enumerate(IPPAgroup['FILENAME']):
            # Update the user on processing status
            print(progressString.format(iFile/numFiles), end='\r')

            # Read in a temporary compy of this image
            bkgFreeFile = os.path.join(bkgFreeDir, filename)
            if os.path.isfile(bkgFreeFile):
                tmpImg = ai.reduced.ReducedScience.read(bkgFreeFile)
            else:
                continue

            # Apply the corresponding user-generated mask to this image
            maskFile = os.path.join(
                artifactMaskDir, filename
            )

            # Check if the mask exists and apply it if it does
            if os.path.isfile(maskFile):
                tmpMask = ai.reduced.ReducedScience.read(maskFile)
                tmpData = tmpImg.data.copy()
                tmpData[np.where(tmpMask.data)] = np.NaN
                tmpImg.data = tmpData

            # Crop the edges of this image to be a 1000x1000 image
            ny, nx = tmpImg.shape
            binningArray = np.array(tmpImg.binning)

            # Compute the amount to crop to get a 1000 x 1000 image
            cy, cx = (ny - 1000, nx - 1000)

            # Compute the crop boundaries and apply them
            lf = np.int(np.round(0.5*cx))
            rt = lf + 1000
            bt = np.int(np.round(0.5*cy))
            tp = bt + 1000

            tmpImg = tmpImg[bt:tp, lf:rt]

            # Transform PPOL coded "bad pixels" to NaN values
            tmpData = tmpImg.data.copy()
            badPix  = np.where(tmpData < -1e4)
            tmpData[badPix] = np.NaN
            tmpImg.data = tmpData

            # Generate the array of uncertainties for this image
            uncertArr = np.sqrt(
                (tmpImg.data + tmpImg.header['SUB_BKG'])/tmpImg.header['ARDNS_01'] +
                tmpImg.header['AGAIN_01']**2
            )
            tmpImg.uncertainty = uncertArr

            # Multiply this image by its scaling coefficient
            filenameMatch = (tableFilenames == filename)
            if np.sum(filenameMatch) != 1:
                print('No unique match for file {} in the scaling factor table'.format(filename))
                continue

            thisImgRow        = np.where(filenameMatch)
            thisScalingFactor = (scalingFactorTable[thisImgRow]['SCALE_FACTOR'])[0]

            # If the scaling factor is set to zero, then simply *skip* the file
            if thisScalingFactor == 0:
                continue

            # otherwise apply the scaling factor
            tmpImg  = thisScalingFactor*tmpImg

            # This scaling factor corrects for airmass, so set airmass to zero
            tmpImg = tmpImg.correct_airmass(0.0)

            # Store this cropped, scaled, masked image image in the imgList
            imgList.append(tmpImg)

        print(progressString.format(1), end='\n\n')

        # Create an image stack for aligning these images and averaging them
        imgStack = ai.utilitywrappers.ImageStack(imgList)

        # Align the images in the stack
        imgStack.align_images_with_wcs(
            subPixel=False,
            padding=np.NaN
        )

        # Do a masked, median-filtered-mean to combine the images
        outImg = imgStack.combine_images()

        # Save the image to disk
        outImg.write(outFile, dtype=np.float64)

print('\nDone computing average images!')
