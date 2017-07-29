# -*- coding: utf-8 -*-
"""
Combines all the images for a given (TARGET, FILTER, POLPOS) combination to
produce a single, average image.

Estimates the sky background level of the on-target position at the time of the
on-target observation using a bracketing pair of off-target observations through
the same POLPOS polaroid rotation value. Subtracts this background level from
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

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# This is a list of targets for which to process each subgroup (observational
# group... never spanning multiple nights, etc...) instead of combining into a
# single "metagroup" for all observations of that target. The default behavior
# is to go ahead and combine everything into a single, large "metagroup". The
# calibration data should probably not be processed as a metagroup though.
processSubGroupList = ['Taurus_Cal', 'Orion_Cal', 'Cyg_OB2']
processSubGroupList = [t.upper() for t in processSubGroupList]

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201612'

# This is the location of the previously generated masks (step 4)
maskDir = os.path.join(pyPol_data, 'Masks')

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

polAngDir = os.path.join(polarimetryDir, 'polAngImgs')
if (not os.path.isdir(polAngDir)):
    os.mkdir(polAngDir, 0o755)

bkgPlotDir = os.path.join(polAngDir, 'bkgPlots')
if (not os.path.isdir(bkgPlotDir)):
    os.mkdir(bkgPlotDir, 0o755)

# # Setup PRISM detector properties
# read_noise = 13.0 # electrons
# effective_gain = 3.3 # electrons/ADU

#########
### Establish the atmospheric extinction (magnitudes/airmass)
#########
# Following table from Hu (2011)
# Data from Gaomeigu Observational Station
# Passband | K'(lambda) [mag/airmass] | K'' [mag/(color*airmass)]
# U	         0.560 +/- 0.023            0.061 +/- 0.004
# B          0.336 +/- 0.021            0.012 +/- 0.003
# V          0.198 +/- 0.024           -0.015 +/- 0.004
# R          0.142 +/- 0.021           -0.067 +/- 0.005
# I          0.093 +/- 0.020            0.023 +/- 0.006


# Following table from Schmude (1994)
# Data from Texas A & M University Observatory
# Passband | K(lambda) [mag/airmass] | dispersion on K(lambda)
# U	         0.60 +/- 0.05             0.120
# B          0.40 +/- 0.06             0.165
# V          0.26 +/- 0.03             0.084
# R          0.19 +/- 0.03             0.068
# I          0.16 +/- 0.02             0.055

kappa = dict(zip(['U',    'B',    'V',    'R'   ],
                 [0.60,   0.40,   0.26,   0.19  ]))


# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='ascii.csv')

# Determine which parts of the fileIndex pertain to HEX dither science images
useFiles    = np.logical_and(
    fileIndex['USE'] == 1,
    fileIndex['DITHER_TYPE'] == 'ABBA'
)
useFileRows = np.where(useFiles)

# Cull the file index to only include files selected for use
fileIndex = fileIndex[useFileRows]

# Define an approximate pixel scale
pixScale  = 0.39*(u.arcsec/u.pixel)


# TODO: implement a FWHM seeing cut... not yet working because PSF getter seems
# to be malfunctioning in step 2
#
#
# # Loop through each unique GROUP_ID and test for bad seeing conditions.
# groupByID = fileIndex.group_by(['GROUP_ID'])
# for subGroup in groupByID.groups:
#     # Grab the FWHM values for this subGroup
#     thisFWHMs = subGroup['FWHM']*u.pixel
#
#     # Grab the median and standard deviation of the seeing for this subgroup
#     medianSeeing = np.median(thisFWHMs)
#     stdSeeing    = np.std(thisFWHMs)
#
#     # Find bad FWHM values
#     badFWHMs = np.logical_not(np.isfinite(subGroup['FWHM']))
#     badFWHMs = np.logical_or(
#         badFWHMs,
#         thisFWHMs <= 0
#     )
#     badFWHM = np.logical_and(
#         badFWHM,
#         thisFWHMs > 2.0*u.arcsec
#     )
#     import pdb; pdb.set_trace()

# Group the fileIndex by...
# 1. Target
# 2. Waveband
fileIndexByTarget = fileIndex.group_by(['TARGET', 'FILTER'])

# Loop through each group
for group in fileIndexByTarget.groups:
    # Grab the current group information
    thisTarget     = str(np.unique(group['TARGET'].data)[0])
    thisFilter     = str(np.unique(group['FILTER'].data)[0])

    # # Skip the Merope nebula for now... not of primary scientific importance
    # if thisTarget == 'MEROPE': continue

    # Update the user on processing status
    print('\nProcessing images for')
    print('Target : {0}'.format(thisTarget))
    print('Filter : {0}'.format(thisFilter))

    # Grab the atmospheric extinction coefficient for this wavelength
    thisKappa = kappa[thisFilter]

    # Further divide this group by its constituent POLPOS values
    indexByPolAng = group.group_by(['POLPOS'])

    # Loop over each of the polAng values, as these are independent from
    # eachother and should be treated entirely separately from eachother.
    for polAngGroup in indexByPolAng.groups:
        # Grab the current polAng information
        thisPolAng = np.unique(polAngGroup['POLPOS'].data)[0]

        # Update the user on processing status
        print('\tPol Ang : {0}'.format(thisPolAng))

        # For ABBA dithers, we need to compute the background levels on a
        # sub-group basis. If this target has not been selected for subGroup
        # averaging, then simply append the background subtracted images to a
        # cumulative list of images to align and average.

        # Initalize an image list to store all the images for this
        # (target, filter, pol-ang) combination
        imgList = []

        indexByGroupID = polAngGroup.group_by(['GROUP_ID'])
        for subGroup in indexByGroupID.groups:
            # Grab the numae of this subGroup
            thisSubGroup = str(np.unique(subGroup['OBJECT'])[0])

            # if (thisSubGroup != 'NGC2023_R1') and (thisSubGroup != 'NGC2023_R2'): continue

            # Construct the output file name and test if it alread exsits.
            if thisTarget in processSubGroupList:
                outFile = '_'.join([thisTarget, thisSubGroup, str(thisPolAng)])
                outFile = os.path.join(polAngDir, outFile) + '.fits'
            elif thisTarget not in processSubGroupList:
                outFile = '_'.join([thisTarget, thisFilter, str(thisPolAng)])
                outFile = os.path.join(polAngDir, outFile) + '.fits'

            # Test if this file has already been constructed and either skip
            # this subgroup or break out of the subgroup loop.
            if os.path.isfile(outFile):
                print('\t\tFile {0} already exists...'.format(os.path.basename(outFile)))
                if thisTarget in processSubGroupList:
                    continue
                elif thisTarget not in processSubGroupList:
                    break

            # Update the user on the current execution status
            print('\t\tProcessing images for subgroup {0}'.format(thisSubGroup))

            # Initalize lists to store the A and B images.
            AimgList  = []
            BimgList  = []

            # Initalize a list to store the off-target sky background levels
            BbkgList  = []

            # Initilaze lists to store the times of observation
            AdatetimeList = []
            BdatetimeList = []

            # Read in all the images for this subgroup
            progressString = '\t\tNumber of Images : {0}'
            for iFile, filename in enumerate(subGroup['FILENAME']):
                # Update the user on processing status
                print(progressString.format(iFile+1), end='\r')

                # Read in a temporary compy of this image
                tmpImg = ai.reduced.ReducedScience.read(filename)

                # Crop the edges of this image
                ny, nx = tmpImg.shape
                binningArray = np.array(tmpImg.binning)
                # TODO: COMPUTE the proper cropping to get a (1000, 1000) image
                cx, cy = (np.array([16, 32])/binningArray).astype(int)

                tmpImg = tmpImg[cy:ny-cy, cx:nx-cx]

                # Grab the on-off target value for this image
                thisAB = subGroup['AB'][iFile]

                # Place the image in a list and store required background values
                if thisAB == 'B':
                    # Place B images in the BimgList
                    BimgList.append(tmpImg)

                    # Place the median value of this off-target image in list
                    mask = make_source_mask(
                        tmpImg.data, snr=2, npixels=5, dilate_size=11
                        )
                    mean, median, std = sigma_clipped_stats(
                        tmpImg.data, sigma=3.0, mask=mask
                    )
                    BbkgList.append(median)

                    # Place the time of this image in a list of time values
                    BdatetimeList.append(tmpImg.julianDate)

                if thisAB == 'A':
                    # Read in any associated masks and store them.
                    maskFile = os.path.join(maskDir, os.path.basename(filename))

                    # If there is a mask for this file, then apply it!
                    if os.path.isfile(maskFile):
                        # Read in the mask file
                        tmpMask = ai.reduced.ReducedScience.read(maskFile)

                        # Crop the mask to match the shape of the original image
                        tmpMask = tmpMask[cy:ny-cy, cx:nx-cx]

                        # Grab the data to be masked
                        tmpData = tmpImg.data

                        # Mask the data and put it back into the tmpImg
                        maskInds = np.where(tmpMask.data)
                        tmpData[maskInds] = np.NaN
                        tmpImg.data = tmpData

                    # Place B images in the BimgList
                    AimgList.append(tmpImg)

                    # Place the time of this image in a list of time values
                    AdatetimeList.append(tmpImg.julianDate)

            # Create a new line for shell output
            print('')

            # Construct an image stack of the off-target images
            BimageStack = ai.utilitywrappers.ImageStack(BimgList)

            # Build a supersky image from these off-target images
            superskyImage = BimageStack.produce_supersky()

            # Locate regions outside of a 5% deviation
            tmpSuperskyData  = superskyImage.data
            maskedPix  = np.abs(tmpSuperskyData - 1.0)  > 0.05

            # Get rid of the small stuff and expand the big stuff
            maskedPix = ndimage.binary_opening(maskedPix, iterations=2)
            maskedPix = ndimage.binary_closing(maskedPix, iterations=2)
            maskedPix = ndimage.binary_dilation(maskedPix, iterations=4)

            # TODO: Make the box_size and filter_size sensitive to binning.
            binningArray = np.array(superskyImage.binning)
            box_size = tuple((100/binningArray).astype(int))
            filter_size = tuple((10/binningArray).astype(int))

            # Setup the sigma clipping and median background estimators
            sigma_clip = SigmaClip(sigma=3., iters=10)
            bkg_estimator = MedianBackground()

            # Compute a smoothed background image
            bkgData = Background2D(superskyImage.data,
                box_size=box_size, filter_size=filter_size, mask=maskedPix,
                sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)

            # Construct a smoothed supersky image object
            smoothedSuperskyImage = ai.reduced.ReducedScience(
                bkgData.background/bkgData.background_median,
                uncertainty = bkgData.background_rms,
                properties={'unit':u.dimensionless_unscaled}
            )

            # Interpolate background values to A times
            AbkgList = np.interp(
                AdatetimeList,
                BdatetimeList,
                BbkgList,
                left=-1e6,
                right=-1e6
            )

            # Cut out any extrapolated data (and corresponding images)
            goodInds = np.where(AbkgList > -1e5)
            AimgList = np.array(AimgList)[goodInds]
            AdatetimeList = np.array(AdatetimeList)[goodInds]
            AbkgList = AbkgList[goodInds]

            AsubtractedList = []
            # Loop through the on-target images and subtract background values
            for Aimg, Abkg in zip(AimgList, AbkgList):
                # Subtract the interpolated background values from the A images
                tmpImg = Aimg - smoothedSuperskyImage*(Abkg*Aimg.unit)

                # Apply an airmass correction
                tmpImg = tmpImg.correct_airmass(thisKappa)

                # Append the subtracted and masked image to the list.
                AsubtractedList.append(tmpImg)

            # Now that the images have been fully processed, pause to generate
            # a plot to store in the "background plots" folder. These plots
            # constitute a good sanity check on background subtraction.
            plt.plot(BdatetimeList, BbkgList, '-ob')
            plt.scatter(AdatetimeList, AbkgList, marker='o', facecolor='r')
            plt.xlabel('Julian Date')
            plt.ylabel('Background Value [ADU]')
            figName = '_'.join([thisTarget, thisSubGroup, str(thisPolAng)])
            figName = os.path.join(bkgPlotDir, figName) + '.png'
            plt.savefig(figName, dpi=300)
            plt.close('all')

            # Here is where I need to decide if each subgroup image should be
            # computed or if I should just continue with the loop.
            if thisTarget.upper() in processSubGroupList:
                # Construct an image combiner for the A images
                AimgStack = ai.utilitywrappers.ImageStack(AsubtractedList)

                # Align the images
                AimgStack.align_images_with_wcs(
                    subPixel=False,
                    padding=np.NaN
                    )

                # Combine the images
                AoutImg = imgStack.combine_images()

                # Save the image
                AoutImg.write(outFile, dtype=np.float64)
            else:
                # Extend the imgList variable with background corrected images
                imgList.extend(AsubtractedList)


        if len(imgList) > 0:
            # At the exit of the loop, process ALL the files from ALL the groups
            # Construct an image combiner for the A images
            imgStack = ai.utilitywrappers.ImageStack(imgList)

            # Align the images
            imgStack.align_images_with_wcs(
                subPixel=False,
                padding=np.NaN
                )

            # Combine the images
            outImg = imgStack.combine_images()

            # Save the image
            outImg.write(outFile, dtype=np.float64)

print('\nDone computing average images!')
