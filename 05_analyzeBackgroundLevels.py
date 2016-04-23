import os
import sys
import subprocess
import datetime
import numpy as np
from astropy.io import ascii
from astropy.table import Table, Column, vstack
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from astropy.coordinates import SkyCoord, ICRS
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from scipy.stats import norm, f
from scipy.odr import *
from scipy.optimize import minimize
from scipy.ndimage.filters import median_filter, gaussian_filter1d
from photutils import detect_sources, Background

# For debugging
import matplotlib.pyplot as plt
import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage
import image_tools

# This script will read in the background level estimated for each on-target
# image in the previous step. The background level in dimmest parts of the
# on-target image will be directly computed, and the residual between the direct
# estimate and the interpolation will be stored. The distribution of these
# residual will be used to estimate which interpolated background levels can be
# trusted.

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# This is the location of all PPOL reduction directory
PPOL_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_reduced'

# Build the path to the S3_Asotremtry files
S3dir = os.path.join(PPOL_dir, 'S3_Astrometry')

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_data'

# This is the directory where the 2MASS tiles of the targets have been saved
# Go to "http://hachi.ipac.caltech.edu/" to download 2MASS tiles
TMASSdir  = "C:\\Users\\Jordan\\Libraries\\python\\Mimir_pyPol\\2MASSimages"

# Setup new directory for background subtracted data
bkgSubDir = os.path.join(pyPol_data, 'bkgSubtracted')
if (not os.path.isdir(bkgSubDir)):
    os.mkdir(bkgSubDir, 0o755)

# Read in Kokopelli mask generated in previous step
kokopelliMask = (AstroImage('kokopelliMask.fits').arr != 0)

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

# Grab the file basenames for later use
fileIndexFileNames = np.array([os.path.basename(file1)
    for file1 in fileIndex['Filename'].data])

# Modify the fileIndex to include rejections by residual value
if 'Background Cut' not in fileIndex.keys():
    fileIndex.add_column(Column(name='Background Cut',
                                data = np.repeat(0, len(fileIndex))))

# Determine which parts of the fileIndex pertain to science images
useFiles = np.where(np.logical_and(fileIndex['Use'].data == 1,
                                   fileIndex['Background'].data >= 0))
skipFiles = np.where(np.logical_or(fileIndex['Use'].data == 0,
                                   fileIndex['Background'].data < 0))

# Cull the file index to only include files selected for use
fileIndex1 = fileIndex[useFiles]
fileIndex2 = fileIndex[skipFiles]

# Group files by target and waveband
groupFileIndex = fileIndex1.group_by(['PPOL Name'])

allFileList     = []
allResidualList = []
# Loop through all the usable images and comute their residuals
for group in groupFileIndex.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisPPOLname = str(np.unique(group['PPOL Name'].data)[0])

    # if thisPPOLname != 'NGC2023_H3': continue

    print('\nProcessing images for')
    print('\tPPOL Group : {0}'.format(thisPPOLname))
    print('')

    # Read in the 2MASS image
    TMASSfile = os.path.join(TMASSdir, '_'.join([thisTarget, thisWaveband]) + '.fits')
    TMASSimg  = AstroImage(TMASSfile)
    TMASSwcs  = WCS(TMASSimg.header)

    # Estimate the "nebula free" level
    mean, median, stddev = sigma_clipped_stats(TMASSimg.arr.flatten())
    bkgThresh = median - 0.5*stddev

    # Find the "nebula free" pixels
    bkgRegion = TMASSimg.arr < bkgThresh
    neighborCount = np.zeros_like(bkgRegion, dtype=int)
    for dx in range(-1,2):
        for dy in range(-1,2):
            neighborCount += np.roll(np.roll(bkgRegion, dy, axis = 0), dx, axis = 1)

    # Find pixels with at least 3 neighbors (other than self)
    bkgRegion = neighborCount > 4

    groupFileList     = []
    groupResidualList = []
    for file1, interpBkg in zip(group['Filename'].data, group['Background'].data):
        # Read in this image.
        img = AstroImage(file1)

        # See which pixels in this image map to background pixels
        ny, nx = img.arr.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        wcs = WCS(img.header)
        RAs, Decs = wcs.wcs_pix2world(xx, yy, 0)
        Tx, Ty    = TMASSwcs.wcs_world2pix(RAs, Decs, 0)
        Tx, Ty    = (Tx.round()).astype(int), (Ty.round()).astype(int)

        # Grab the value of the TMASS background mask for each pixel
        MimirBkgRegion = bkgRegion[Ty, Tx]

        # Get the indices of the background pixel
        bkgInds  = np.where(MimirBkgRegion)
        bkgVals  = img.arr[bkgInds]

        # Compute the direct estimate of background level
        mean, median, stddev = sigma_clipped_stats(bkgVals)

        # Compute the residual level and store it in the list
        thisResidual = mean - interpBkg
        groupFileList.append(os.path.basename(file1))
        groupResidualList.append(thisResidual)

    # Place this residual list in the final total residual list
    allFileList.extend(groupFileList)
    allResidualList.extend(groupResidualList)

    # Convert the lists to arrays
    groupFileList     = np.array(groupFileList)
    groupResidualList = np.array(groupResidualList)

    # Check for outliers and mark residuals 5-sigma outside this group's median
    mean, median, stddev = sigma_clipped_stats(groupResidualList)
    residMin, residMax = mean - 5*stddev, mean + 5*stddev
    badInds = np.where(np.logical_or(groupResidualList < residMin,
                                     groupResidualList > residMax))

    # If some of these residuals are more than 5-sigma from the group mean, then
    # mark them as bad background levels in the file index.
    if len(badInds[0]) > 0:
        # Select the file names of the bad backgrounds
        badFiles = groupFileList[badInds]
        # Grab the indices of these files in the fileIndex and mark them as bad
        fileIndexInds = np.array([np.where(fileIndexFileNames == file1)[0][0]
            for file1 in badFiles])
        fileIndex['Background Cut'][fileIndexInds] = 1

# Convert the lists to arrays
allFileList     = np.array(allFileList)
allResidualList = np.array(allResidualList)

# Now that we have the residuals for each group, plot them up as histograms
# # Start by parsing out the residuals for each group
# Now create a plot with all groups clumpped together
fig2 = plt.figure()
ax2  = fig2.add_subplot(1,1,1)
ax2.hist(allResidualList, 10, normed=1, histtype='stepfilled', stacked=True)
plt.xlabel('Residual Counts')
plt.ylabel('Fraction of Fields')

# Prepare some statistical comments
xmin, xmax = ax2.get_xlim()
ymin, ymax = ax2.get_ylim()
mean, median, stddev = sigma_clipped_stats(allResidualList)

# Mark the mean
ax2.axvline(mean, color='k', linewidth=2.0)
ax2.text(mean+0.02*(xmax-xmin), 0.95*ymax, 'mean', rotation='vertical')

# Mark the median
ax2.axvline(median, color='k', linewidth=2.0)
ax2.text(median-0.04*(xmax-xmin), 0.95*ymax, 'median', rotation='vertical')

# Mark the 3-sigma upper and lower limits
ax2.axvline(median - 5*stddev, color='k', linewidth=2.0)
ax2.axvline(median + 5*stddev, color='k', linewidth=2.0)

# Prepare the limits of the acceptable residual range
residMin, residMax = mean - 5*stddev, mean + 5*stddev

# Find any background levels that are outside the 5-sigma limits
badInds = np.where(np.logical_or(allResidualList < residMin,
                                 allResidualList > residMax))

# If some of these residuals are more than 5-sigma from the group mean, then
# mark them as bad background levels in the file index.
if len(badInds[0]) > 0:
    # Select the file names of the bad backgrounds
    badFiles = allFileList[badInds]
    # Grab the indices of these files in the fileIndex and mark them as bad
    fileIndexInds = np.array([np.where(fileIndexFileNames == file1)[0][0]
        for file1 in badFiles])
    fileIndex['Background Cut'][fileIndexInds] = 1

# Then save to disk
print('*************************************')
print('Writing all background levels to disk')
print('*************************************')
pdb.set_trace()
fileIndex.write(indexFile, format='csv')

print('Done!')
