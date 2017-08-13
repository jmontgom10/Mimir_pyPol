# This script will construct one average background image per waveplate angle,
# per waveband, per night. This assumes that the supersky structure is constant
# throughout the night. This is a reasonable assumption for the actual sky
# background contribution, but  the telescope contribution may change. To check
# if the telescope contribution changes through th night, the frist and last
# off-target "B image" from each night will be displayed to the user for
# examination. along with a residual different image. If the residual imge shows
# significant sky structure, then it will be necessary to identify when during
# the night the telescope background changed.
#

import os
import sys
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from scipy.ndimage.filters import median_filter, gaussian_filter

# Add the AstroImage class
import astroimage as ai

# Add the header handler to the BaseImage class
from Mimir_header_handler import Mimir_header_handler
ai.reduced.ReducedScience.set_header_handler(Mimir_header_handler)
ai.set_instrument('mimir')

# This is the location of all PPOL reduction directory
PPOL_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_reduced\\201611'

# Build the path to the S3_Asotrometry files
S3_dir = os.path.join(PPOL_dir, 'S3_Astrometry')

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\'

# Build the path to the supersky directory
bkgImagesDir = os.path.join(pyPol_data, 'bkgImages')
if (not os.path.isdir(bkgImagesDir)):
    os.mkdir(bkgImagesDir, 0o755)

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

# Determine which parts of the fileIndex pertain to science images
useFiles = np.where(fileIndex['USE'] == 1)

# Cull the file index to only include files selected for use
fileIndex = fileIndex[useFiles]

# Group the fileIndex by...
# 1. FILTER
# 2. Night
# 3. Dither (pattern)
# 4. HWP Angle
# 5. ABBA value

# fileIndexByGroup = fileIndex.group_by(['FILTER', 'Night',
#     'Dither', 'HWP', 'ABBA'])
fileIndexByGroup = fileIndex.group_by(['GROUP_ID', 'IPPA', 'AB'])

# Loop through each grouping
for group in fileIndexByGroup.groups:
    # Check if we're dealing with the A or B positions
    thisABBA = str(np.unique(group['AB'].data)[0])

    # Skip over the A images
    if thisABBA == 'A': continue

    # Grab the current target information
    thisGroupName = str(np.unique(group['OBJECT'].data)[0])
    thisGroupID   = str(np.unique(group['GROUP_ID'].data)[0])
    thisFilter    = str(np.unique(group['FILTER'].data)[0])
    thisIPPA      = str(np.unique(group['IPPA'].data)[0])

    # Test if this target-waveband-HWPang combo was previously processed
    outFile = os.path.join(
        bkgImagesDir,
        '{}_G{}_IPPA{}.fits'.format(thisGroupName, thisGroupID, thisIPPA)
    )

    if os.path.isfile(outFile):
        print('File ' + os.path.basename(outFile) +
            ' already exists... skipping to next group')
        continue

    numImgs = len(group)
    print('\nProcessing {0} images for'.format(numImgs))
    print('\tOBJECT : {0}'.format(thisGroupName))
    print('\tFILTER : {0}'.format(thisFilter))
    print('\tIPPA   : {0}'.format(thisIPPA))

    # Test if this should be continued
    if numImgs == 0:
        print("Well that's odd... it shouldn't be possible to have zero images.")
        continue

    if numImgs == 1:
        print("Only one image found. Masking stars and inpainting")

        # Read in the image
        thisFile = os.path.join(S3_dir, group['FILENAME'].data[0])
        thisImg  = ai.reduced.ReducedScience.read(thisFile)

        # Normalize by the image median
        mean, median, stddev = sigma_clipped_stats(thisImg.data)
        thisImg = thisImg/median

        # Detect a 3-sigma threshold
        threshold = detect_threshold(thisImg.data, snr=3.0)
        sigma     = 3.0 * gaussian_fwhm_to_sigma    # FWHM = 3.0

        # Build a kernel for detecting pixels above the threshold
        kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
        kernel.normalize()

        # Build the segmentation image
        segm    = detect_sources(thisImg.data, threshold, npixels=5, filter_kernel=kernel)
        starPix = (segm.data != 0)

        # Do not include edge effects in expansion by convolution
        starPix[0:20,:  ] = False
        starPix[-21:-1,:] = False

        # Expand the mask by convolution
        sigma  = 8.0 * gaussian_fwhm_to_sigma    # FWHM = 8.0
        kernel = Gaussian2DKernel(sigma, x_size=10, y_size=10)
        kernel.normalize()
        starPix1 = convolve_fft(
            starPix.astype(float),
            kernel
        )

        # Mask any pixels with values greater than 0.04 (which seems to
        # produce a reasonable result.)
        peakValue     = 1/(200*np.pi)
        maskThreshold = 0.06
        starPix11     = (starPix1 > maskThreshold).astype(bool)

        # Inpaint the "star pixels"
        superskyInpainter = ai.utilitywrappers.Inpainter(thisImg)
        superskyImage     = superskyInpainter.inpaint_nans(starPix11)

        # Identify the "bad pixels" and inpaint them
        badPix            = superskyImage.data < 0.50
        superskyInpainter = ai.utilitywrappers.Inpainter(superskyImage)
        superskyImage2    = superskyInpainter.inpaint_nans(badPix)

        # Write the repaired image to disk
        superskyImage2.write(outFile, dtype=np.float32)

    elif numImgs >= 2:
        # Read in all the relevant images for constructing this HWP image
        thisFileList = [os.path.join(S3_dir, f) for f in group['FILENAME']]
        imgList = [ai.reduced.ReducedScience.read(file1) for file1 in thisFileList]

        # Construct a basic supersky image
        # Construct an image stack of the off-target images
        imageStack = ai.utilitywrappers.ImageStack(imgList)

        # Build a supersky image from these off-target images
        superskyImage = imageStack.produce_supersky()
        superskyArray = superskyImage.data.copy()
        badPix        = superskyArray < 0.25
        superskyInpainter = ai.utilitywrappers.Inpainter(superskyImage)
        superskyImage2    = superskyInpainter.inpaint_nans(badPix)

        # The goal of the script is complete. Write the image to disk.
        superskyImage2.write(outFile, dtype=np.float32)

print('Done!')
