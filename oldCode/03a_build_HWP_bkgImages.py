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
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from photutils import detect_threshold, detect_sources
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

# Read in the kokopelli mask
from astropy.io import fits
kokopelliHDUlist = fits.open('kokopelliMask.fits')
kokopelliMask = (kokopelliHDUlist[0].data > 0)

################################################################################
# Define a function to locate even dim stars in the image
################################################################################
from scipy.signal import medfilt
def find_dim_stars(array):
    # Perform a (3x3) median filter
    medArr3 = medfilt(array, 3)
    medArr9 = medfilt(array, 9)

    # Compute array statistics
    mean, median, stddev = sigma_clipped_stats(medArr3)

    # Locate pixels with more that 3-sigma deviation from the local median
    starPix  = (medArr3 - medArr9)/stddev > 2

    # Clean up the edge-effects (and kokopelli)
    starPix[0:20,   :]  = False
    starPix[-21:-1, :]  = False
    starPix[:,   0:20]  = False
    starPix[:, -21:-1]  = False
    starPix[kokopelliMask] = False

    # Dialate the pixel mask
    sigma = 4.0 * gaussian_fwhm_to_sigma    # FWHM = 3.0

    # Build a kernel for detecting pixels above the threshold
    kernel = Gaussian2DKernel(sigma, x_size=9, y_size=9)
    kernel.normalize()
    starPix1 = convolve_fft(
        starPix.astype(float),
        kernel.array
    )
    starPix1 = (starPix1 > 0.01)

    # Clean up the edge-effects
    starPix1[0:20,   :]  = False
    starPix1[-21:-1, :]  = False
    starPix1[:,   0:20]  = False
    starPix1[:, -21:-1]  = False

    # Expand a second time to be conservative
    starPix11 = convolve_fft(
        starPix1.astype(float),
        kernel.array
    )

    return starPix11 > 0.01

################################################################################
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
fileIndexByGroup = fileIndex.group_by(['GROUP_ID', 'HWP', 'AB'])

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
    thisHWP       = str(np.unique(group['HWP'].data)[0])

    # Test if this target-waveband-HWPang combo was previously processed
    outFile = os.path.join(
        bkgImagesDir,
        '{}_G{}_HWP{}.fits'.format(thisGroupName, thisGroupID, thisHWP)
    )

    if os.path.isfile(outFile):
        print('File ' + os.path.basename(outFile) +
            ' already exists... skipping to next group')
        continue

    numImgs = len(group)
    print('\nProcessing {0} images for'.format(numImgs))
    print('\tOBJECT : {0}'.format(thisGroupName))
    print('\tFILTER : {0}'.format(thisFilter))
    print('\tHWP    : {0}'.format(thisHWP))

    # Read in all the relevant images and backgrounds for constructing this HWP image
    thisFileList = [os.path.join(S3_dir, f) for f in group['FILENAME']]
    imgList      = [ai.reduced.ReducedScience.read(file1) for file1 in thisFileList]
    bkgList      = [b for b in group['BACKGROUND']]

    # Finn in all the stars (including the dim ones) with NaNs
    cleanImgList = []
    for img, bkg in zip(imgList, bkgList):
        # Locate the pixels inside the very dim (small stars)
        starPix = find_dim_stars(img.data)

        # Locate the pixels with counts below -1e5
        badPix = img.data < -1e5

        # Build the combined mask
        maskPix = np.logical_or(starPix, badPix)

        # Divide by background level and fill the star pixels with nans
        cleanImg    = img.copy()
        cleanArray  = img.data.copy()
        cleanArray /= bkg
        cleanArray[maskPix] = np.nan
        cleanImg.data = cleanArray

        # Place the nan-filled array in the cleanImgLIst
        cleanImgList.append(cleanImg)

    # Test if this should be continued
    if numImgs == 0:
        print("Well that's odd... it shouldn't be possible to have zero images.")
        import pdb; pdb.set_trace()
        continue

    if numImgs == 1:
        print("Only one image found. Masking stars and inpainting")

        # Inpaint the "star pixels"
        superskyInpainter = ai.utilitywrappers.Inpainter(cleanImgList[0])
        superskyImage     = superskyInpainter.inpaint_nans()

    elif numImgs >= 2:

        # Construct an image stack of the off-target images
        imageStack = ai.utilitywrappers.ImageStack(cleanImgList)

        # Build a supersky image from these off-target images
        superskyImage = imageStack.produce_supersky()

    # Identify the "bad pixels" and inpaint them
    badPix            = superskyImage.data < 0.50
    superskyInpainter = ai.utilitywrappers.Inpainter(superskyImage)
    superskyImage2    = superskyInpainter.inpaint_nans(badPix)

    # Should I compute and force one more normalization by the median?
    # For now, yes...
    _, median, _ = sigma_clipped_stats(superskyImage2.data)
    superskyImage2 = superskyImage2/median

    # Write the repaired image to disk
    superskyImage2.write(outFile, dtype=np.float32)

print('Done!')
