import os
import glob
import numpy as np
from skimage import measure, morphology
from astropy.table import Table, Column
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

# For debugging
import matplotlib.pyplot as plt

# Add the AstroImage class
import astroimage as ai

# Add the header handler to the BaseImage class
from Mimir_header_handler import Mimir_header_handler
ai.set_instrument('2mass')

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
PPOL_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_Reduced\\201611'

# Build the path to the S3_Asotremtry files
S3_dir = os.path.join(PPOL_dir, 'S3_Astrometry')

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611'

# This is the directory where the 2MASS tiles of the targets have been saved
# Go to "http://hachi.ipac.caltech.edu/" to download 2MASS tiles
TMASSdir  = ".\\2MASSimages"

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

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

    # Dialate the pixel mask
    sigma = 10.0 * gaussian_fwhm_to_sigma    # FWHM = 3.0

    # Build a kernel for detecting pixels above the threshold
    kernel = Gaussian2DKernel(sigma, x_size=21, y_size=21)
    kernel.normalize()
    starPix1 = convolve_fft(
        starPix.astype(float),
        kernel.array
    )

    return starPix1 > 0.005

################################################################################
def find_2MASS_flux(array):
    # Identify which pixels have acceptable "background" levels. Start by
    # grabbing the image statistics
    mean, median, stddev = sigma_clipped_stats(array)

    # Idesntify pixels more than 2-sigma above the background
    fgdThresh = median + 2.0*stddev
    fgdRegion = array > fgdThresh

    # Repeat the classification withiout the *definitely* nebular pixels
    bkgPix = np.logical_not(fgdRegion)
    mean, median, stddev = sigma_clipped_stats(array[bkgPix])
    fgdThresh = median + 2.0*stddev
    fgdRegion = array > fgdThresh

    # Clean the foreground ID region
    all_labels  = measure.label(fgdRegion)
    all_labels1 = morphology.remove_small_objects(all_labels, min_size=50)
    fgdRegion   = all_labels1 > 0

    # Dilate a TOOON to be conservatine...
    sigma = 20.0 * gaussian_fwhm_to_sigma    # FWHM = 3.0

    # Build a kernel for detecting pixels above the threshold
    kernel = Gaussian2DKernel(sigma, x_size=41, y_size=41)
    kernel.normalize()
    fgdRegion= convolve_fft(
        fgdRegion.astype(float),
        kernel.array
    )
    fgdRegion = (fgdRegion > 0.01)

    # Expand a second time to be conservative
    fgdRegion= convolve_fft(
        fgdRegion.astype(float),
        kernel.array
    )
    fgdRegion = (fgdRegion > 0.01)

    # Return the flux-bright pixels to the user
    return fgdRegion
################################################################################

# Read in all the 2MASS images and store them in a dictionary for quick reference
TMASS_Hfiles = np.array(glob.glob(os.path.join(TMASSdir, '*H.fits')))
TMASS_Kfiles = np.array(glob.glob(os.path.join(TMASSdir, '*Ks.fits')))

# Read in the 2MASS images
TMASS_HimgList = [ai.reduced.ReducedScience.read(f) for f in TMASS_Hfiles]
TMASS_KimgList = [ai.reduced.ReducedScience.read(f) for f in TMASS_Kfiles]

# Convert the images to "background masks"
TMASS_HbkgList = []
for img in TMASS_HimgList:
    print('Building background mask for {}'.format(os.path.basename(img.filename)))
    tmp = img.copy()
    tmp.data = np.logical_not(find_2MASS_flux(img.data)).astype(int)
    TMASS_HbkgList.append(tmp)

TMASS_KbkgList = []
for img in TMASS_KimgList:
    print('Building background mask for {}'.format(os.path.basename(img.filename)))
    tmp = img.copy()
    tmp.data = np.logical_not(find_2MASS_flux(img.data)).astype(int)
    TMASS_KbkgList.append(tmp)

# Parse the targets for each file
TMASS_Htargets = [os.path.basename(f).split('_')[0] for f in TMASS_Hfiles]
TMASS_Ktargets = [os.path.basename(f).split('_')[0] for f in TMASS_Kfiles]

# Store these masks in a dictionary
TMASS_HimgDict = dict(zip(
    TMASS_Htargets,
    TMASS_HbkgList
))
TMASS_KimgDict = dict(zip(
    TMASS_Ktargets,
    TMASS_KbkgList
))

TMASS_backgroundRegions = {
    'H': TMASS_HimgDict,
    'Ks': TMASS_KimgDict
}

# Now that the 2MASS files have been read in, it is safe to set the Mimir_header_handler
ai.reduced.ReducedScience.set_header_handler(Mimir_header_handler)

#Loop through each file in the fileList variable
numberOfFiles = len(fileIndex)
bkgLevels = fileIndex['BACKGROUND']
print('{0:3.1%} complete'.format(0), end='\r')
for iRow, row in enumerate(fileIndex):
    # Skip rows with background levels already measured
    if row['BACKGROUND'] > 0: continue

    # Construct the path to the PPOL S3_Astrometry file
    thisFile = os.path.join(S3_dir, row['FILENAME'])

    # Read in this image
    thisImg = ai.reduced.ReducedScience.read(thisFile)

    # Grab the relevant information for this file
    thisTarget     = row['TARGET']
    thisFilter     = row['FILTER']
    thisDitherType = row['DITHER_TYPE']
    thisAB         = row['AB']

    # Determine of a 2MASS tile exists for this target-filter combo
    if thisFilter == 'H':
        has2MASStile = thisTarget in TMASS_Htargets
    if thisFilter == 'Ks':
        has2MASStile = thisTarget in TMASS_Ktargets

    # Attempt to recover a background estimate. If not, then just fill with -1e6
    try:
        # If this is an on-target (A) frame, then use the 2MASS tile to decide which
        # pixels are dim enough for estimating a background level
        if (thisAB == 'A') and has2MASStile:
            # Grab the (precomputed) 2MASS background level
            this2MASSbkg = TMASS_backgroundRegions[thisFilter][thisTarget]

            # Locate pixels in this frame within the 2MASS region
            ny, nx = thisImg.shape
            yy, xx = np.mgrid[0:ny, 0:nx]
            RA, Dec = thisImg.wcs.wcs_pix2world(xx, yy, 0)
            xx2, yy2 = this2MASSbkg.wcs.wcs_world2pix(RA, Dec, 0)
            xx2, yy2 = xx2.round().astype(int), yy2.round().astype(int)

            # Check if these pixls are outside the accepable bounds and trim if necessary
            goodXind = np.where(np.sum(xx2 > 1, axis=0) == ny)[0]
            lf = np.min(goodXind)
            goodXind = np.where(np.sum(xx2 < nx - 2, axis=0) == ny)[0]
            rt = np.max(goodXind) + 1
            goodYind = np.where(np.sum(yy2 > 0, axis=1) == nx)[0]
            bt = np.min(goodYind)
            goodYind = np.where(np.sum(yy2 < ny - 2, axis=1) == nx)[0]
            tp = np.max(goodYind) + 1
            yy2, xx2, = yy2[bt:tp, lf:rt], xx2[bt:tp, lf:rt]

            # Locate which corresponding pixels fall in background regions
            bkgRegion = np.zeros((ny, nx), dtype=bool)
            bkgRegion[bt:tp, lf:rt] = this2MASSbkg.data[yy2, xx2].astype(bool)
            bkgRegion = np.logical_and(bkgRegion, thisImg.data > -1e5)

            # Compute the image background statistics
            _, bkgMedian, _ = sigma_clipped_stats(thisImg.data[bkgRegion])

        else:
            # Locate the non-stellar pixels in this image
            bkgRegion = np.logical_not(find_dim_stars(thisImg.data))
            bkgRegion = np.logical_and(bkgRegion, thisImg.data > -1e5)

            # Compute the image background statistics
            _, bkgMedian, _ = sigma_clipped_stats(thisImg.data[bkgRegion])

    # Something went wrong, so just fill in with a junk value for now. This can
    # be reprocessed later.
    except:
        bkgMedian = -1e6

    # Append that background levels to the list
    bkgLevels[iRow] = bkgMedian

    # Update on progress
    print('{0:3.1%} complete'.format(iRow/numberOfFiles), end='\r')

# Alert the user that everything is complete
print('{0:3.1%} complete'.format(1), end='\n\n')

# Now store those values in the background and write to disk
fileIndex['BACKGROUND'] = bkgLevels

# Then save to disk
print('*************************************')
print('Writing all background levels to disk')
print('*************************************')
fileIndex.write(indexFile, format='csv')

import pdb; pdb.set_trace()

print('Done!')
