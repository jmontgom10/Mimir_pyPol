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
import glob
import warnings
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.modeling import models, fitting
from photutils import detect_threshold, detect_sources
from scipy.ndimage.filters import median_filter, gaussian_filter
from skimage import measure, morphology
from photutils import Background2D

# Add the AstroImage class
import astroimage as ai

# Add the header handler to the BaseImage class
from Mimir_header_handler import Mimir_header_handler

# This is the location of all PPOL reduction directory
PPOL_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_reduced\\201611'

# Build the path to the S3_Asotrometry files
S3_dir = os.path.join(PPOL_dir, 'S3_Astrometry')

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\'

# This is the location of the (pre-computed) star masks for the B-images
maskDir = os.path.join(pyPol_data, 'Masks')
starMaskDir = os.path.join(maskDir, 'starMasks')

# Build the path to the supersky directory
bkgFreeImagesDir = os.path.join(pyPol_data, 'bkgFreeHWPimages')
if (not os.path.isdir(bkgFreeImagesDir)):
    os.mkdir(bkgFreeImagesDir, 0o755)

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

# Read in the kokopelli mask
from astropy.io import fits
kokopelliHDUlist = fits.open('kokopelliMask.fits')
kokopelliMask = (kokopelliHDUlist[0].data > 0)

# Read in the 2MASS masks
TMASSdir  = ".\\2MASSimages"

# Set the instrument to 2MASS
ai.set_instrument('2MASS')

# Read in all the 2MASS images and store them in a dictionary for quick reference
TMASS_Hfiles = np.array(glob.glob(os.path.join(TMASSdir, '*H_mask.fits')))
TMASS_Kfiles = np.array(glob.glob(os.path.join(TMASSdir, '*Ks_mask.fits')))

# Read in the 2MASS images
TMASS_HmaskList = [ai.reduced.ReducedScience.read(f) for f in TMASS_Hfiles]
TMASS_KmaskList = [ai.reduced.ReducedScience.read(f) for f in TMASS_Kfiles]

# Parse the targets for each file
TMASS_Htargets = [os.path.basename(f).split('_')[0] for f in TMASS_Hfiles]
TMASS_Ktargets = [os.path.basename(f).split('_')[0] for f in TMASS_Kfiles]

# Store these masks in a dictionary
TMASS_HimgDict = dict(zip(
    TMASS_Htargets,
    TMASS_HmaskList
))
TMASS_KimgDict = dict(zip(
    TMASS_Ktargets,
    TMASS_KmaskList
))

TMASS_masks = {
    'H': TMASS_HimgDict,
    'Ks': TMASS_KimgDict
}

# Set the instrument to Mimir
ai.reduced.ReducedScience.set_header_handler(Mimir_header_handler)
ai.set_instrument('Mimir')

# Read in the flat images
flatDir  = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\BDP_Data\\201611\\10_Flat_Field_Images'
flatImgs = np.array([
    ai.reduced.ReducedScience.read(f)
    for f in glob.glob(os.path.join(flatDir, '*.fits'))
])

# Get the flat filter from the flat list
flatFilters = np.array([
    flat.filter
    for flat in flatImgs
])

# Get the HWPs fromm the flat list
HWPstepList    = np.array([
    0, 33, 67, 100,
    133, 167, 200, 233,
    267, 300, 333, 367,
    400, 433, 467, 500
])
HWPlist         = np.arange(16, dtype=int) + 1
IPPAlist        = np.array(4*[0, 45, 90, 135])
HWPstep_to_HWP  = dict(zip(HWPstepList, HWPlist))
flatHWPs        = np.array([
    HWPstep_to_HWP[flat.header['HWP']]
    for flat in flatImgs
])

# Define a quick mode function
def mode(array):
    """An estimate of the statistical mode of this image"""
    # SUPER fast and sloppy mode estimate:
    array1 = array[np.where(np.isfinite(array))]
    mean, median, std = sigma_clipped_stats(array1)
    quickModeEst = 3*median - 2*mean

    # Compute an approximately 3-sigma range about this
    modeRegion = quickModeEst + std*np.array([-1.5, +1.5])

    # Now compute the number of bins to generate in this range
    numBins = np.int(np.ceil(0.2*(np.max(modeRegion) - np.min(modeRegion))))
    bins    = np.linspace(modeRegion[0], modeRegion[1], numBins)

    # Loop through larger and larger binning until find unique solution
    foundMode = False
    while not foundMode:
        # Generate a histogram of the flat field
        hist, flatBins = np.histogram(array1, bins=bins)

        # Locate the histogram maximum
        maxInds = (np.where(hist == np.max(hist)))[0]
        if maxInds.size == 1:
            # Grab the index of the maximum value and shrink
            maxInd = maxInds[0]
            foundMode = True
        else:
            # Shrink the NUMBER of bins to help find a unqiue maximum
            numBins *= 0.9
            bins     = np.linspace(modeRegion[0], modeRegion[1], numBins)

    # Estimate flatMode from histogram maximum
    flatMode = np.mean(flatBins[maxInd:maxInd+2])

    return flatMode

# Write a quiick row-by-row flattener
def row_by_row_flatten(array, mask):
    """Flattens residual row-by-row variation in on-target images"""
    # Mask the array
    tmpArr = array.copy().astype(float)
    tmpArr[np.where(mask)] = np.NaN

    # Compute the median of the unmasked region
    arrayMedian = np.nanmedian(tmpArr)

    # Compute the median of each row
    rowMedians  = np.nanmedian(tmpArr, axis = 1)

    # Compute the difference between each row median and array median
    medianDiffs = np.array([rowMedians - arrayMedian]).T

    # Expand the row differences to match the shape of the original array
    ny, nx    = array.shape
    diffArray = np.tile(medianDiffs, (1, nx))

    # Subtract the difference from the input array
    outputArr = array - diffArray

    return outputArr

################################################################################
# Determine which parts of the fileIndex pertain to science images
useFiles = np.where(fileIndex['USE'] == 1)

# Cull the file index to only include files selected for use
fileIndex = fileIndex[useFiles]

# Group the fileIndex by...
# 1. GROUP_ID
# 2. HWP

# fileIndexByGroup = fileIndex.group_by(['FILTER', 'Night',
#     'Dither', 'HWP', 'ABBA'])
fileIndexByGroup = fileIndex.group_by(['GROUP_ID', 'HWP'])

# Loop through each grouping
for group in fileIndexByGroup.groups:
    # Grab the current target information
    thisGroupName = str(np.unique(group['OBJECT'].data)[0])
    thisTarget    = str(np.unique(group['TARGET'].data)[0])
    thisGroupID   = str(np.unique(group['GROUP_ID'].data)[0])
    thisFilter    = str(np.unique(group['FILTER'].data)[0])
    thisHWP       = str(np.unique(group['HWP'].data)[0])

    # Figure out which flat image to use
    thisFlatInd = np.where(
        np.logical_and(
            flatFilters == thisFilter,
            flatHWPs == int(thisHWP)
        )
    )
    thisFlat = (flatImgs[thisFlatInd])[0]

    # Find the 2MASS mask for this image
    this2MASSmask = TMASS_masks[thisFilter][thisTarget]

    numImgs = len(group)
    print('\nProcessing {0} images for'.format(numImgs))
    print('\tOBJECT : {0}'.format(thisGroupName))
    print('\tFILTER : {0}'.format(thisFilter))
    print('\tHWP    : {0}'.format(thisHWP))

    Ainds  = np.where(group['AB'] == 'A')
    Binds  = np.where(group['AB'] == 'B')
    Afiles = group[Ainds]['FILENAME']
    BimgFiles  = [os.path.join(S3_dir, f) for f in group[Binds]['FILENAME']]
    BmaskFiles = [os.path.join(starMaskDir, f) for f in group[Binds]['FILENAME']]

    # Catch the case where there are no B images to use (skip it!)
    numBfiles = len(BimgFiles)
    if (numBfiles == 0): continue

    # Read in the on-target images
    AimgFiles = [os.path.join(S3_dir, f) for f in Afiles if os.path.isfile(os.path.join(S3_dir, f))]
    AimgFiles = [os.path.join(S3_dir, f) for f in Afiles if os.path.isfile(os.path.join(S3_dir, f))]
    AoutFiles = [os.path.join(bkgFreeImagesDir, f) for f in Afiles]
    AoutExist = [os.path.isfile(f) for f in AoutFiles]

    # Check if any expected input files actually exist
    if len(AimgFiles) == 0: continue

    # Check if *both* of these images have already been processed
    if np.sum(AoutExist) == len(Afiles): continue

    # Read in the on-target frames
    Aimgs     = [ai.reduced.ReducedScience.read(f) for f in AimgFiles]

    # Quickly read in both B images
    Bimgs  = [ai.reduced.ReducedScience.read(f) for f in BimgFiles]
    Btimes = np.array([img.julianDate for img in Bimgs])
    B0ind  = Btimes.argmin()
    B1ind  = Btimes.argmax()

    # Read in both masks and create a combined mask
    Bmasks = [ai.reduced.ReducedScience.read(f) for f in BmaskFiles]
    combined_Bmask = False
    for Bmask in Bmasks:
        combined_Bmask = np.logical_or(combined_Bmask, Bmask.data.astype(bool))

    unmasked_Binds = np.where(np.logical_not(combined_Bmask))

    # In the case of a single off-target frame, just mask the stars and move on
    if (numBfiles == 1):
        Bimg = Bimgs[0].copy()
        tmpData = Bimg.data.copy()
        maskInds = np.where(Bmasks[0].data)
        tmpData[maskInds] = np.NaN
        Bimg.data = tmpData
        Bimgs = [Bimg]

    elif (numBfiles == 2):
        ########################################################################
        # Use information from the dithered off-target frames to fill in pixels
        # within stellar PSFs
        ########################################################################
        # Approximate solution for building complete "off-target background"
        # is to subtract the offTargetDifference from the second Bimgs image
        # then compute the stacked median image.
        # Compute the difference between the two off-target frames
        thermalFree_offTarget = Bimgs[B1ind] - Bimgs[B0ind]
        offTargetDifference   = mode(thermalFree_offTarget.data[unmasked_Binds])

        Bimg1  = Bimgs[B1ind] - (offTargetDifference*Bimgs[B1ind].unit)
        Bstack = []
        for Bimg, Bmask in zip([Bimgs[B0ind], Bimg1], Bmasks):
            tmpData = Bimg.data.copy()
            tmpData[Bmask.data.astype(bool)] = np.NaN
            Bstack.append(tmpData)

        # Do a quick median based fill-in-the-blank
        Bstack = np.array(Bstack)
        Bstack = np.nanmedian(Bstack, axis=0)
        Bstack = ai.reduced.ReducedScience(Bstack)

        # # This is the inpainter portion... it may not be the best procedure
        # # Inpaint remaining NaN pixels
        # BstackInpainter = ai.utilitywrappers.Inpainter(Bstack)
        # Bstack          = BstackInpainter.inpaint_nans()

        # Now fill in the bad data for Bimg0 and Bimg1
        Bimg0data = Bimgs[B0ind].data
        fillInds  = np.where(Bmasks[B0ind].data.astype(bool))
        Bimg0data[fillInds] = Bstack.data[fillInds]
        Bimgs[B0ind].data = Bimg0data

        # Include the original off-target difference in the Bimgs[1] image
        Bimg1data = Bimgs[B1ind].data
        fillInds  = np.where(Bmasks[B1ind].data.astype(bool))
        Bimg1data[fillInds] = Bstack.data[fillInds] + offTargetDifference
        Bimgs[B1ind].data = Bimg1data

    # Loop through all the on-target files and subctract off-target values
    for Afile in Afiles:
        # Build the output file
        outFile = os.path.join(bkgFreeImagesDir, Afile)

        # Check if this file already exists
        if os.path.isfile(outFile):
            print('File {} already exists... skipping to next group'.format(os.path.basename(outFile)))
            continue

        # Read in this Aimg
        Aimg = ai.reduced.ReducedScience.read(
            os.path.join(S3_dir, Afile)
        )

        # Locate pixels in this frame within the 2MASS region
        ny, nx = Aimg.shape
        yy, xx = np.mgrid[0:ny, 0:nx]
        RA, Dec = Aimg.wcs.wcs_pix2world(xx, yy, 0)
        xx2, yy2 = this2MASSmask.wcs.wcs_world2pix(RA, Dec, 0)
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
        thisMask = np.zeros((ny, nx), dtype=bool)
        thisMask[bt:tp, lf:rt] = this2MASSmask.data[yy2, xx2].astype(bool)
        thisMask = np.logical_or(thisMask, Aimg.data < -1e5)

        # Create the absolute mask of all A and B images
        totalMask    = np.logical_or(combined_Bmask, thisMask)
        maskedInds   = np.where(totalMask)
        unmaskedInds = np.where(np.logical_not(totalMask))

        ########################################################################
        # Construct a background subtracted on-target frame
        ########################################################################
        if (numBfiles == 1):
            # In this case simply directly subtract the off-target frame
            Aimg1 = Aimg - Bimgs[0]

            # Compute the modes of the on-target and off-target frames
            Amode        = mode(Aimg.data[unmaskedInds])
            BmodeAtAtime = mode(Bimgs[0].data[unmaskedInds])
            AmodeDiff     = Amode - BmodeAtAtime

            # Divide the residual image by the flat field
            Aimg1 = Aimg1/thisFlat - AmodeDiff

        if (numBfiles == 2):
            # Construct the linear time interpolation variable
            c1 = (Aimg.julianDate - np.min(Btimes))/(np.max(Btimes) - np.min(Btimes))

            # Construct the interpolated background image
            Aimg1  = Aimg - (c1*Bimgs[B0ind] + (1-c1)*Bimgs[B1ind])

            # Compute the background level removed from this image
            # Find the mode of the unmasked pixels
            Amode        = mode(Aimg.data[unmaskedInds])
            B1mode       = mode(Bimgs[B0ind].data[unmaskedInds])
            B2mode       = mode(Bimgs[B1ind].data[unmaskedInds])
            BmodeAtAtime = (c1*B1mode + (1-c1)*B2mode)
            AmodeDiff    = Amode - BmodeAtAtime

            # Divide the residual image by the flat field
            Aimg1 = Aimg1/thisFlat - AmodeDiff

        ########################################################################
        # Construct a median, background-free on-target frame and fit a
        # polynomial function to that median image
        ########################################################################
        # Use 50-pixel wide bins starting at (xoff, yoff) = (13, 12) to generate
        # a grid of bins within which to compute median values
        yOff, xOff = 13, 12
        dy, dx     = 50, 50

        # Compute a *GRID* of median values (Use the -G- character to indicate
        # this is a GRID of center and edge values)
        yyG,  xxG  = np.mgrid[
            yOff+25:yOff+1001:dy,
            xOff+25:xOff+1001:dx
        ]
        yyGedge, xxGedge = np.ogrid[
            yOff:yOff+1001:dy,
            xOff:xOff+1001:dx
        ]
        yyGcen, xxGcen   = np.ogrid[
            yOff+25:yOff+1001:dx,
            xOff+25:xOff+1001:dy
        ]

        # Flatten the ogrid arrays so they can be quickly indexed
        yyGedge, xxGedge = yyGedge.flatten(), xxGedge.flatten()
        yyGcen,  xxGcen  = yyGcen.flatten(), xxGcen.flatten()

        # Loop through each grid location
        maskedXX   = xx.copy().astype(float)
        maskedXX[maskedInds] = np.NaN
        maskedYY   = yy.copy().astype(float)
        maskedYY[maskedInds] = np.NaN
        maskedData = Aimg1.data.copy()
        maskedData[maskedInds] = np.NaN
        medianX = np.zeros((20, 20))
        medianY = np.zeros((20, 20))
        medianZ = np.zeros((20, 20))
        for ix, x1 in enumerate(xxGcen):
            for iy, y1 in enumerate(yyGcen):
                # Grab the patch for this zone
                bt, tp = yyGedge[iy], yyGedge[iy+1]
                lf, rt = xxGedge[ix], xxGedge[ix+1]
                thisXpatch = maskedXX[bt:tp, lf:rt]
                thisYpatch = maskedYY[bt:tp, lf:rt]
                thisZpatch = maskedData[bt:tp, lf:rt]

                # Check if there is enough data to do a reasonable median estimate
                if np.sum(np.isfinite(thisZpatch)) < (0.25*thisZpatch.size):
                    # Fill in mostly empty zones with NaNs
                    medianX[iy, ix] = np.NaN
                    medianY[iy, ix] = np.NaN
                    medianZ[iy, ix] = np.NaN
                else:
                    # Compute the median in this grid cell location and value
                    medianX[iy, ix] = np.nanmedian(thisXpatch)
                    medianY[iy, ix] = np.nanmedian(thisYpatch)
                    medianZ[iy, ix] = np.nanmedian(thisZpatch)

        # Compute a plane-fit to this median filtered image
        medianInds = np.where(np.isfinite(medianZ))

        # Compute some reasonable weights for these values. Since it is possible
        # that nebular emission contributes to the pixels *closer* to the image
        # center, let's compute the median of the nebular region and weight
        # by radial distance from that point as (r**(-1)).
        weights = 1
        # For now, I don't think it's essential to worry about this. The nebular
        # mask is *extremely* conservative, so I don't think we're being
        # influenced by nebular emission, and we do need as much information
        # about the near-nebula sky variation as possible to avoid having the
        # polynomial dominate the polarization signal.

        # Model the median image with a 3rd (or 4th?) degree polynomial
        p_init = models.Polynomial2D(degree=4)
        fit_p  = fitting.LevMarLSQFitter()

        with warnings.catch_warnings():
            # Ignore model linearity warning from the fitter
            warnings.simplefilter('ignore')
            poly_fit = fit_p(
                p_init,
                medianX[medianInds],
                medianY[medianInds],
                medianZ[medianInds]
            )

        # Subtract the polynomial fit from the on-target frame
        tmpData  = Aimg1.data - poly_fit(xx, yy)

        # Perform a row-by-row flattening to make sure the background subtracted
        # image does not have any residual row-by-row issues.
        flatData = row_by_row_flatten(tmpData, thisMask)

        # Redundantly force the on-target non-nebular pixels to have a median
        # flux of *zero*.
        flatData = flatData - np.median(flatData[unmaskedInds])

        # look for divots from unmasked star in the off-target frames
        # Start by smoothing the data
        median9Data = median_filter(flatData, 9)

        # Mask any clusters of more that 5 pixels less than negative 2-sigma
        mean9, median9, stddev9 = sigma_clipped_stats(median9Data)
        starDivots  = median9Data < -2*stddev9
        all_labels  = measure.label(starDivots)
        all_labels1 = morphology.remove_small_objects(all_labels, min_size=5)
        starDivots  = all_labels1 > 0

        # Remove any pixels along extreme top
        starDivots[ny-10:ny,:] = False

        # Dialate the starDivots mask
        stellarSigma = 5.0 * gaussian_fwhm_to_sigma    # FWHM = 3.0

        # Build a kernel for detecting pixels above the threshold
        stellarKernel = Gaussian2DKernel(stellarSigma, x_size=41, y_size=41)
        stellarKernel.normalize()
        starDivots = convolve_fft(
            starDivots.astype(float),
            stellarKernel.array
        )
        starDivots = (starDivots > 0.01)

        # Capture NaNs and bad values and set them to -1e6 so that PPOL will
        # know what to do with those values.
        badPix = np.logical_or(
            np.logical_not(np.isfinite(flatData)),
            np.abs(flatData) > 1e5
        )
        badPix = np.logical_or(
            badPix,
            starDivots
        )
        badPix = np.logical_or(
            badPix,
            kokopelliMask
        )
        flatData[np.where(badPix)] = -1e6

        # Store the data in the image object
        Aimg1.data = flatData

        # Estimate the amount of background light removed from this image
        subtractedBackground = BmodeAtAtime + AmodeDiff + poly_fit(0.5*nx, 0.5*ny)

        # Store that background subtraction in the header for later erro-analysis
        tmpHead = Aimg1.header.copy()
        tmpHead['SUB_BKG'] = (subtractedBackground, 'The amount of background subtracted')
        Aimg1.header = tmpHead

        # Write the background free file to disk
        Aimg1.write(outFile, dtype=np.float32)

print('Done!')
