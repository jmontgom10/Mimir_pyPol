# Diagnoses the gradient problem...

import os
import sys
import glob
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from photutils import detect_threshold, detect_sources
from scipy.ndimage.filters import median_filter, gaussian_filter
from photutils import Background2D

# Add the AstroImage class
import astroimage as ai

# Add the header handler to the BaseImage class
sys.path.insert(0, 'C:\\Users\\Jordan\\Libraries\\python\\Mimir_pyPol')
from Mimir_header_handler import Mimir_header_handler

# This is the location of all PPOL reduction directory
PPOL_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_reduced\\201611\\notPreFlattened'

# Build the path to the S3_Asotrometry files
S3_dir = os.path.join(PPOL_dir, 'S3_Astrometry')

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\'

# This is the location of the (pre-computed) star masks for the B-images
maskDir = os.path.join(pyPol_data, 'Masks')
starMaskDir = os.path.join(maskDir, 'starMasks')

# Build the path to the supersky directory
hwpImagesDir = os.path.join(pyPol_data, 'bkgFreeHWPimages')
if (not os.path.isdir(hwpImagesDir)):
    os.mkdir(hwpImagesDir, 0o755)

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

# Read in the kokopelli mask
from astropy.io import fits
kokopelliHDUlist = fits.open('..\\kokopelliMask.fits')
kokopelliMask = (kokopelliHDUlist[0].data > 0)

# Read in the 2MASS masks
TMASSdir  = "..\\2MASSimages"

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

# # Define a (very) quick plane-fitting function
# def fitPlaneSVD(XYZ):
#     """Solves for thebest fitting plane to the provided (x,y,z) points"""
#     [rows,cols] = XYZ.shape
#     # Set up constraint equations of the form  AB = 0,
#     # where B is a column vector of the plane coefficients
#     # in the form b(1)*X + b(2)*Y +b(3)*Z + b(4) = 0.
#     p = (np.ones((rows,1)))
#     AB = np.hstack([XYZ,p])
#     [u, d, v] = np.linalg.svd(AB,0)
#     B = v[3,:];                    # Solution is last column of v.
#     nn = np.linalg.norm(B[0:3])
#     B = B / nn
#     # return B[0:3]
#     return B

# Define a plane fitting function for use within this method only
def planeFit(points):
    """
    p, n = planeFit(points)

    Given an array, points, of shape (d,...)
    representing points in d-dimensional space,
    fit an d-dimensional plane to the points.
    Return a point, p, on the plane (the point-cloud centroid),
    and the normal, n.
    """

    points = np.reshape(points, (np.shape(points)[0], -1)) # Collapse trialing dimensions
    assert points.shape[0] <= points.shape[1], "There are only {} points in {} dimensions.".format(points.shape[1], points.shape[0])
    ctr = points.mean(axis=1)
    x = points - ctr[:,np.newaxis]
    M = np.dot(x, x.T) # Could also use np.cov(x) here.
    return ctr, np.linalg.svd(M)[0][:,-1]

# Define a quick mode function
def mode(array):
    """An estimate of the statistical mode of this array"""
    # SUPER fast and sloppy mode estimate:
    mean, median, std = sigma_clipped_stats(array)
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
        hist, flatBins = np.histogram(array.flatten(), bins=bins)

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
fileIndexByGroup = fileIndex.group_by(['GROUP_ID', 'HWP'])

gradientDict = {
    'NGC2023':{
        'H':{
            'gx':[],
            'gy':[],
            'HWP':[]
        },
        'Ks':{
            'gx':[],
            'gy':[],
            'HWP':[]
        }
    },
    'NGC7023':{
        'H':{
            'gx':[],
            'gy':[],
            'HWP':[]
        },
        'Ks':{
            'gx':[],
            'gy':[],
            'HWP':[]
        }
    },
    'M78':{
        'H':{
            'gx':[],
            'gy':[],
            'HWP':[]
        },
        'Ks':{
            'gx':[],
            'gy':[],
            'HWP':[]
        }
    }

}

# Loop through each grouping
for group in fileIndexByGroup.groups:
    # Grab the current target information
    thisGroupName = str(np.unique(group['OBJECT'].data)[0])
    thisTarget    = str(np.unique(group['TARGET'].data)[0])
    thisGroupID   = str(np.unique(group['GROUP_ID'].data)[0])
    thisFilter    = str(np.unique(group['FILTER'].data)[0])
    thisHWP       = str(np.unique(group['HWP'].data)[0])

    # if thisFilter == 'H': continue

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
    if len(BimgFiles) == 0: continue

    # Think about what to do in the case of only one B image (skip it for now)
    if len(BimgFiles) == 1: continue

    # Quickly read in both B images
    Bimgs  = [ai.reduced.ReducedScience.read(f) for f in BimgFiles]
    Btimes = np.array([img.julianDate for img in Bimgs])
    B1ind  = Btimes.argmin()
    B2ind  = Btimes.argmax()

    # Read in both masks and create a combined mask
    Bmasks = [ai.reduced.ReducedScience.read(f) for f in BmaskFiles]
    combinedBmask = False
    for Bmask in Bmasks:
        combinedBmask = np.logical_or(combinedBmask, Bmask.data.astype(bool))

    for Afile in Afiles:
        # Build the output file
        outFile = os.path.join(hwpImagesDir, Afile)

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
        totalMask    = np.logical_or(combinedBmask, thisMask)
        maskedInds   = np.where(totalMask)
        unmaskedInds = np.where(np.logical_not(totalMask))

        # Construct the linear time interpolation variable
        c1 = (Aimg.julianDate - np.min(Btimes))/(np.max(Btimes) - np.min(Btimes))

        # Construct the interpolated background image
        Aimg1  = Aimg - (c1*Bimgs[B1ind] + (1-c1)*Bimgs[B2ind])

        # Divide the *thermal-emission-free* image by the flat
        Aimg1 = Aimg1/thisFlat

        # Correct for the oversubtraction of airglow
        # Find the mode of the unmasked pixels for A and B frames
        Amode  = mode(Aimg.data[unmaskedInds])
        B1mode = mode(Bimgs[B1ind].data[unmaskedInds])
        B2mode = mode(Bimgs[B2ind].data[unmaskedInds])

        # Compute the difference in the on-target and off-target sky modes
        BmodeAtAtime = (c1*B1mode + (1-c1)*B2mode)
        Amode        = mode(Aimg.data[unmaskedInds])

        # Compute the difference between the apparent mode and the expected mode
        AmodeDiff = Amode - BmodeAtAtime

        # # Remove this 'oversubtraction' effect
        # # (possible undersubtraction in same cases)
        # Aimg1 = Aimg1 - (AmodeDiff*Aimg1.unit)

        # Compute a grid of median values
        yy,  xx  = np.mgrid[13+25:1014:50, 12+25:1013:50]
        yyEdge, xxEdge = np.ogrid[13:1014:50, 12:1013:50]
        yyCen, xxCen   = np.ogrid[13+25:1014:50, 12+25:1013:50]

        yyEdge, xxEdge = yyEdge.flatten(), xxEdge.flatten()
        yyCen,  xxCen  = yyCen.flatten(), xxCen.flatten()

        # Use 50-pixel wide bins starting at (xoff, yoff) = (13, 12)
        yoff, xoff = 13, 12
        dy, dx     = 50, 50

        # Loop through each grid location
        maskedArr = Aimg1.data.copy()
        maskedArr[maskedInds] = np.NaN
        medianArr = np.zeros((20,20))
        for ix, x1 in enumerate(xxCen):
            for iy, y1 in enumerate(yyCen):
                # Grab the patch for this zone
                bt, tp = yyEdge[iy], yyEdge[iy+1]
                lf, rt = xxEdge[ix], xxEdge[ix+1]
                thisPatch = maskedArr[bt:tp, lf:rt]

                # Check if there is enough data to do a reasonable median estimate
                if np.sum(np.isfinite(thisPatch)) < (0.25*thisPatch.size):
                    medianArr[iy, ix] = np.NaN
                else:
                    # Compute the median in this grid cell
                    medianArr[iy, ix] = np.nanmedian(thisPatch)

        # Compute a plane-fit to this median filtered image
        medianInds = np.where(np.isfinite(medianArr))
        xyzPts = np.array([xx[medianInds], yy[medianInds], medianArr[medianInds]])

        # gradientPlaneFit = fitPlaneSVD(XYZ)
        # b(1)*X + b(2)*Y +b(3)*Z + b(4) = 0.
        #
        # gradientArr = (
        #     gradientPlaneFit[0]*xx +
        #     gradientPlaneFit[1]*yy +
        # )

        point, normalVec = planeFit(xyzPts)

        # # Grab the airmasses
        # Bairmass = [Bimg.airmass for Bimg in Bimgs]

        # Store the gradient values
        gradientDict[thisTarget][thisFilter]['gx'].append(-normalVec[0]/normalVec[2])
        gradientDict[thisTarget][thisFilter]['gy'].append(-normalVec[1]/normalVec[2])
        gradientDict[thisTarget][thisFilter]['HWP'].append(int(thisHWP))
        # # Compute the value of the fited plane background
        # gradientArr = (
        #     point[2] +
        #     (-normalVec[0]/normalVec[2])*(xx - point[0]) +
        #     (-normalVec[1]/normalVec[2])*(yy - point[1])
        # )
        #
        # # Compute the residual array
        # residArr = medianArr - gradientArr
        #
        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.figure()
        # plt.imshow(medianArr, origin='lower', interpolation = 'nearest')
        #
        # plt.figure()
        # plt.imshow(gradientArr, origin='lower', interpolation='nearest')
        #
        # plt.figure()
        # plt.imshow(residArr, origin='lower', interpolation='nearest')
        # import pdb; pdb.set_trace()
        # plt.close('all')








        # # Subtract the plane from the Aimg1
        # ny, nx = Aimg1.shape
        # yy, xx = np.mgrid[0:ny, 0:nx]
        # gradientArr = (
        #     point[2] +
        #     (-normalVec[0]/normalVec[2])*(xx - point[0]) +
        #     (-normalVec[1]/normalVec[2])*(yy - point[1])
        # )
        # tmpData = Aimg1.data - gradientArr
        # Aimg1.data = tmpData
        #
        # # Now that the "nearby star-scattered-light" has been subtracted...
        # # Divide the *thermal-emission-free* image by the flat
        # Aimg1 = Aimg1/thisFlat
        #
        # # Recompute the median of this subtracted array
        # # Compute a grid of median values
        # yy,  xx  = np.mgrid[13+25:1014:50, 12+25:1013:50]
        # yyEdge, xxEdge = np.ogrid[13:1014:50, 12:1013:50]
        # yyCen, xxCen   = np.ogrid[13+25:1014:50, 12+25:1013:50]
        #
        # yyEdge, xxEdge = yyEdge.flatten(), xxEdge.flatten()
        # yyCen,  xxCen  = yyCen.flatten(), xxCen.flatten()
        #
        # # Use 50-pixel wide bins starting at (xoff, yoff) = (13, 12)
        # yoff, xoff = 13, 12
        # dy, dx     = 50, 50
        #
        # # Loop through each grid location
        # maskedArr = Aimg1.data.copy()
        # maskedArr[maskedInds] = np.NaN
        # medianArr = np.zeros((20,20))
        # for ix, x1 in enumerate(xxCen):
        #     for iy, y1 in enumerate(yyCen):
        #         # Grab the patch for this zone
        #         bt, tp = yyEdge[iy], yyEdge[iy+1]
        #         lf, rt = xxEdge[ix], xxEdge[ix+1]
        #         thisPatch = maskedArr[bt:tp, lf:rt]
        #
        #         # Check if there is enough data to do a reasonable median estimate
        #         if np.sum(np.isfinite(thisPatch)) < (0.25*thisPatch.size):
        #             medianArr[iy, ix] = np.NaN
        #         else:
        #             # Compute the median in this grid cell
        #             medianArr[iy, ix] = np.nanmedian(thisPatch)
        #
        #
        # plt.figure()
        # plt.imshow(medianArr, origin='lower', interpolation = 'nearest')


import pdb; pdb.set_trace()
print('Done!')
