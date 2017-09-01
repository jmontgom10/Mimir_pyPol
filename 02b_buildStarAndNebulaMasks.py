import os
import glob
import numpy as np
from skimage import measure, morphology
from scipy import ndimage
from astropy.table import Table, Column
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel

# For debugging
import matplotlib.pyplot as plt

# Add the AstroImage class
import astroimage as ai

# Add the header handler to the BaseImage class
from Mimir_header_handler import Mimir_header_handler
ai.set_instrument('2MASS')

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

# Setup new directory for masks
maskDir = os.path.join(pyPol_data, 'Masks')
if (not os.path.isdir(maskDir)):
    os.mkdir(maskDir, 0o755)

starMaskDir = os.path.join(maskDir, 'starMasks')
if (not os.path.isdir(starMaskDir)):
    os.mkdir(starMaskDir, 0o755)

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

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

    # Grab the part *only* connected to the central, nebular region
    ny, nx = fgdRegion.shape
    all_labels   = measure.label(fgdRegion)
    nebularLabel = all_labels[ny//2, nx//2]
    nebularMask  = all_labels == nebularLabel
    starMask     = np.logical_and(
        all_labels > 0,
        all_labels != nebularLabel
    )
    all_labels  = measure.label(starMask)
    all_labels1 = morphology.remove_small_objects(all_labels, min_size=50)
    starMask    = all_labels1 > 0

    # Dilate a TOOON to be conservatine...
    nebularSigma = 20.0 * gaussian_fwhm_to_sigma    # FWHM = 3.0

    # Build a kernel for detecting pixels above the threshold
    nebularKernel = Gaussian2DKernel(nebularSigma, x_size=41, y_size=41)
    nebularKernel.normalize()
    nebularMask = convolve_fft(
        nebularMask.astype(float),
        nebularKernel.array
    )
    nebularMask = (nebularMask > 0.01)

    # Expand a second time to be conservative
    nebularMask = convolve_fft(
        nebularMask.astype(float),
        nebularKernel.array
    )
    nebularMask = (nebularMask > 0.01)

    # Do a less aggressive dilation of the stellar mask
    stellarSigma = 10.0 * gaussian_fwhm_to_sigma    # FWHM = 3.0

    # Build a kernel for detecting pixels above the threshold
    stellarKernel = Gaussian2DKernel(stellarSigma, x_size=41, y_size=41)
    stellarKernel.normalize()
    stellarMask = convolve_fft(
        fgdRegion.astype(float),
        stellarKernel.array
    )
    stellarMask = (stellarMask > 0.01)

    # Recombine the nebular and stellar components
    fgdRegion = np.logical_or(nebularMask, stellarMask)

    # Return the flux-bright pixels to the user
    return fgdRegion

################################################################################

# Read in the Kokopelli Mask
kokopelliMask = ai.reduced.ReducedScience.read('kokopelliMask.fits')

# Dilate the mask in order to be more conservative.
kokopelliMask.data = ndimage.binary_dilation(kokopelliMask.data).astype(int)

# Construct the 2MASS masks and save to disk
# Read in all the 2MASS images and store them in a dictionary for quick reference
TMASS_Hfiles = np.array(glob.glob(os.path.join(TMASSdir, '*H.fits')))
TMASS_Kfiles = np.array(glob.glob(os.path.join(TMASSdir, '*Ks.fits')))

# Read in the 2MASS images
TMASS_HimgList = [ai.reduced.ReducedScience.read(f) for f in TMASS_Hfiles]
TMASS_KimgList = [ai.reduced.ReducedScience.read(f) for f in TMASS_Kfiles]

# Convert the images to "background masks"
for img in TMASS_HimgList:
    # Construct the output name for this mask
    base        = os.path.basename(img.filename)
    targetMask  = base.split('.')[0] + '_mask.fits'
    outFilename = os.path.join(TMASSdir, targetMask)

    # Skip files that have already been done
    if os.path.isfile(outFilename): continue

    print('Building background mask for {}'.format(os.path.basename(img.filename)))
    tmp = img.copy()
    tmp.data = find_2MASS_flux(img.data).astype(int)
    tmp.write(outFilename, dtype=np.uint8)

for img in TMASS_KimgList:
    # Construct the output name for this mask
    base        = os.path.basename(img.filename)
    targetMask  = base.split('.')[0] + '_mask.fits'
    outFilename = os.path.join(TMASSdir, targetMask)

    # Skip files that have already been done
    if os.path.isfile(outFilename): continue

    print('Building background mask for {}'.format(os.path.basename(img.filename)))
    tmp = img.copy()
    tmp.data = find_2MASS_flux(img.data).astype(int)
    tmp.write(outFilename, dtype=np.uint8)

# Now that the 2MASS files have been read in, it is safe to set the Mimir_header_handler
ai.set_instrument('Mimir')
ai.reduced.ReducedScience.set_header_handler(Mimir_header_handler)

#Loop through each file in the fileList variable
numberOfFiles = len(fileIndex)
bkgLevels = fileIndex['BACKGROUND']
print('{0:3.1%} complete'.format(0), end='\r')
for iRow, row in enumerate(fileIndex):
    # Grab the relevant information for this file
    thisTarget     = row['TARGET']
    thisFilter     = row['FILTER']
    thisDitherType = row['DITHER_TYPE']
    thisAB         = row['AB']

    # If this is an on-target (A) frame, then skip it!
    if thisAB == 'A': continue

    # Construct the path to the PPOL S3_Astrometry file
    thisFile = os.path.join(S3_dir, row['FILENAME'])

    # Check if the file has already been written and skip those which have been
    maskBasename = os.path.basename(thisFile)
    maskFullname = os.path.join(starMaskDir, maskBasename)
    if os.path.isfile(maskFullname): continue

    # Read in this image
    thisImg = ai.reduced.ReducedScience.read(thisFile)

    # Attempt to recover a background estimate. If not, then just fill with -1e6
    try:
        # Locate the non-stellar pixels in this image
        photAnalyzer = ai.utilitywrappers.PhotometryAnalyzer(thisImg)
        try:
            _, psfParams = photAnalyzer.get_psf()
            fwhm = 2.355*np.sqrt(psfParams['sminor']*psfParams['smajor'])
        except:
            fwhm = 4.5

        xs, ys = thisImg.get_sources(FWHMguess = fwhm, minimumSNR = 4.0, satLimit = 1e20)
        starFluxes, fluxUncerts = photAnalyzer.aperture_photometry(
            xs, ys, 2.5, 24, 26, mask=(thisImg.data < -1e4)
        )

        # import matplotlib.pyplot as plt
        # plt.ion()
        # plt.figure()
        # plt.imshow(kokopelliMask.data, origin='lower', interpolation='nearest')
        # plt.autoscale(False)
        # plt.scatter(xs, ys, facecolor='none', edgecolor='w')

        # Catch bad stars
        inKokopelli = kokopelliMask.data[ys.round().astype(int), xs.round().astype(int)]
        goodInds    = np.where(
            np.logical_and(
                starFluxes > 0,
                np.logical_not(inKokopelli)
            )
        )
        xs         = xs[goodInds]
        ys         = ys[goodInds]
        starFluxes = starFluxes[goodInds]
        starRadii  = 5*np.log10(starFluxes)

        # thisImg.show()
        # thisImg.image.axes.autoscale(False)
        # thisImg.image.axes.scatter(xs,ys, facecolor='none', edgecolor='white')
        # import pdb; pdb.set_trace()
        # plt.close('all')
        #
        # Loop through each star and make its mask
        ny, nx   = thisImg.shape
        yy, xx   = np.mgrid[0:ny, 0:nx]
        starMask = False
        for xs1, ys1, rs in zip(xs, ys, starRadii):
            if not np.isfinite(rs): import pdb;
            # Compute the distances from this star
            # Mask any pixels within 1 radius of this star
            starMask = np.logical_or(
                starMask,
                np.sqrt((xx - xs1)**2 + (yy - ys1)**2) < rs
            )
        # Write the mask to disk
        maskImg = ai.reduced.ReducedScience(starMask.astype(int))
        maskImg.write(maskFullname, dtype=np.uint8)

    except:
        print('Failed to save file {}'.format(maskFullname))

    # Update on progress
    print('{0:3.1%} complete'.format(iRow/numberOfFiles), end='\r')

# Alert the user that everything is complete
print('{0:3.1%} complete'.format(1), end='\n\n')

print('Done!')
