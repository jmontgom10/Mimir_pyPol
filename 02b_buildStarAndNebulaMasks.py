import os
import glob
import numpy as np
import warnings
from skimage import measure, morphology
from scipy import ndimage
from astropy.table import Table, Column
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astroquery.vizier import Vizier

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
useRows   = np.where(fileIndex['USE'])
fileIndex = fileIndex[useRows]

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
kokopelliMask.data = ndimage.binary_dilation(kokopelliMask.data, iterations=8).astype(int)

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

# Set the Vizier download size to be unlimited
Vizier.ROW_LIMIT = -1

# Group by target HWP
groupedFileIndex = fileIndex.group_by(['GROUP_ID', 'HWP'])

#Loop through each file in the fileList variable
numberOfFiles = len(fileIndex)
bkgLevels = fileIndex['BACKGROUND']
print('{0:3.1%} complete'.format(0), end='\r')

iRow = 0
for group in groupedFileIndex.groups:
    # Increment the row counter
    iRow += len(group)

    # Grab the relevant information for this group
    thisTarget     = np.unique(group['TARGET'])[0]
    thisFilter     = np.unique(group['FILTER'])[0]

    # Re-group by dither pointing
    ABBAsubGroups = group.group_by(['AB'])

    for ABBAgroup in ABBAsubGroups.groups:
        # Grab the relevant information for this subgroup
        thisAB = np.unique(ABBAgroup['AB'])[0]

        # If this is an on-target (A) subgroup, then skip it!
        if thisAB == 'A': continue

        # Grab the off-target files
        Bfiles = []
        maskFilenames = []
        for thisFile in ABBAgroup['FILENAME']:
            # Append the B-file to use
            Bfiles.append(os.path.join(S3_dir, thisFile))

            # BUild the mask name
            maskBasename = os.path.basename(thisFile)
            maskFilenames.append(os.path.join(starMaskDir, maskBasename))

    # Check if the file has already been written and skip those which have been
    if all([os.path.isfile(f) for f in maskFilenames]): continue

    # Read in the off-target frames
    Bimgs = [ai.reduced.ReducedScience.read(f) for f in Bfiles]

    numBimgs = len(Bimgs)
    if numBimgs > 1:
        # Combine the images to get a quick map of the region to download
        BimgStack = ai.utilitywrappers.ImageStack(Bimgs, gobble=False)
        BimgStack.align_images_with_wcs()

        # Determine the boundaries of the region to download 2MASS data
        referenceImage = BimgStack.imageList[0]
    else:
        referenceImage = Bimgs[0]

    # Get the image shape and coordinates
    ny, nx = referenceImage.shape
    lfrt, bttp = referenceImage.wcs.wcs_pix2world([0, ny], [0, nx], 0)
    lf, rt = lfrt
    bt, tp = bttp

    # Grab the maximum width and the median (RA, Dec)
    RAcen, DecCen = 0.5*(lf + rt), 0.5*(bt + tp)
    height = (tp - bt)*u.deg
    width  = (lf - rt)*np.cos(np.deg2rad(DecCen))*u.deg

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        # Download the 2MASS point source catalog
        tmassCatalog = Vizier.query_region(
            SkyCoord(
                ra=RAcen, dec=DecCen,
                unit=(u.deg, u.deg),
                frame='fk5'
            ),
            width=width,
            height=height,
            catalog='II/246/out'
        )[0]

        # Cut off low SNR detections
        tmassFilter = referenceImage.filter[0]
        tmassSNR    = tmassCatalog[tmassFilter+'snr']
        goodDetections = np.logical_and(
            tmassSNR.data.data > 5.0,
            np.logical_not(tmassSNR.mask)
        )
        goodDetections = np.logical_and(
            goodDetections,
            np.logical_not(tmassCatalog[tmassFilter+'mag'].mask)
        )

        # Cull the bad data
        tmassCatalog = tmassCatalog[goodDetections]

    # Grab the RAs, Decs, and magnitudes
    RAs, Decs = tmassCatalog['_RAJ2000'], tmassCatalog['_DEJ2000']
    mags      = tmassCatalog[tmassFilter+'mag']

    # Loop through each file and build the preliminary mask
    starMasks = []
    for thisImg in Bimgs:
        # # Read in the image and store it for possible later use
        # thisImg = ai.reduced.ReducedScience.read(Bfile)
        #
        # # Attempt to recover a background estimate. If not, then just fill with -1e6
        # # Locate the non-stellar pixels in this image
        # photAnalyzer = ai.utilitywrappers.PhotometryAnalyzer(thisImg)
        # try:
        #     _, psfParams = photAnalyzer.get_psf()
        #     FWHM = 2.355*np.sqrt(psfParams['sminor']*psfParams['smajor'])
        # except:
        #     FWHM = 4.5

        # xs, ys = thisImg.get_sources(FWHMguess = FWHM, minimumSNR = 3.5,
        #     satLimit = 1e20, edgeLimit = 21)
        # starFluxes, fluxUncerts = photAnalyzer.aperture_photometry(
        #     xs, ys, FWHM, 24, 26, mask=(thisImg.data < -1e4)
        # )
        #
        # # Catch bad stars
        # kokopelliArtifacts = kokopelliMask.data[ys.round().astype(int), xs.round().astype(int)]
        #
        # # Look through the stars in Kokopelly and determine which are *real*
        # realStars          = (kokopelliArtifacts.astype(int)*starFluxes > 4e3)
        # kokopelliArtifacts = np.logical_and(
        #     kokopelliArtifacts,
        #     np.logical_not(realStars)
        # )
        #
        # # Only keep those stars which are not kokopilli artifacts
        # goodInds    = np.where(
        #     np.logical_and(
        #         starFluxes > 0,
        #         np.logical_not(kokopelliArtifacts)
        #     )
        # )
        # xs         = xs[goodInds]
        # ys         = ys[goodInds]
        # starFluxes = starFluxes[goodInds]

        # Now simply mask *any* of the stars downloaded
        xs, ys    = thisImg.wcs.wcs_world2pix(RAs, Decs, 0)
        starRadii = 35 - 1.5*mags

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

        # Store the mask for later use
        starMasks.append(starMask)

    # # If more than one image exists in this group, then do a secondary pass to
    # # locate the dimmer stars
    # numBimgs = len(Bimgs)
    # if numBimgs == 1:
    #     # Grab the available Bimg
    #     tmpImg  = Bimgs[0].copy()
    #
    #     # Smooth the data to look for lower SNR stars
    #     tmpImg.data = ndimage.median_filter(tmpImg.data, 3)
    #
    #     # Construct a temporary image to do another pass as the low SNR stars
    #     tmpFWHM = np.sqrt(FWHM**2 + (0.5*3)**2)
    #     xs, ys = tmpImg.get_sources(FWHMguess = tmpFWHM, minimumSNR = 2.5,
    #         satLimit = 1e20, edgeLimit = 21)
    #     starFluxes, fluxUncerts = photAnalyzer.aperture_photometry(
    #         xs, ys, tmpFWHM, 24, 26, mask=(tmpImg.data < -1e4)
    #     )
    #
    #     # Catch bad stars
    #     kokopelliArtifacts = kokopelliMask.data[ys.round().astype(int), xs.round().astype(int)]
    #
    #     # Look through the stars in Kokopelly and determine which are *real*
    #     realStars          = (kokopelliArtifacts.astype(int)*starFluxes > 4e3)
    #     kokopelliArtifacts = np.logical_and(
    #         kokopelliArtifacts,
    #         np.logical_not(realStars)
    #     )
    #
    #     # Only keep those stars which are not kokopilli artifacts
    #     goodInds    = np.where(
    #         np.logical_and(
    #             starFluxes > 0,
    #             np.logical_not(kokopelliArtifacts)
    #         )
    #     )
    #     xs         = xs[goodInds]
    #     ys         = ys[goodInds]
    #     starFluxes = starFluxes[goodInds]
    #     starRadii  = 5*np.log10(starFluxes)
    #
    #     # Loop through each star and make its mask
    #     ny, nx   = thisImg.shape
    #     yy, xx   = np.mgrid[0:ny, 0:nx]
    #     starMask = starMasks[0]
    #     for xs1, ys1, rs in zip(xs, ys, starRadii):
    #         if not np.isfinite(rs): import pdb;
    #         # Compute the distances from this star
    #         # Mask any pixels within 1 radius of this star
    #         starMask = np.logical_or(
    #             starMask,
    #             np.sqrt((xx - xs1)**2 + (yy - ys1)**2) < rs
    #         )
    #
    #     # Store the mask for later use
    #     starMasks[0] = starMask
    #
    # elif numBimgs > 1:
    #     # Loop through each
    #     for iImg in range(numBimgs):
    #         # Determine which image is the primary image and which is secondary
    #         if iImg == 0:
    #             thisImg  = Bimgs[0]
    #             otherImg = Bimgs[1]
    #         elif iImg == 1:
    #             thisImg  = Bimgs[1]
    #             otherImg = Bimgs[0]
    #         else:
    #             print('What?! How did you even get here?!')
    #             import pdb; pdb.set_trace()
    #
    #         # Grab the corresponding mask
    #         thisMask = starMasks[iImg]
    #
    #         # Subtract the two images from eachother
    #         diffData = otherImg.data - thisImg.data
    #
    #         # Smooth the difference image
    #         median9Data  = ndimage.median_filter(diffData, 9)
    #
    #         # LOOK FOR DIVOTS IN GENERAL MEDIAN FILTERED IMAGE
    #         # Locate pixels less than negative 2-sigma
    #         mean9, median9, stddev9 = sigma_clipped_stats(median9Data)
    #         starDivots = np.nan_to_num(median9Data) < (mean9 -4*stddev9)
    #
    #         # Remove anything that is smaller than 20 pixels
    #         all_labels  = measure.label(starDivots)
    #         all_labels1 = morphology.remove_small_objects(all_labels, min_size=20)
    #         label_hist, label_bins = np.histogram(
    #             all_labels1,
    #             bins=np.arange(all_labels1.max() - all_labels1.min())
    #         )
    #         label_mode  = label_bins[label_hist.argmax()]
    #         starDivots = all_labels1 != label_mode
    #
    #         # Remove any pixels along extreme top
    #         starDivots[ny-10:ny,:] = False
    #
    #         # Dialate the starDivots mask
    #         stellarSigma = 5.0 * gaussian_fwhm_to_sigma    # FWHM = 3.0
    #
    #         # Build a kernel for detecting pixels above the threshold
    #         stellarKernel = Gaussian2DKernel(stellarSigma, x_size=41, y_size=41)
    #         stellarKernel.normalize()
    #         starDivots = convolve_fft(
    #             starDivots.astype(float),
    #             stellarKernel.array
    #         )
    #         starDivots = (starDivots > 0.01)
    #
    #         # Compbine the divots mask and the original mask
    #         fullMask = np.logical_or(thisMask, starDivots)
    #
    #         # Store the mask back in its list
    #         starMasks[iImg] = ai.reduced.ReducedScience(fullMask.astype(int))
    #
    # # Do a finel loop-through to make sure there is as much agreement between
    # # the two masks as possible
    # if numBimgs > 1:
    #     # Construct an image stack and compute image offsets
    #     BimgStack = ai.utilitywrappers.ImageStack(Bimgs)
    #     dx, dy = BimgStack.get_wcs_offsets(BimgStack)
    #
    #     try:
    #         starMask0 = starMasks[0].copy()
    #         starMask1 = starMasks[1].copy()
    #     except:
    #         print('Why are there not 2 starMasks?')
    #         import pdb; pdb.set_trace()
    #
    #     for iMask in range(numBimgs):
    #         # Determine which image is the primary image and which is secondary
    #         if iMask == 0:
    #             dx1 = dx[1] - dx[0]
    #             dy1 = dy[1] - dy[0]
    #             thisMask  = starMask0
    #             otherMask = starMask1
    #         elif iMask == 1:
    #             dx1 = dx[0] - dx[1]
    #             dy1 = dy[0] - dy[1]
    #             thisMask  = starMask1
    #             otherMask = starMask0
    #         else:
    #             print('What?! How did you even get here?!')
    #             import pdb; pdb.set_trace()
    #
    #         # Shift the mask accordingly
    #         shiftedOtherMask = otherMask.shift(dx1, dy1)
    #
    #         # Combine this mask and the shifted mask
    #         fullMask = np.logical_or(
    #             thisMask.data,
    #             shiftedOtherMask.data
    #         )
    #
    #         # Store the mask for a final write-to-disk
    #         starMasks[iMask] = fullMask

    # Look through the masks and write to disk
    for maskFile, starMask in zip(maskFilenames, starMasks):
        try:
            # Write the mask to disk
            maskImg = ai.reduced.ReducedScience(starMask.astype(int))
            maskImg.write(maskFile, dtype=np.uint8)

        except:
            print('Failed to save file {}'.format(maskFile))

    # Update on progress
    print('{0:3.1%} complete'.format(iRow/numberOfFiles), end='\r')

# Alert the user that everything is complete
print('{0:3.1%} complete'.format(1), end='\n\n')

print('Done!')
