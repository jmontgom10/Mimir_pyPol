# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:08:13 2015

@author: jordan
"""
import os
import sys
import glob
from datetime import datetime, timedelta
import warnings
import numpy as np
from astropy.table import Table, Column, hstack
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.modeling import models, fitting
from astroquery.vizier import Vizier
import scipy.odr as odr
from skimage.measure import moments
from matplotlib import pyplot as plt
from photutils import daofind, aperture_photometry, CircularAperture, CircularAnnulus
import emcee
import corner
import pdb

# Add the AstroImage class
from astroimage.astroimage import AstroImage

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201501\\'

# Set the saturation limit for this image (a property of the detector)
satLimit = 12e3
satLimit = 16e3
# satLimit = 60e3

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

polAngDir = os.path.join(polarimetryDir, 'polAngImgs')
if (not os.path.isdir(polAngDir)):
    os.mkdir(polAngDir, 0o755)

stokesDir = os.path.join(polarimetryDir, 'stokesImgs')
if (not os.path.isdir(stokesDir)):
    os.mkdir(stokesDir, 0o755)

################################################################################
# Import the utility functions to be used here...
from utils_08b import *

# Build a dictionary contaning references to these transformation functions
USNOBtransforms = dict(zip(['V',     'R'    , 'V-R'],
                           [USNOB_V, USNOB_R, USNOB_VR]))

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='ascii.csv')

fileList  = fileIndex['Filename']

# Determine which parts of the fileIndex pertain to science images
useFiles = np.logical_and((fileIndex['Use'] == 1), (fileIndex['Dither'] == 'ABBA'))

# Cull the file index to only include files selected for use
fileIndex = fileIndex[np.where(useFiles)]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
fileIndexByTarget = fileIndex.group_by(['Target', 'Dither'])

# Use the following data for final calibration
# Bands and zero point flux [in Jy = 10^(-26) W /(m^2 Hz)]
# Following table from Bessl, Castelli & Plez (1998)
# Passband  Effective wavelength (microns)  Zero point (Jy)
# U	        0.366                           1790
# B         0.438                           4063
# V         0.545                           3636
# R         0.641                           3064
# I         0.798                           2416
# J         1.22                            1589
# H         1.63                            1021
# K         2.19                            640

zeroFlux = dict(zip(['U',    'B',    'V',    'R'   , 'I'   ],
                    [1790.0, 4063.0, 3636.0, 3064.0, 2416.0]))

wavelength = dict(zip(['U',   'B',   'V',   'R'  , 'I'  ],
                       [0.366, 0.438, 0.545, 0.798, 0.798]))

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

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    print('\nProcessing images for {0}'.format(thisTarget))

    # Look for a photometric star catalog for this target
    catFile = os.path.join(stokesDir, thisTarget + '_stars.csv')
    if os.path.isfile(catFile):
        starCatalog = Table.read(catFile, format='ascii.csv')
    else:
        print('Could not find catalog file for this target')
        print('Please re-run script "08a_selectPhotStars.py"')
        continue

    # Search for all Stokes Intensity images
    Ifile  = os.path.join(stokesDir, thisTarget + '*I.fits')
    Ifiles = glob.glob(Ifile)

    # Search for all the Stokes U images
    Ufile  = os.path.join(stokesDir, thisTarget + '*U.fits')
    Ufiles = glob.glob(Ufile)

    # Search for all the Stokes Q images
    Qfile  = os.path.join(stokesDir, thisTarget + '*Q.fits')
    Qfiles = glob.glob(Qfile)

    # Read in all the Stokes Images found  for this target, and strip the
    # waveband from the header of each
    stokesIimgs = [AstroImage(file1) for file1 in Ifiles]
    waveBands   = [img.header['FILTNME3'].strip() for img in stokesIimgs]

    # Read in the Stokes U images
    stokesUimgs = [AstroImage(file1) for file1 in Ufiles]

    # Read in the Stokes Q images
    stokesQimgs = [AstroImage(file1) for file1 in Qfiles]

    # Compose a dictionary of stokes I, U, and Q images
    stokesIdict = dict(zip(waveBands, stokesIimgs))
    stokesUdict = dict(zip(waveBands, stokesUimgs))
    stokesQdict = dict(zip(waveBands, stokesQimgs))

    del stokesIimgs, stokesUimgs, stokesQimgs, waveBands

    # Grab the WCS info from the header of the stokes Images
    wcsDict   = {}
    yr2000    = datetime(2000,1,1)
    deltaTime = timedelta(0)
    for key, img in stokesIdict.items():
        wcsDict[key] = WCS(img.header)
        thisDate  = img.header['DATE'].split('T')[0]
        thisDate  = datetime.strptime(thisDate, '%Y-%m-%d')
        deltaTime += (thisDate - yr2000)

    # Divide accumulated time vectors by number of measurements
    secPerYr  = 365.25*24*60*60
    deltaTime = deltaTime.total_seconds()/(float(len(stokesIdict))*secPerYr)

    # Form a "catalog" of position entries for matching
    ra1  = starCatalog['_RAJ2000'].data.data*u.deg
    dec1 = starCatalog['_DEJ2000'].data.data*u.deg

    # Propagate proper motions into ra1 and dec1 positions
    pmRA = starCatalog['pmRA'].data.data*(1e-3)*u.arcsec
    pmDE = starCatalog['pmRA'].data.data*(1e-3)*u.arcsec
    ra   = ra1 + pmRA*deltaTime
    dec  = dec1 + pmDE*deltaTime

    # Determine PSF properties for each image
    # Initalize a 2D gaussian model and fitter
    g_init = models.Gaussian2D(amplitude = 2e2,
                               x_mean = 8.0,
                               y_mean = 8.0,
                               x_stddev = 3.0,
                               y_stddev = 3.0,
                               theta = 0.0)
    fit_g = fitting.LevMarLSQFitter()

    #####
    #####
    # PERHAPS I NEED TO ALIGN THE IMAGES *BEFORE* I PERFORM PHOTOMETRY.
    # THAT WAY, THERE ARE NO EXTRA TRANSFORMATIONS APPLIED TO THE IMAGE BETWEEN
    # CALIBRATION AND SAVING TO DISK.
    #####
    #####

    # 1) Loop through all the images
    # 2) Determine more accurate star pixel positions (store in dictionary)
    # 3) Determine star PSF properties (store in dictionary)
    PSF_FWHMs     = []
    xyStarsDict   = {}
    keepStarsDict = {}
    for key, img in stokesIdict.items():
        # Convert the stellar celestial coordinates to pixel positions
        xStars, yStars = wcsDict[key].wcs_world2pix(ra, dec, 0)

        # Grab the image shape
        ny, nx = img.arr.shape

        # Loop through each star
        starFWHMs = []
        xStars1   = []
        yStars1   = []
        keepStars = []
        for xs, ys in zip(xStars, yStars):
            # Cut out a 16x16 pixel region around this estimated location
            x0 = np.int(np.round(xs)) - 8 if np.int(np.round(xs)) - 8 >= 1 else 1
            y0 = np.int(np.round(ys)) - 8 if np.int(np.round(ys)) - 8 >= 1 else 1

            # Compute upper bounds based on lower bounds
            x1 = x0 + 16
            y1 = y0 + 16

            # Double check that upper bounds don't break the rules either
            if x1 > nx - 2:
                x1 = nx - 2
                x0 = x1 - 16
            if y1 > ny - 2:
                y1 = ny - 2
                y0 = y1 - 16

            # Cut out the actual star patch
            patch  = img.arr[y0:y1, x0:x1]

            # Estimate the local sky value
            bigPatch = img.arr[y0-1:y1+1, x0-1:x1+1]
            padPatch = np.pad(patch, ((1,1), (1,1)), mode='constant')

            skyPatch = bigPatch - padPatch
            skyPix   = (np.abs(skyPatch) > 1e-3)
            if np.sum(skyPix) > 0:
                skyInds = np.where(skyPix)
                skyVals = skyPatch[skyInds]
            else:
                print('Cannot find sky')
                pdb.set_trace()

            skyVal = np.median(skyVals)

            # Do a centroid estimate to find the star position
            m    = moments(patch - skyVal, 1)
            xcen = (m[1, 0]/m[0, 0]) + x0
            ycen = (m[0, 1]/m[0, 0]) + y0

            # Re-cut a 16x16 pixel region around this corrected star position
            x0 = np.int(np.round(xcen)) - 8 if np.int(np.round(xcen)) - 8 >= 0 else 0
            y0 = np.int(np.round(ycen)) - 8 if np.int(np.round(ycen)) - 8 >= 0 else 0

            # Compute upper bounds based on lower bounds
            x1 = x0 + 16
            y1 = y0 + 16

            # Double check that upper bounds don't break the rules either
            if x1 > nx - 1:
                x1 = nx - 1
                x0 = x1 - 16
            if y1 > ny - 1:
                y1 = ny - 1
                y0 = y1 - 16

            # Cut out the actual star patch
            patch  = img.arr[y0:y1, x0:x1]

            # Redo a centroid estimate to find the star position.
            # Use this value to test whether or not the Gaussian fit is good.
            m     = moments(patch - skyVal, 1)
            xcen  = (m[1, 0]/m[0, 0])
            ycen  = (m[0, 1]/m[0, 0])
            xcen1 = xcen + x0
            ycen1 = ycen + y0

            # Fit a Gaussian to the star cutout
            with warnings.catch_warnings():
                # Ignore model linearity warning from the fitter
                warnings.simplefilter('ignore')
                yy, xx = np.mgrid[0:patch.shape[0], 0:patch.shape[1]]
                g_fit  = fit_g(g_init, xx, yy, patch - skyVal)

            # Test whether the fitted gaussian is close to the epected location
            xFit, yFit = (g_fit.x_mean.value, g_fit.y_mean.value)
            fitDist    = np.sqrt((xcen - xFit)**2 + (ycen - yFit)**2)

            # Test for fitting and saturation problems
            fitBool      = (fitDist < 2.5)
            satBool      = (patch.max() < satLimit) and (patch.min() > -100)
            thisKeepBool =  fitBool and satBool

            if thisKeepBool == True:
                keepStars.append(True)
                xStars1.append(xcen1)
                yStars1.append(ycen1)
            else:
                # Build the problem analysis string
                probString = ''
                if fitBool:
                    probString = probString + 'fitting '
                if satBool:
                    if len(probString) > 0:
                        probString = probString + 'and saturation'
                    else:
                        probString = probString + 'saturation '
                probString = probString + 'problems'
                print('skipping star at ({0:4d}, {1:4d}): for {2}'.format(
                    np.int(xs.round()), np.int(ys.round()), probString))
                keepStars.append(False)
                xStars1.append(-1)
                yStars1.append(-1)
                continue

            # Store the Gaussian fitted PSF properties in the starFWHMs list
            thisSigma = np.sqrt(np.abs(g_fit.x_stddev.value*g_fit.y_stddev.value))
            thisFWHM  = thisSigma/gaussian_fwhm_to_sigma
            starFWHMs.append(thisFWHM)

        # Store the mean PSF value in the FWHMlist
        mean, med, std = sigma_clipped_stats(starFWHMs)
        # mode = 3.0*med - 2.0*mean
        # mode = 2.5*med - 1.5*mean
        # PSF_FWHMs.append(mode)
        PSF_FWHMs.append(mean)

        # Store star positions in the xyStarsDict
        xyStars = np.array([(xs, ys) for xs, ys in zip(xStars1, yStars1)])
        xyStarsDict[key] = xyStars

        # Store the star test booleans in the keepStarDict
        keepStarsDict[key] = keepStars

    # Grab maximum stellar PSF width and use apRad = 2.5*FWHM  for photometry
    maxFWHM  = np.max(PSF_FWHMs)
    apRad    = 2.5*maxFWHM
    anInRad  = apRad + 2.0
    anOutRad = apRad + 4.0

    # Cull the starCatalog entry to only include non-saturated stars.
    # First check which stars passed the test in ALL bands.
    keepStars = np.ones(len(starCatalog), dtype=bool)
    for key, val in keepStarsDict.items():
        # Make sure the keep tests are all passed
        keepStars = np.logical_and(keepStars, val)

        # Make sure the stars are also far enough from the edge using the newly
        # determined aperture radius to compute the edge criterion.
        ny, nx   = stokesIdict[key].arr.shape
        edgeCut  = np.ceil(anOutRad)
        edgeBool = xyStarsDict[key][:,0] > edgeCut
        edgeBool = np.logical_and(edgeBool,
            xyStarsDict[key][:,0] < nx - 1 - edgeCut)
        edgeBool = np.logical_and(edgeBool,
            xyStarsDict[key][:,1] > edgeCut)
        edgeBool = np.logical_and(edgeBool,
            xyStarsDict[key][:,1] < ny - 1 - edgeCut)

        # Combine the edge test with the previously determined photometry test
        keepStars = np.logical_and(keepStars, edgeBool)

    # Test if at least 4 stars passed the tests in all bands
    if np.sum(keepStars) >= 4:
        # Cull the star catalog to only include detected stars
        keepInds = np.where(keepStars)
        starCatalog = starCatalog[keepInds]

        # Also cull the list of star positions to match between all bands
        xyStarsDict1 = xyStarsDict.copy()
        for key, val in xyStarsDict.items():
            xyStarsDict1[key] = val[keepInds]

        # Delete temporary variables
        xyStarsDict = xyStarsDict1.copy()
        del xyStarsDict1
    else:
        print('Fewer than 4 stars passed the quality tests in all bands.')
        print('Color photometry for this target is impossible')
        pdb.set_trace()

    # Separate out O, J, E, F magnitudes for predicting V and R bands
    # Surveys used for USNO-B1.0:
    # ----------------------------------------------------------
    # #   Name    Emuls  B/R Wavelen.   Zones  Fld  Dates    Epoch
    #                         (nm)     (Dec)         Obs.
    # ----------------------------------------------------------
    # 0 = POSS-I  103a-O (B) 350-500 -30..+90 936 1949-1965 (1st)
    # 1 = POSS-I  103a-E (R) 620-670 -30..+90 936 1949-1965 (1st)
    # 2 = POSS-II IIIa-J (B) 385-540 +00..+87 897 1985-2000 (2nd)
    # 3 = POSS-II IIIa-F (R) 610-690 +00..+87 897 1985-1999 (2nd)
    # 4 = SERC-J  IIIa-J (B) 385-540 -90..-05 606 1978-1990 (2nd)
    # 5 = ESO-R   IIIa-F (R) 630-690 -90..-05 624 1974-1994 (1st)
    # 6 = AAO-R   IIIa-F (R) 590-690 -90..-20 606 1985-1998 (2nd)
    # 7 = POSS-II IV-N   (I) 730-900 +05..+87 800 1989-2000 (N/A)
    # 8 = SERC-I  IV-N   (I) 715-900 -90..+00 892 1978-2002 (N/A)
    # 9 = SERC-I* IV-N   (I) 715-900 +05..+20  25 1981-2002 (N/A)
    # --------------------------------------------------

    # Note: Check that the confirmed sources all come from the expected
    # surveys. If not, then stop and re-evaluate.
    # First grab all the sources used in this data (minus masked points)
    B1src = np.unique(starCatalog['B1S'].data.filled(255))[-2::-1]
    R1src = np.unique(starCatalog['R1S'].data.filled(255))[-2::-1]
    B2src = np.unique(starCatalog['B2S'].data.filled(255))[-2::-1]
    R2src = np.unique(starCatalog['R2S'].data.filled(255))[-2::-1]

    # Now test if all the specified sources are the expected ones
    B1eqO = all([src in [0] for src in B1src])
    R1eqE = all([src in [1] for src in R1src])
    B2eqJ = all([src in [2, 4] for src in B2src])
    R2eqF = all([src in [3, 5, 6] for src in R2src])

    if (B1eqO and R1eqE and B2eqJ and R2eqF):
        # If the sources are all the expected ones, then parse the emulsions
        Omags = starCatalog['B1mag'].data.data
        Emags = starCatalog['R1mag'].data.data
        Jmags = starCatalog['B2mag'].data.data
        Fmags = starCatalog['R2mag'].data.data

        # Build a dictionary of USNO-B1.0 magnitudes
        USNOBmagDict = dict(zip(['O',   'E',   'J',   'F'  ],
                                [Omags, Emags, Jmags, Fmags]))
    else:
        # Insert a pause if one of the sources is wrong...
        print('There are some unexpected sources for the magnitudes')
        print('...stopping...')
        pdb.set_trace()

    # 1) Loop through all the images.
    # 2) Do aperture photometry on the stars
    # 3) Store photometry in photDict
    # Initalize a dictionary to store the airmass corrected (AMC) stokes I imgs
    stokesIdict_AMC = {}
    # Initalize a dictionary to store the photometry tables
    photDict = {}
    for key, img in stokesIdict.items():
        # Now that all the pre-requisites for photometry have been met, it is time
        # to apply a waveband based airmass correction and normalize by the exposure
        # time. The result, stored in the img1 variable, should be used for all
        # subsequent photometry
        atmExtMag = kappa[key]*img.header['AIRMASS']
        expTime   = img.header['EXPTIME']
        img1      = img*((10.0**(0.4*atmExtMag))/expTime)

        # Store corrected image in the stokesIdict_AMC dictionary
        stokesIdict_AMC[key] = img1

        # Grab the star positions
        xyStars        = xyStarsDict[key]
        xStars, yStars = xyStars[:,0], xyStars[:,1]

        # Establish circular apertures for photometry
        apertures = CircularAperture(xyStars, r = apRad)
        annulus_apertures = CircularAnnulus(xyStars,
            r_in = anInRad, r_out = anOutRad)

        # Perform the basic photometry
        rawflux_table = aperture_photometry(img1.arr, apertures,
            error=img1.sigma)
        bkgflux_table = aperture_photometry(img1.arr, annulus_apertures,
            error=img1.sigma)
        phot_table = hstack([rawflux_table, bkgflux_table],
            table_names=['raw', 'bkg'])

        # Compute background contribution and subtract from raw photometry
        bkg_mean = phot_table['aperture_sum_bkg'] / annulus_apertures.area()
        bkg_sig  = phot_table['aperture_sum_err_bkg'] / annulus_apertures.area()

        bkg_sum = bkg_mean * apertures.area()
        bkg_sig = bkg_sig * apertures.area()

        # Compute the variance in the background pixels for each star
        ny, nx  = img1.arr.shape
        yy, xx  = np.mgrid[0:ny, 0:nx]
        bkg_var = []

        # Loop through each star and add the local variance to the uncertainty
        for xy in xyStars:
            xs, ys = xy
            distFromStar = np.sqrt((xx - xs)**2 + (yy - ys)**2)
            skyPixInds   = np.where(np.logical_and(
                (distFromStar > anInRad), (distFromStar < anOutRad)))
            bkg_var.append(np.var(img1.arr[skyPixInds]))

        # Convert the background variance into an array
        bkg_var = np.array(bkg_var)

        # Compute the final photometry and its uncertainty
        final_sum = phot_table['aperture_sum_raw'] - bkg_sum
        final_sig = np.sqrt(phot_table['aperture_sum_err_raw']**2
            + bkg_sig**2
            + bkg_var)
        phot_table['residual_aperture_sum'] = final_sum
        phot_table['residual_aperture_sum_err'] = final_sig

        # Compute the signal-to-noise ratio and find the stars with SNR < 3.0
        SNR = final_sum/final_sig

        # Now estimate the photometry from USNO-B1.0 and store it for later use
        catMags, sigCatMags = USNOBtransforms[key](USNOBmagDict)
        phot_table[key+'_catMag']    = catMags
        phot_table[key+'_sigCatMag'] = sigCatMags

        # Loop through all the stars and detect any duplicate entries. Mark each
        # entry with a semi-unique 'Star ID'
        # Extract the star positions from the photometry table
        # (this is redundant but a nice confirmation that these will be right)
        xStars = phot_table['xcenter_raw']
        yStars = phot_table['ycenter_raw']

        # Initalize an empty list to store the starIDs
        starIDs = -1*np.ones(len(phot_table), dtype=int)
        for ind, row in enumerate(phot_table):
            # Skip over any rows that have been previously treated
            if starIDs[ind] > 0: continue

            # Compute the distance between the current star and all other stars
            xs, ys = row['xcenter_raw'], row['ycenter_raw']
            dist   = np.sqrt((xs - xStars)**2 + (ys - yStars)**2).value

            if np.sum(dist < 2.0) > 0:
                # Mark all stars within 2.0 pixels of the current star with an
                # identical ID.
                IDinds = np.where(dist < 2.0)
                starIDs[IDinds] = ind

        # Add the StarID column to the phot_table
        phot_table.add_column(Column(name='star_id', data=starIDs), index=0)
        # plt.ion()
        # plt.imshow(stokesIdict[key].arr, vmin=0,vmax=800,cmap='gray_r')
        # plt.scatter(phot_table['xcenter_raw'], phot_table['ycenter_raw'],
        #     marker='x', color='red')
        # pdb.set_trace()

        # Sort the phot_table by starID
        sortInds = phot_table['star_id'].data.argsort()
        phot_table = phot_table[sortInds]

        # Store this photometry table in the dictionary for later use
        photDict[key] = phot_table

        # ###########################################################################
        # # PRINT OUT THE PHOTOMETRY TO CHECK FOR CONSISTENCY
        # ###########################################################################
        # xFmtStr    = '{x[0]:>6}.{x[1]:<3}'
        # yFmtStr    = '{y[0]:>6}.{y[1]:<3}'
        # starFmtStr = '{star[0]:>9}.{star[1]:<3}'
        # bkgFmtStr  = '{bkg[0]:>9}.{bkg[1]:<3}'
        # snrFmtStr  = '{snr[0]:>9}.{snr[1]:<3}'
        # print('final photometry is...')
        # print('      x         y        Star Flux     Bkg Flux       SNR')
        # print('===========================================================')
        # printStr = xFmtStr + yFmtStr + starFmtStr + bkgFmtStr + snrFmtStr
        # for i in range(len(SNR)):
        #     xVal = str(xStars[i]).split('.')
        #     xVal[1] = (xVal[1])[0:3]
        #     yVal = str(yStars[i]).split('.')
        #     yVal[1] = (yVal[1])[0:3]
        #     starVal = str(final_sum[i]).split('.')
        #     starVal[1] = (starVal[1])[0:3]
        #     bkgVal = str(bkg_sum[i]).split('.')
        #     bkgVal[1] = (bkgVal[1])[0:3]
        #     snrVal = str(SNR[i]).split('.')
        #     snrVal[1] = (snrVal[1])[0:3]
        #     print(printStr.format(x = xVal, y = yVal, star = starVal,
        #         bkg = bkgVal, snr = snrVal))

    # I need to simultaneously solve a set of linear regressions for photometric
    # zero-point magnitudes and color correction terms
    #
    # E.g.
    # (V_corrected - V_apparent) = a_0 + a_1 * (V_apparent - R_apparent)
    # and
    # (R_corrected - R_apparent) = a_2 + a_3 * (V_apparent - R_apparent)
    # and
    # (V_corrected - R_corrected) = a_4 + a_5 * (V_apparent - R_apparent)
    #

    # Grab all the successfully measured bandpasses
    bandKeys1 = [key for key in stokesIdict.keys()]

    # Ensure that they're in wavelength order
    # Start by constructing an array with
    # Column 0: list of wavebands
    # Column 1: list of wavelengths for that bands
    bandLamArr = np.array([[key, val] for key, val in wavelength.items()])

    # Figure our how to sort this array by increasing wavelength, and create a
    # list of possible wavebands in that sorted order
    sortArr  = bandLamArr[:,1].argsort()
    bandList = (bandLamArr[:,0].flatten())[sortArr]

    # Loop through the wavebands and construct a wavelength ordered list of
    # observed waveband keys in the stokesIdict dictionary.
    bandKeys = []
    for band in bandList:
        if band in bandKeys1:
            bandKeys.append(band)

    # Loop through the bands and construct keys for a "color dictionary"
    colorKeys = []
    for ind, band1 in enumerate(bandKeys[0:len(bandKeys)-1]):
        # Only worry about colors from adjacent wavebands, one index over
        band2 = bandKeys[ind+1]
        colorKeys.append('-'.join([band1, band2]))

    # Prepare for the linear regressions to be done on each band and color
    # Define the model to be used in the fitting
    def lineFunc(B, x):
        return B[1] + B[0]*x

    # Set up ODR with the model and data.
    lineModel = odr.Model(lineFunc)

    # loop through each linear regression
    for colorKey in colorKeys:
        print('Preparing the model outliers with MCMC')

        # Setup the walker count, burn-in steps, and production steps
        n_walkers       = 100
        n_burn_in_steps = 1000
        n_steps         = 2000

        # Treat the photometric regressions for this set of bands...
        # Establish the boundaries of acceptable parameters for the prior
        labels = [
            r"$\theta$",
            r"$b_p$",
            r"$P_b$",
            r"$M_x$",
            r"$\ln V_x$",
            r"$M_y$",
            r"$\ln V_y$"]

        # Create a separate set of labels and indices for those parameters which
        # will be plotted in the posterior distribution "corner plot"
        plotLabels = [
            r"$\theta$",
            r"$b_p$",
            r"$P_b$",
            r"$\ln V_y$"]
        plotInds = np.array([0,1,2,6])

        bounds1 = [(-1.0, 1.0),     # Theta (angle of the line slope)
                   (18.0, 28.0),    # b_perp (min-dist(line-origin))
                   (0.0, 1.0),      # Pb (Probability of sampling an outliers)
                   (-8.0, +8.0),    # Mx (<x> of outlier distribution)
                   (-2.0, 5.0),     # lnVx (log-x-variance of outlier distribution)
                   (-8.0, +8.0),    # My (<y> of outlier distribution)
                   (-2.0, 5.0)]     # lnVy (log-y-variance of outlier distribution)

        bounds2 = [(+0.0, +1.5),    # Theta (angle of the line slope)
                   (18.0, 28.0),    # b_perp (min-dist(line-origin))
                   (0.0, 1.0),      # Pb (Probability of sampling an outliers)
                   (-8.0, +8.0),    # Mx (<x> of outlier distribution)
                   (-2.0, 5.0),     # lnVx (log-x-variance of outlier distribution)
                   (-8.0, +8.0),    # My (<y> of outlier distribution)
                   (-2.0, 5.0)]     # lnVy (log-y-variance of outlier distribution)

        boundsC = [(-0.5, +1.0),    # Theta (angle of the line slope)
                   (-0.4, +0.75),   # b_perp (min-dist(line-origin))
                   (0.0, 1.0),      # Pb (Probability of sampling an outliers)
                   (-8.0, +8.0),    # Mx (<x> of outlier distribution)
                   (-2.0, 5.0),     # lnVx (log-x-variance of outlier distribution)
                   (-8.0, +8.0),    # My (<y> of outlier distribution)
                   (-2.0, 5.0)]     # lnVy (log-y-variance of outlier distribution)

        # Parse the bands used in this color
        band1, band2 = colorKey.split('-')

        # Grab the photometry table for these two bands
        phot_table1 = photDict[band1]
        phot_table2 = photDict[band2]

        # Double check that the star IDs are all matched up
        if len(phot_table1) != len(phot_table2):
            print('Photometry tables do not match!')
            pdb.set_trace()
        totalMatch = np.sum(phot_table1['star_id'].data == phot_table2['star_id'].data)
        if totalMatch < len(phot_table1):
            print('Photometry tables do not match!')
            pdb.set_trace()

        # Since we have confirmed that all the starIDs match up, we will store
        # the values from the first phot_table
        starIDs = phot_table['star_id'].data

        # Grab the fluxes for the calibration stars for these two bands
        flux1 = phot_table1['residual_aperture_sum'].data
        flux2 = phot_table2['residual_aperture_sum'].data
        sigFlux1 = phot_table1['residual_aperture_sum_err'].data
        sigFlux2 = phot_table2['residual_aperture_sum_err'].data

        # Compute the instrumental magnitudes for these two bands
        instMags1    = -2.5*np.log10(flux1)
        instMags2    = -2.5*np.log10(flux2)
        sigInstMags1 = 2.5*np.abs(sigFlux1/(flux1*np.log(10)))
        sigInstMags2 = 2.5*np.abs(sigFlux2/(flux2*np.log(10)))

        # Now grab the catalog magnitudes for the calibration stars
        catMags1    = phot_table1[band1+'_catMag'].data
        catMags2    = phot_table2[band2+'_catMag'].data
        sigCatMags1 = phot_table1[band1+'_sigCatMag'].data
        sigCatMags2 = phot_table2[band2+'_sigCatMag'].data

        # Begin by culling any data from extremely unexpected regions
        # Compute the catalog colors for these stars
        catColors, sig_catColors = USNOBtransforms[colorKey](USNOBmagDict)

        # Compute the band1 - band2 color
        xTest = instMags1 - instMags2
        yTest = catColors

        # Set some boundaries for acceptable color-color data
        # slope1, intercept1 = np.tan(0.561), 0.0055/np.cos(0.561)
        # slope2, intercept2 = np.tan(0.658), 0.233/np.cos(0.658)
        slope1, intercept1 = np.tan(0.45), 0.00/np.cos(0.45)
        slope2, intercept2 = np.tan(0.70), 0.25/np.cos(0.70)
        keepPts = (yTest > (slope1*xTest + intercept1 - 0.25))
        keepPts = np.logical_and(keepPts,
            (yTest < slope2*xTest + intercept2 + 0.25))
        keepInds = np.where(keepPts)

        # Now perform the actual data cuts
        starIDs       = starIDs[keepInds]
        instMags1     = instMags1[keepInds]
        instMags2     = instMags2[keepInds]
        sigInstMags1  = sigInstMags1[keepInds]
        sigInstMags2  = sigInstMags2[keepInds]
        catMags1      = catMags1[keepInds]
        catMags2      = catMags2[keepInds]
        sigCatMags1   = sigCatMags1[keepInds]
        sigCatMags2   = sigCatMags2[keepInds]
        catColors     = catColors[keepInds]
        sig_catColors = sig_catColors[keepInds]

        ########################################################################
        ############################# COLOR-COLOR ##############################
        ########################################################################
        print('Running initial Color-Color regression')
        # Compute the colors for these stars
        xC  = instMags1 - instMags2
        yC  = catColors
        sxC = np.sqrt(sigInstMags1**2 + sigInstMags2**2)
        syC = sig_catColors

        ### THIS CODE SIMPLY DISPLAYS THE DATA TO THE USER TO SEE IF
        ### THE SELECTED "GOOD-DATA" REGION IS ACCEPTABLE.
        ###
        # slope1, intercept1 = np.tan(0.45), 0.00/np.cos(0.45)
        # slope2, intercept2 = np.tan(0.70), 0.25/np.cos(0.70)
        # plt.errorbar(xC, yC, xerr=sxC, yerr=syC, fmt='None', ecolor='k')
        # plt.plot(xC, slope1*xC + intercept1 - 0.25, color='k')
        # plt.plot(xC, slope2*xC + intercept2 + 0.25, color='k')
        # pdb.set_trace()
        # plt.close('all')
        # continue

        # Perform the MCMC sampling of the posterior
        data = (xC, yC, sxC, syC)
        samplerC = MCMCfunc(data, boundsC,
            n_walkers=n_walkers,
            n_burn_in_steps=n_burn_in_steps,
            n_steps=n_steps)

        # Plot the posteriors to see if a reasonable result was obtained.
        # plotSamples = samplerC.flatchain[:,plotInds]
        # plotBounds  = np.array(boundsC)[plotInds]
        # corner.corner(plotSamples, bins=100,
        #     range=plotBounds,
        #     labels=plotLabels)
        #
        # # Save the figure to disk
        # fname = os.path.join(stokesDir, thisTarget + '_MCMC.png')
        # plt.savefig(fname, dpi=300)
        # plt.close('all')

        # Compute the posterior probability that each data-point is "good"
        norm = 0.0
        post_probC = np.zeros(len(data[0]))
        for i in range(samplerC.chain.shape[1]):
            for j in range(samplerC.chain.shape[0]):
                ll_fg, ll_bg = samplerC.blobs[i][j]
                post_probC += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
                norm += 1
        post_probC /= norm

        # Loop through all entries and eliminate the less probable of all
        # *PAIRED* entries.
        keepBool = np.zeros(len(post_probC), dtype=bool)
        for ind, idNum in enumerate(starIDs):
            # Skip over already treated indices
            if keepBool[ind] == True: continue

            # Test which starIDs equal *THIS* starID
            testBool = (starIDs == idNum)
            if np.sum(testBool) > 1:
                # If this ID number is shared by more than one entry, then
                # figure out which entry is more probable and keep only that one
                testIDs   = np.where(starIDs == idNum)
                testProbs = post_probC[testIDs]
                testBool  = testProbs == testProbs.max()
                testInds  = np.where(testBool)[0]
                keepBool[testIDs] = testBool
            else:
                keepBool[ind] = True

        # Now that we've eliminated duplicate data-points, let's eliminate data
        # with less than a 50% posterior probability of being "good"
        keepBool = np.logical_and(keepBool, (post_probC > 0.50))
        keepInds = np.where(keepBool)

        # Now cull the data and re-do the fit
        print('Culling duplicate and probable outlier data')
        starIDs       = starIDs[keepInds]
        instMags1     = instMags1[keepInds]
        instMags2     = instMags2[keepInds]
        sigInstMags1  = sigInstMags1[keepInds]
        sigInstMags2  = sigInstMags2[keepInds]
        catMags1      = catMags1[keepInds]
        catMags2      = catMags2[keepInds]
        sigCatMags1   = sigCatMags1[keepInds]
        sigCatMags2   = sigCatMags2[keepInds]
        catColors     = catColors[keepInds]
        sig_catColors = sig_catColors[keepInds]

        ########################################################################
        ################################ BAND 1 ################################
        ########################################################################
        print('Running {0}-band regression'.format(band1))
        x1  = instMags1 - instMags2
        y1  = catMags1 - instMags1
        sx1 = np.sqrt(sigInstMags1**2 + sigInstMags2**2)
        sy1 = np.sqrt(sigCatMags1**2 + sigInstMags1**2)

        # Perform the MCMC sampling of the posterior
        data = (x1, y1, sx1, sy1)
        sampler1 = MCMCfunc(data, bounds1,
            n_walkers=n_walkers,
            n_burn_in_steps=n_burn_in_steps,
            n_steps=n_steps)

        # # Plot the posteriors to see if a reasonable result was obtained.
        # plt.ion()
        # plotSamples = sampler1.flatchain[:,plotInds]
        # plotBounds  = np.array(bounds1)[plotInds]
        # corner.corner(plotSamples, bins=100,
        #     range=plotBounds,
        #     labels=plotLabels)
        # pdb.set_trace()
        # plt.close('all')

        # Compute the posterior probability that each data-point is "good"
        norm = 0.0
        post_prob1 = np.zeros(len(data[0]))
        for i in range(sampler1.chain.shape[1]):
            for j in range(sampler1.chain.shape[0]):
                ll_fg, ll_bg = sampler1.blobs[i][j]
                post_prob1 += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
                norm += 1
        post_prob1 /= norm

        # Track the outliers from the band-1 MCMC fit
        keepBool = (post_prob1 > 0.5)

        ########################################################################
        ################################ BAND 2 ################################
        ########################################################################
        print('Running {0}-band regression'.format(band2))
        x2  = instMags1 - instMags2
        y2  = catMags2 - instMags2
        sx2 = np.sqrt(sigInstMags1**2 + sigInstMags2**2)
        sy2 = np.sqrt(sigCatMags2**2 + sigInstMags2**2)

        # Perform the MCMC sampling of the posterior
        data = (x2, y2, sx2, sy2)
        sampler2 = MCMCfunc(data, bounds2,
            n_walkers=n_walkers,
            n_burn_in_steps=n_burn_in_steps,
            n_steps=n_steps)

        # # Plot the posteriors to see if a reasonable result was obtained.
        # plotSamples = sampler2.flatchain[:,plotInds]
        # plotBounds  = np.array(bounds1)[plotInds]
        # corner.corner(plotSamples, bins=100,
        #     range=plotBounds,
        #     labels=plotLabels)
        # pdb.set_trace()
        # plt.close('all')

        # Compute the posterior probability that each data-point is "good"
        norm = 0.0
        post_prob2 = np.zeros(len(data[0]))
        for i in range(sampler2.chain.shape[1]):
            for j in range(sampler2.chain.shape[0]):
                ll_fg, ll_bg = sampler2.blobs[i][j]
                post_prob2 += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
                norm += 1
        post_prob2 /= norm

        # Track the outliers from the band-2 MCMC fit
        keepBool = np.logical_and(keepBool, (post_prob2 > 0.5))

        ########################################################################
        ############################# COLOR-COLOR ##############################
        ########################################################################
        # Begin by culling any data marked as outliers in band1 or band2 MCMC.
        keepInds = np.where(keepBool)

        # Now cull the data and perform ODR fits
        print('Culling probable outlier data')
        starIDs       = starIDs[keepInds]
        instMags1     = instMags1[keepInds]
        instMags2     = instMags2[keepInds]
        sigInstMags1  = sigInstMags1[keepInds]
        sigInstMags2  = sigInstMags2[keepInds]
        catMags1      = catMags1[keepInds]
        catMags2      = catMags2[keepInds]
        sigCatMags1   = sigCatMags1[keepInds]
        sigCatMags2   = sigCatMags2[keepInds]
        catColors     = catColors[keepInds]
        sig_catColors = sig_catColors[keepInds]

        # Make sure to identically cull the posterior probabilities too!
        post_prob1 = post_prob1[keepInds]
        post_prob2 = post_prob2[keepInds]

        print('Running final color-color regression')
        # Compute the colors for these stars
        xC  = instMags1 - instMags2
        yC  = catColors
        sxC = np.sqrt(sigInstMags1**2 + sigInstMags2**2)
        syC = sig_catColors

        # Perform the MCMC sampling of the posterior
        data = (xC, yC, sxC, syC)
        samplerC = MCMCfunc(data, boundsC,
            n_walkers=n_walkers,
            n_burn_in_steps=n_burn_in_steps,
            n_steps=n_steps)

        # Compute the posterior probability that each data-point is "good"
        norm = 0.0
        post_probC = np.zeros(len(data[0]))
        for i in range(samplerC.chain.shape[1]):
            for j in range(samplerC.chain.shape[0]):
                ll_fg, ll_bg = samplerC.blobs[i][j]
                post_probC += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
                norm += 1
        post_probC /= norm

        # # Track the outliers from the color-color MCMC fit
        # keepBool = np.logical_and(keepBool, (post_probC > 0.5))
        if np.sum(post_probC < 0.5) > 0:
            print('Color-Color fit still has some outliers?!')
            pdb.set_trace()

        # Grab the "confidence intervals" for each parameter
        truthRanges = [(v[1], v[2]-v[1], v[1]-v[0]) for v in
            zip(*np.percentile(samplerC.flatchain, [16, 50, 84], axis=0))]
        truths = [t[0] for t in truthRanges]

        # Convert these data to slope-intercept space
        tmpData = np.array([
            np.tan(samplerC.flatchain[:,0]),
            samplerC.flatchain[:,1]/np.cos(samplerC.flatchain[:,0])])

        # Compute the median slope and intercept and the covariance matrix
        truthsC = np.percentile(tmpData, 50, axis=1)
        covC    = np.cov(tmpData)

        ########################################################################
        ############################# FINAL PLOTS ##############################
        ########################################################################
        ### BAND 1 ###
        # Compute magnitudes and colors
        x1  = instMags1 - instMags2
        y1  = catMags1 - instMags1
        sx1 = np.sqrt(sigInstMags1**2 + sigInstMags2**2)
        sy1 = np.sqrt(sigCatMags1**2 + sigInstMags1**2)

        # Setup the orthogonal-distance-regression (ODR) for band 1
        data1 = odr.RealData(x1, y1, sx=sx1, sy=sy1)
        odr1  = odr.ODR(data1, lineModel, beta0=[0.0, np.median(y1)])
        out1  = odr1.run()

        # Plot the fitted data with its errorbars
        plt.errorbar(x1, y1, xerr=sx1, yerr=sy1,
            fmt='None', ecolor='k', zorder=999)
        plt.scatter(x1, y1, marker='s',
            c=post_prob1, cmap="gray_r", vmin=0, vmax=1, zorder=1000)
        # plt.scatter(x1[badInds1], y1[badInds1], marker='x', c='r', zorder=10001)

        # Overplot the fitted line
        xl = np.array(plt.xlim())
        if np.min(xl) > 0:
            xl = np.array([0, xl[1]])
            plt.xlim(xl)
        plt.autoscale(False)
        slope, intercept = out1.beta
        plt.plot(xl, slope*xl + intercept, c='k', lw=2.0)
        plt.xlabel(r'$V_{\rm inst}-R_{\rm inst}$')
        plt.ylabel(r'$V_{\rm cat}-V_{\rm inst}$')

        # Save the figure to disk
        fname = os.path.join(stokesDir, thisTarget + '_regression1.png')
        plt.savefig(fname, dpi=300)
        plt.close('all')

        ### BAND 2 ###
        # Compute magnitudes and colors
        x2  = instMags1 - instMags2
        y2  = catMags2 - instMags2
        sx2 = np.sqrt(sigInstMags1**2 + sigInstMags2**2)
        sy2 = np.sqrt(sigCatMags2**2 + sigInstMags2**2)

        # Setup the orthogonal-distance-regression (ODR) for band 2
        data2 = odr.RealData(x2, y2, sx=sx2, sy=sy2)
        odr2  = odr.ODR(data2, lineModel, beta0=[0.0, np.median(y2)])
        out2  = odr2.run()

        # Plot the fitted data with its errorbars
        plt.errorbar(x2, y2, xerr=sx2, yerr=sy2,
           fmt='None', ecolor='k', zorder=999)
        plt.scatter(x2, y2, marker='s',
            c=post_prob2, cmap="gray_r", vmin=0, vmax=1, zorder=1000)
        # plt.scatter(x2[badInds2], y2[badInds2], marker='x', c='r', zorder=10001)

        # Overplot the fitted line
        xl = np.array(plt.xlim())
        if np.min(xl) > 0:
            xl = np.array([0, xl[1]])
            plt.xlim(xl)
        plt.autoscale(False)
        slope, intercept = out2.beta
        plt.plot(xl, slope*xl + intercept, c='k', lw=2.0)
        plt.xlabel(r'$V_{\rm inst}-R_{\rm inst}$')
        plt.ylabel(r'$R_{\rm cat}-R_{\rm inst}$')

        # Save the figure to disk
        fname = os.path.join(stokesDir, thisTarget + '_regression2.png')
        plt.savefig(fname, dpi=300)
        plt.close('all')

        ### COLOR-COLOR ###
        # Plot the posteriors to see if a reasonable result was obtained.
        plotSamples = samplerC.flatchain[:,plotInds]
        plotBounds  = np.array(boundsC)[plotInds]
        corner.corner(plotSamples, bins=100,
            range=plotBounds,
            labels=plotLabels)

        # Save the figure to disk
        fname = os.path.join(stokesDir, thisTarget + '_MCMC.png')
        plt.savefig(fname, dpi=300)
        plt.close('all')

        # Setup the orthogonal-distance-regression (ODR) for band 1
        dataC = odr.RealData(xC, yC, sx=sxC, sy=syC)
        odrC  = odr.ODR(dataC, lineModel, beta0=[0.0, np.median(yC)])
        outC  = odrC.run()

        # Plot the fitted data with its errorbars
        plt.errorbar(xC, yC, xerr=sxC, yerr=syC,
            fmt='None', ecolor='k', zorder=999)
        plt.scatter(xC, yC, marker='s',
            c=post_probC, cmap="gray_r", vmin=0, vmax=1, zorder=1000)
        # plt.scatter(xC[badIndsC], yC[badIndsC], marker='x', c='r', zorder=10001)

        # Grab (and set) the x-axis boundaries
        xl = np.array(plt.xlim())
        if np.min(xl) > 0:
            xl = np.array([0, xl[1]])
            plt.xlim(xl)
        plt.autoscale(False)

        # Plot a sampling of acceptable MCMC parameters
        samples = samplerC.flatchain[:,0:2]
        for theta, b_perp in samples[np.random.randint(len(samples), size=1000)]:
            m, b = np.tan(theta), b_perp/np.cos(theta)
            plt.plot(xl, m*xl + b, color="k", alpha=0.025)
        slope, intercept = np.tan(truths[0]), truths[1]/np.cos(truths[0])
        plt.plot(xl, slope*xl + intercept,
            color='blue')

        # Overplot the ODR fit
        slope1, intercept1 = outC.beta
        plt.plot(xl, slope1*xl + intercept1,
            color='red', linestyle='dashed', linewidth=2.0)
        plt.xlabel(r'$V_{\rm inst}-R_{\rm inst}$')
        plt.ylabel(r'$V_{\rm cat}-R_{\rm cat}$')

        # Save the figure to disk
        fname = os.path.join(stokesDir, thisTarget + '_colorRegression.png')
        plt.savefig(fname, dpi=300)
        plt.close('all')


        ########################################################################
        print('...Photometric and Color transformations...\n')
        print(('\t({0}cal - {0}inst) = {3:.4g} + {2:.4g}*({0}inst - {1}inst)').format(
            band1, band2, *out1.beta))
        print(('\t({1}cal - {1}inst) = {3:.4g} + {2:.4g}*({0}inst - {1}inst)').format(
            band1, band2, *out2.beta))
        print(('\t({0}cal - {1}cal)  = {3:.4g} + {2:.4g}*({0}inst - {1}inst)').format(
            band1, band2, *truthsC))

        # Compute the photometrically calibrated images
        # Frist grab the airmass corrected images from the stokesIdict_AMC
        img1 = stokesIdict_AMC[band1]
        img2 = stokesIdict_AMC[band2]

        # Grab the U and Q images, too...
        Qimg1 = stokesQdict[band1]
        Qimg2 = stokesQdict[band2]
        Uimg1 = stokesUdict[band1]
        Uimg2 = stokesUdict[band2]

        # Compute the flux ratio, fix negative values, and convert to colors
        fluxRatio = img1/img2

        # First fix the indefinite values
        badPix = np.logical_not(np.isfinite(fluxRatio.arr))
        if np.sum(badPix) > 0:
            badInds = np.where(badPix)
            fluxRatio.arr[badInds] = 1e-6
            if hasattr(fluxRatio, 'sigma'):
                fluxRatio.sigma[badInds] = 1e-6

        # Then fix the negative values
        badPix = (fluxRatio.arr < 1e-6)
        if np.sum(badPix) > 0:
            badInds = np.where(badPix)
            fluxRatio.arr[badInds] = 1e-6
            if hasattr(fluxRatio, 'sigma'):
                fluxRatio.sigma[badInds] = 1e-6

        # Compute the instrumental color map and convert it to calibrated scale
        instColor = -2.5*np.log10(fluxRatio)

        # Use covariance matrix from MCMC process to estimate uncertainty in the
        # calibrated color map
        sig_m2    = covC[0,0]
        sig_b2    = covC[1,1]
        rhosmsb   = covC[0,1]
        sig_Color = np.sqrt(sig_m2*(instColor.arr**2) + sig_b2
            + 2*rhosmsb*instColor.arr + instColor.sigma**2)

        # Compute the calibrated color image and replace the simple uncertainty
        # with the full blown uncertainty...
        calColor       = instColor.copy()
        calColor.arr   = truthsC[1] + truthsC[0]*instColor.arr
        calColor.sigma = sig_Color

        # Now that the color-map has been computed, apply the calibration
        # transformations to the individual bands
        # First, compute the correction factors based on the regression above
        CF1_1     = 10.0**(-0.4*out1.beta[1])
        sig_CF1_1 = np.abs(CF1_1*0.4*np.log(10)*out1.sd_beta[1])
        CF1_2     = (fluxRatio)**out1.beta[0]
        CF2_1     = 10.0**(-0.4*out2.beta[1])
        sig_CF2_1 = np.abs(CF2_1*0.4*np.log(10)*out2.sd_beta[1])
        CF2_2     = (fluxRatio)**out1.beta[0]

        # Apply the color term to the actual image before writing to disk
        img1a = img1*CF1_2
        img2a = img2*CF2_2

        # Grab the pixel area to include in the linear scaling constants
        wcs1     = WCS(img1.header)
        wcs2     = WCS(img2.header)

        # Store the linear scaling constants in the "BSCALE" keyword
        BSCALE1  = zeroFlux[band1]*CF1_1
        BSCALE2  = zeroFlux[band2]*CF2_1
        SBSCALE1 = zeroFlux[band1]*sig_CF1_1
        SBSCALE2 = zeroFlux[band2]*sig_CF2_1

        # Describe the scaling offset and linear units
        BUNIT = 'Jy'
        BZERO = 0.0

        # Store the calibration data in the header
        img1a.header.set('BUNIT', value = BUNIT,
                   comment='Physical units for the image')
        img1a.header.set('BSCALE', value = BSCALE1, after='BUNIT',
                   comment='Conversion factor for physical units', )
        img1a.header.set('SBSCALE', value = SBSCALE1, after='BUNIT',
                   comment='Conversion factor for physical units', )
        img1a.header.set('BZERO', value = BZERO, after='BSCALE',
                   comment='Zero level for the physical units.')
        img2a.header.set('BUNIT', value = BUNIT,
                   comment='Physical units for the image')
        img2a.header.set('BSCALE', value = BSCALE2, after='BUNIT',
                   comment='Conversion factor for physical units', )
        img2a.header.set('SBSCALE', value = SBSCALE2, after='BUNIT',
                   comment='Conversion factor for physical units', )
        img2a.header.set('BZERO', value = BZERO, after='BSCALE',
                   comment='Zero level for the physical units.')

        # Finally write the updated image to disk
        img1aCalFile = '_'.join([thisTarget, band1, 'I', 'cal']) + '.fits'
        img1aCalFile = os.path.join(stokesDir, img1aCalFile)
        img1a.write(img1aCalFile)

        img2aCalFile = '_'.join([thisTarget, band2, 'I', 'cal']) + '.fits'
        img2aCalFile = os.path.join(stokesDir, img2aCalFile)
        img2a.write(img2aCalFile)

        Qimg1CalFile = '_'.join([thisTarget, band1, 'Q', 'cal']) + '.fits'
        Qimg1CalFile = os.path.join(stokesDir, Qimg1CalFile)
        Qimg1.write(Qimg1CalFile)

        Qimg2CalFile = '_'.join([thisTarget, band2, 'Q', 'cal']) + '.fits'
        Qimg2CalFile = os.path.join(stokesDir, Qimg2CalFile)
        Qimg2.write(Qimg2CalFile)

        Uimg1CalFile = '_'.join([thisTarget, band1, 'U', 'cal']) + '.fits'
        Uimg1CalFile = os.path.join(stokesDir, Uimg1CalFile)
        Uimg1.write(Uimg1CalFile)

        Uimg2CalFile = '_'.join([thisTarget, band2, 'U', 'cal']) + '.fits'
        Uimg2CalFile = os.path.join(stokesDir, Uimg2CalFile)
        Uimg2.write(Uimg2CalFile)

        colorFile = '_'.join([thisTarget, band1+'-'+band2]) + '.fits'
        colorFile = os.path.join(stokesDir, colorFile)
        calColor.write(colorFile)

print('Done!')
