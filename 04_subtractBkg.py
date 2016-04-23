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

# This script will compute an average "sky image" for each on-target time and
# subtract this image from the on-target image. Each subtracted image will be
# re-saved in pyPol_Data
#
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

# This dictionary specifies the angle range (degrees East of North) to include
# in producing radial-brightness proflies. This is especially for NGC2023
angleDict = {'NGC2023': (0, 360)}

# This is the location of the supersky images
bkgImagesDir = os.path.join(pyPol_data, 'bkgImages')

# Setup new directory for background subtracted data
bkgSubDir = os.path.join(pyPol_data, 'bkgSubtracted')
if (not os.path.isdir(bkgSubDir)):
    os.mkdir(bkgSubDir, 0o755)

# Read in Kokopelli mask generated in previous step
kokopelliMask = (AstroImage('kokopelliMask.fits').arr != 0)

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

# Check if the fileIndex already has a "Background" column.
# If not, then add one.
if 'Background' not in fileIndex.keys():
    fileIndex.add_column(Column(name='Background',
                                data = np.repeat(-99.9, len(fileIndex))))

# Determine which parts of the fileIndex pertain to science images
useFiles = np.where(fileIndex['Use'] == 1)
badFiles = np.where(fileIndex['Use'] == 0)

# Cull the file index to only include files selected for use
fileIndex1 = fileIndex[useFiles]
fileIndex2 = fileIndex[badFiles]

# Group the fileIndex by...
# 1. PPOL Name
# 3. Dither (pattern)

fileIndexByGroup = fileIndex1.group_by(['PPOL Name'])

# Loop through each grouping
for group in fileIndexByGroup.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisPPOLname = str(np.unique(group['PPOL Name'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])

    print('\nProcessing images for')
    print('\tPPOL Name : {0}'.format(thisPPOLname))
    print('\tWaveband  : {0}'.format(thisWaveband))
    print('')

    # Process images in this group according to dither type
    ditherType = (np.unique(group['Dither'].data))[0]
    numImgs    = len(group)
    if ditherType == "ABBA":
        # Grab the Ainds and Binds
        Ainds = (np.where(group['ABBA'].data == 'A'))[0]
        Binds = (np.where(group['ABBA'].data == 'B'))[0]

        # Parse A and B image entries in the group index
        Agroup = group[Ainds]
        Bgroup = group[Binds]

        # Grab the HWPs of each A and B image
        A_HWPs = Agroup['HWP']
        B_HWPs = Bgroup['HWP']

        # Now determine if this is a 16x(ABBA) or (16A,16B,16B,16A) group
        # Find where the HWP changes
        thisGroupHWPs = group['HWP'].data
        HWPshifts     = thisGroupHWPs != np.roll(thisGroupHWPs, 1)
        HWPshifts[0]  = 0
        if np.sum(HWPshifts) < 0.5*numImgs:
            # This is a 16x(ABBA) group
            # This observational procedure chops between A and B positions
            # quickly and rotates the HWP slowly.
            obsType = 'rapidChop'
        else:
            # This is a (16xA, 16xB, 16xB, 16xA) group
            # This observational procedure chops between and A and B positions
            # slowly and rotates the HWP quickly.
            obsType = 'slowChop'

        ########################################################################
        # Now we need to estimate the background of the on-target images.
        # We need to handle different observational procedures differently
        ########################################################################

        # Read in the A images
        AimgList = np.array([AstroImage(file1) for file1 in Agroup['Filename'].data])

        # Grab the timestamp and sky estimate for each image in the list
        AimgTimes = []
        AimgSky   = []
        for img in AimgList:
            # Grab the image timestamp
            thisDate = (img.header['DATE-OBS'])[0:10]
            thisDate = datetime.datetime.strptime(thisDate, "%Y-%m-%d")
            thisTime = (img.header['UTCSTART'])
            h, m, s  = thisTime.split(':')
            timeStp  = thisDate + datetime.timedelta(
                hours=int(h),
                minutes=int(m),
                seconds=(float(s) + 0.5*img.header['EXPTIME']))
            AimgTimes.append(timeStp.timestamp())

            # Grab the image sky value
            ny, nx = img.arr.shape
            subSampled = (img.arr[0:ny:10, 0:nx:10]).flatten()
            mean, median, stddev = sigma_clipped_stats(subSampled)
            AimgSky.append(mean)


        # Convert the image time and sky values to arrays
        AimgTimes = np.array(AimgTimes)
        AimgSky   = np.array(AimgSky)

        # Cleanup memory
        del AimgList

        # Read in the A images
        BimgList = np.array([AstroImage(file1) for file1 in Bgroup['Filename'].data])

        # Grab the timestamp for each image in the list
        BimgTimes = []
        BimgSky   = []
        for img in BimgList:
            # Grab the image timestamp
            thisDate = (img.header['DATE-OBS'])[0:10]
            thisDate = datetime.datetime.strptime(thisDate, "%Y-%m-%d")
            thisTime = (img.header['UTCSTART'])
            h, m, s  = thisTime.split(':')
            timeStp  = thisDate + datetime.timedelta(
                hours=int(h),
                minutes=int(m),
                seconds=(float(s) + 0.5*img.header['EXPTIME']))
            BimgTimes.append(timeStp.timestamp())

            # Grab the image sky value
            ny, nx = img.arr.shape
            subSampled = (img.arr[0:ny:10, 0:nx:10]).flatten()
            mean, median, stddev = sigma_clipped_stats(subSampled)
            BimgSky.append(mean)

        # Convert the image time and sky values to arrays
        BimgTimes = np.array(BimgTimes)
        BimgSky   = np.array(BimgSky)

        # Memory cleanup
        del BimgList

        # Store sky background levels in this dictionary.
        # Use estimated time-stamp as the keys
        skyLevelDict = dict()
        if obsType == 'rapidChop':
            # This is a 16x(ABBA) group
            ####################################################################
            # Test for FFT power at a cycle of 4 HWP rotations in the Bimgs
            ####################################################################
            # Start by subtracting the secular variation from the sky levels
            # First do an F-test justified polyfit
            # Start with a zeroth order polynomial
            numSamp    = BimgSky.size # total number of samples
            polyCoeffs = np.polyfit(BimgTimes, BimgSky, 0)
            polyValues = np.polyval(polyCoeffs, BimgTimes)
            chi2_m     = np.sum((BimgSky - polyValues)**2)

            ## Loop through polynomial degrees and store Ftest result
            alpha  = 0.05   #Use a 5% "random probability" as a cutoff
            # sigma  = 3.0    #Use a 3 sigma requirement for so much data
            # alpha  = (1 - norm.cdf(sigma))
            Ftests = []
            coeffs = []
            for deg in range(1,6):
                dof        = numSamp - deg - 1
                polyCoeffs = np.polyfit(BimgTimes, BimgSky, deg)
                coeffs.append(polyCoeffs)
                polyValues = np.polyval(polyCoeffs, BimgTimes)
                chi2_m1    = np.sum((BimgSky - polyValues)**2)
                Fchi       = (chi2_m - chi2_m1)/(chi2_m1/dof)
                prob       = 1.0 - f.cdf(Fchi, 1, dof)
                Ftests.append(prob < alpha)

                ## Store chi2_m1 in chi2 for use in next iteration
                chi2_m     = chi2_m1

            # Find the lowest order FAILED F-test to get higest order good fit
            bestDegree = np.min(np.where([not test for test in Ftests]))

            # Fit the best fitting polynomial
            polyCoeffs = np.polyfit(BimgTimes, BimgSky, bestDegree)
            polyValues = np.polyval(polyCoeffs, BimgTimes)

            # Subtract the best fitting polynomial and save for use in the FFT
            BimgSky1 = BimgSky - polyValues

            # Loop through unique HWP values and compute average sky levels
            B_HWP_for_fft = np.unique(B_HWPs)
            B_HWP_for_fft.sort()
            Bsky_for_fft  = []
            for hwp in B_HWP_for_fft:
                # For each HWP average the BimgSky values
                HWPinds = np.where(B_HWPs == hwp)

                # Append the average sky value
                Bsky_for_fft.append(np.mean(BimgSky1[HWPinds]))

            # Covert sky estimates to arrays
            Bsky_for_fft = np.array(Bsky_for_fft)

            ####################################################################
            # Do an FFT analysis to see if there is any power in the 4 rotations
            # Setup my sampling rates
            Fs = 16.0   # 16 samples per full HWP rotation
            Ts = 1.0/Fs # 1/16 of full HWP rotation per sample
            numRots = 1.0 # Number of full HWP rotatios in signal
            fracRot = np.arange(0, numRots, Ts) # fractional rotation of HWP
            numSamp = Bsky_for_fft.size # total number of samples

            # Setup wavenumber and frequency vectors
            k = np.arange(numSamp) # Generate vector of wavenumbers
            T = numSamp/Fs # Total range of samples (1 full HWP rotation)
            frq = k/T # two sides frequency range (*** NOT REALLY TWO SIDES? ***)
            frq = frq[0:np.int(np.floor(numSamp/2))] # one side frequency range

            BskyFFT = np.fft.fft(Bsky_for_fft)/numSamp # fft computing and normalization
            BskyFFT = BskyFFT[0:np.int(np.floor(numSamp/2))] # Truncate left side

            ###
            ###
            # TODO Figure out how to (if to) use the FFT results
            ###
            ###
            # # Plot results
            # plt.ion()
            #
            # fig, ax = plt.subplots(3, 1, figsize=(8, 10))
            # ax[0].plot(BimgTimes, BimgSky, '-bo')
            # ax[0].plot(BimgTimes, polyValues)
            # ax[0].set_xlabel('Time')
            # ax[0].set_ylabel('Sky Brightness')
            # ax[1].plot(fracRot, Bsky_for_fft)
            # ax[1].set_xlabel('Fractional HWP Rotation')
            # ax[1].set_ylabel('Mean Sky Brightness')
            # ax[2].plot(frq, abs(BskyFFT), 'r') # plotting the FFT
            # ax[2].set_xlabel('Freq (HWP #)')
            # ax[2].set_ylabel('|FFT(freq)|')
            # pdb.set_trace()
            # plt.close('all')
            # continue

            # Store the final (perhaps FFT modified) sky levels in dictionary
            for time, sky in zip(BimgTimes, BimgSky):
                skyLevelDict[time] = sky

        ########################################################################
        # The other kind of group is the "slow chop" group... handle those here.
        ########################################################################
        elif obsType == 'slowChop':
            ####################################################################
            # Construct the 2MASS radial profile
            # Test this with the M78_H image
            TMASSfile = os.path.join(TMASSdir, '_'.join([thisTarget, thisWaveband]) + '.fits')
            TMASSimg  = AstroImage(TMASSfile)


            TMASSfile = os.path.join(TMASSdir, '_'.join([thisTarget, thisWaveband]) + '.fits')
            TMASSimg  = AstroImage(TMASSfile)

            PSFproperties  = TMASSimg.get_PSF()
            TMASS_PSFwidth = np.sqrt(PSFproperties[0]*PSFproperties[1])

            # Grab image dimensions
            ny, nx     = TMASSimg.arr.shape
            yc, xc     = np.array(TMASSimg.arr.shape)//2

            # Grab the image center coordinates
            TMASSwcs = WCS(TMASSimg.header)
            RA_cen, Dec_cen = TMASSwcs.wcs_pix2world(xc, yc, 0)
            TMASS_pl_sc = proj_plane_pixel_scales(TMASSwcs)
            TMASS_pl_sc = np.sqrt(TMASS_pl_sc[0]*TMASS_pl_sc[1])*3600.0

            # Compute distances from image center
            yy, xx = np.mgrid[0:ny, 0:nx]
            dist = np.sqrt((xx - xc)**2 + (yy - yc)**2)
            dist *= TMASS_pl_sc

            # Check if there are specified "good angles" for this target. If
            # there are, then build a mask to make sure radial profiles are
            # constructed using only user-specified regions of the nebula.
            if thisTarget in angleDict.keys():
                # This target has a specified "good angle" range.
                TMASS_angleMap = (np.rad2deg(np.arctan2((yy - yc), (xx - xc)))
                    - 90.0 + 360.0) % 360.0
                minAng, maxAng = np.min(angleDict[thisTarget]), np.max(angleDict[thisTarget])
                TMASS_mask = np.logical_and(TMASS_angleMap > minAng,
                                            TMASS_angleMap < maxAng)
                # Set all masked points in the distance mask to "zero"
                dist *= TMASS_mask.astype(float)
                dist  = (dist.round()).astype(int)
            else:
                # TMASS_mask = np.ones_like(TMASSimg.arr, dtype=bool)
                dist = (dist.round()).astype(int)

            # Compute maximum and minimum valid distances
            minRad = np.min(dist)
            maxRad = np.max(dist)

            fluxDict = dict()
            for rad in range(10, maxRad):
                thesePix = np.where(dist == rad)
                if len(thesePix[0]) > 100:
                    theseVals = TMASSimg.arr[thesePix]
                    # Compute sigma-clipped-stats
                    mean, median, stddev = sigma_clipped_stats(theseVals)
                    # Grab the pixels within 3-sigma of the median
                    thesePix1  = np.where(np.abs(theseVals - median) < 3.0*stddev)
                    theseVals1 = theseVals[thesePix1]
                    # Recompute the mean and uncertainty
                    thisMean = np.mean(theseVals1)
                    thisSig  = np.std(theseVals1)/np.sqrt(theseVals.size - 1)
                    if thisMean > 0:
                        fluxDict[rad] = (thisMean, thisSig)

            # Separate the dictionary keys and values
            TMASS_dist, TMASS_flux1 = fluxDict.keys(), fluxDict.values()
            TMASS_dist  = np.array([key for key in TMASS_dist])
            TMASS_flux  = np.array([val for val, sig in TMASS_flux1])
            TMASS_sigma = np.array([sig for val, sig in TMASS_flux1])

            ####################################################################
            # Now do a radial profile from an on-target Mimir image
            ####################################################################
            # Grab the HWP values and the shift values
            A_HWPs = thisGroupHWPs[Ainds]
            B_HWPs = thisGroupHWPs[Binds]

            # Loop through each A_HWP and do a background level analysis
            completedFlags = np.zeros_like(A_HWPs, dtype=bool)
            for HWP1_ind, HWP1 in enumerate(A_HWPs):
                # First check if this HWP image has already been handled
                # Skip the image if it has already been handled
                if completedFlags[HWP1_ind] == True: continue

                # Steup the proper HWP2 value (Should always be two up)
                HWP2 = HWP1 + 2

                # Check if this combination of HWP images will work
                if (HWP1_ind >= (A_HWPs.size - 1)): continue
                if (A_HWPs[HWP1_ind + 1] != HWP2):
                    # If the combination will not work, then find the best
                    # possible alternate combination
                    # Begin by grabbing all possibe second image combinations
                    HWP2_candidates = ((HWP2 + 4*np.arange(4) - 1) % 16) + 1

                    # Loop through all candidates and get their indices
                    HWP2_imgs = np.zeros_like(A_HWPs, dtype=bool)
                    for HWP2_cand in HWP2_candidates:
                        # Add these HWP images to the list
                        HWP2_imgs = np.logical_or(HWP2_imgs, A_HWPs == HWP2_cand)

                    # If some candidates were found, then get their indices
                    if np.sum(HWP2_imgs) > 0:
                        HWP2_inds = (np.where(HWP2_imgs))[0]
                    else:
                        print('There are NO possible image combinations for HWP {0}'.format(HWP1))
                        pdb.set_trace()

                    # Select the candidate closest in time!
                    HWP1_time  = AimgTimes[HWP1_ind]
                    HWP2_times = AimgTimes[HWP2_inds]
                    deltaTime  = np.abs(HWP2_times - HWP1_time)
                    HWP2_best  = (np.where(deltaTime == np.min(deltaTime)))[0][0]

                    # Grab the index if the HWP2 winner!
                    HWP2_ind = HWP2_inds[HWP2_best]
                    HWP2     = A_HWPs[HWP2_ind]

                    # Only mark the HWP1 image as "completed" since the HWP2
                    # image is a bit odd-ball
                    completedFlags[HWP1_ind] = True
                else:
                    # If the image combination will work, then mark both images
                    # as "completed"
                    HWP2_ind = HWP1_ind + 1
                    completedFlags[HWP1_ind] = True
                    completedFlags[HWP2_ind] = True

                # Update the user on progress
                print('\tProcessing HWP pair ({0}, {1})'.format(HWP1, HWP2))

                # Now actually build the on-target Stokes Image
                HWP1_img  = AstroImage(Agroup['Filename'][HWP1_ind])
                HWP2_img  = AstroImage(Agroup['Filename'][HWP2_ind])
                imgList   = HWP1_img.align(HWP2_img, mode = 'cross_correlate')
                # imgList   = HWP1_img.align(HWP2_img, mode = 'wcs')
                StokesImg = 0.5*(imgList[0] + imgList[1])

                # Get the PSF size of the Mimir image
                PSFproperties  = StokesImg.get_PSF()
                Mimir_PSFwidth = np.sqrt(PSFproperties[0]*PSFproperties[1])

                # Compute coordinates of each pixel
                MimirWCS    = WCS(StokesImg.header)
                Xcen, Ycen  = MimirWCS.wcs_world2pix(RA_cen, Dec_cen, 0)
                Mimir_pl_sc = proj_plane_pixel_scales(MimirWCS)
                Mimir_pl_sc = np.sqrt(Mimir_pl_sc[0]*Mimir_pl_sc[1])*3600.0

                # Compute distances from nebula center
                ny, nx = StokesImg.arr.shape
                yy, xx = np.mgrid[0:ny, 0:nx]
                dist   = np.sqrt((xx - Xcen)**2 + (yy - Ycen)**2)
                dist  *= Mimir_pl_sc

                # Check if there are specified "good angles" for this target. If
                # there are, then build a mask to make sure radial profiles are
                # constructed using only user-specified regions of the nebula.
                if thisTarget in angleDict.keys():
                    # This target has a specified "good angle" range.
                    Mimir_angleMap = (np.rad2deg(np.arctan2((yy - Ycen), (xx - Ycen)))
                        - 90.0 + 360.0) % 360.0
                    minAng, maxAng = np.min(angleDict[thisTarget]), np.max(angleDict[thisTarget])
                    Mimir_mask = np.logical_and(Mimir_angleMap > minAng,
                                                Mimir_angleMap < maxAng)
                    # Set all masked points in the distance mask to "zero"
                    dist *= Mimir_mask.astype(float)
                    dist  = (dist.round()).astype(int)
                else:
                    # TMASS_mask = np.ones_like(TMASSimg.arr, dtype=bool)
                    dist = (dist.round()).astype(int)

                # Create a flux dictionary for the Mimir image
                fluxDict = dict()
                for rad in range(5, maxRad + 1):
                    thesePix = np.where(dist == rad)
                    if len(thesePix[0]) > 50:
                        theseVals = StokesImg.arr[thesePix]
                        # Compute sigma-clipped-stats
                        mean, median, stddev = sigma_clipped_stats(theseVals)
                        # Grab the pixels within 3-sigma of the median
                        thesePix1  = np.where(np.abs(theseVals - median) < 3.0*stddev)
                        theseVals1 = theseVals[thesePix1]
                        # Recompute the mean and uncertainty
                        thisMean = np.mean(theseVals1)
                        thisSig  = np.std(theseVals1)/np.sqrt(theseVals.size - 1)
                        if thisMean > 0:
                            fluxDict[rad] = (thisMean, thisSig)

                # Separate the dictionary keys and values
                Mimir_onDist, Mimir_onFlux1 = fluxDict.keys(), fluxDict.values()
                Mimir_onDist  = np.array([key for key in Mimir_onDist])
                Mimir_onFlux  = np.array([val for val, sig in Mimir_onFlux1])
                Mimir_onSigma = np.array([sig for val, sig in Mimir_onFlux1])

                ####################################################################
                # Now do a radial profile for an off-target Mimir image
                ####################################################################
                # Grab the B images corresponding to the HWP images used for the
                # on-target Stokes image. Begin by looking for exact HWP matches
                HWP1_inds = (np.where(B_HWPs == HWP1))[0]

                # Look for the best candidate to match this HWP
                # If there is not a perfect HWP match, then look for other
                # possible matches.
                if len(HWP1_inds) == 0:
                    # Begin by grabbing all possibe replacement HWP images
                    HWP1_candidates = ((HWP1 + 4*np.arange(4) - 1) % 16) + 1

                    # Loop through all candidates and get their indices
                    HWP1_imgs = np.zeros_like(B_HWPs, dtype=bool)
                    for HWP1_cand in HWP1_candidates:
                        # Add these HWP images to the list
                        HWP1_imgs = np.logical_or(HWP1_imgs, B_HWPs == HWP1_cand)

                    # If some candidates were found, then get their indices
                    if np.sum(HWP1_imgs) > 0:
                        HWP1_inds = (np.where(HWP1_imgs))[0]
                    else:
                        print('There are NO possible image combinations for HWP {0}'.format(HWP1))
                        pdb.set_trace()

                # From all possible HWP matches, select the one closest in time
                # to the previously selected on-target image
                Atime     = AimgTimes[HWP1_ind]
                Btimes    = BimgTimes[HWP1_inds]
                deltaTime = np.abs(Atime - Btimes)
                HWP1_best = (np.where(deltaTime == np.min(deltaTime)))[0][0]

                # Store the index and HWP of the selection
                HWP1_indB = HWP1_inds[HWP1_best]
                HWP1_B    = B_HWPs[HWP1_indB]

                # Repeat this process for the HWP2 pairing
                HWP2_inds = (np.where(B_HWPs == HWP2))[0]

                # Look for the best candidate to match this HWP
                # If there is not a perfect HWP match, then look for other
                # possible matches.
                if len(HWP2_inds) == 0:
                    # Begin by grabbing all possibe replacement HWP images
                    HWP2_candidates = ((HWP2 + 4*np.arange(4) - 1) % 16) + 1

                    # Loop through all candidates and get their indices
                    HWP2_imgs = np.zeros_like(B_HWPs, dtype=bool)
                    for HWP2_cand in HWP2_candidates:
                        # Add these HWP images to the list
                        HWP2_imgs = np.logical_or(HWP2_imgs, B_HWPs == HWP2_cand)

                    # If some candidates were found, then get their indices
                    if np.sum(HWP2_imgs) > 0:
                        HWP1_inds = (np.where(HWP2_imgs))[0]
                    else:
                        print('There are NO possible image combinations for HWP {0}'.format(HWP1))
                        pdb.set_trace()

                # From all possible HWP matches, select the one closest in time
                # to the previously selected on-target image
                Atime     = AimgTimes[HWP2_ind]
                Btimes    = BimgTimes[HWP2_inds]
                deltaTime = np.abs(Atime - Btimes)
                HWP2_best = (np.where(deltaTime == np.min(deltaTime)))[0][0]

                # Store the index and HWP of the selection
                HWP2_indB = HWP2_inds[HWP2_best]
                HWP2_B    = B_HWPs[HWP2_indB]

                # Finally build the Stokes image
                HWP1_imgB = AstroImage(Bgroup['Filename'][HWP1_indB])
                HWP2_imgB = AstroImage(Bgroup['Filename'][HWP2_indB])
                StokesImg = 0.5*(HWP1_imgB + HWP2_imgB)

                # Compute distances from image center
                # (using same position as on-target radial profile)
                ny, nx = StokesImg.arr.shape
                yy, xx = np.mgrid[0:ny, 0:nx]
                dist = np.sqrt((xx - Xcen)**2 + (yy - Ycen)**2)
                dist *= Mimir_pl_sc

                # Check if there are specified "good angles" for this target. If
                # there are, then build a mask to make sure radial profiles are
                # constructed using only user-specified regions of the nebula.
                if thisTarget in angleDict.keys():
                    # This target has a specified "good angle" range.
                    Mimir_angleMap = (np.rad2deg(np.arctan2((yy - Ycen), (xx - Ycen)))
                        - 90.0 + 360.0) % 360.0
                    minAng, maxAng = np.min(angleDict[thisTarget]), np.max(angleDict[thisTarget])
                    Mimir_mask = np.logical_and(Mimir_angleMap > minAng,
                                                Mimir_angleMap < maxAng)
                    # Set all masked points in the distance mask to "zero"
                    dist *= Mimir_mask.astype(float)
                    dist  = (dist.round()).astype(int)
                else:
                    # TMASS_mask = np.ones_like(TMASSimg.arr, dtype=bool)
                    dist = (dist.round()).astype(int)

                # Create a flux dictionary for the Mimir image
                Mimir_onDist1  = list()
                Mimir_onFlux1  = list()
                Mimir_onSigma1 = list()
                fluxDict = dict()
                for i, rad in enumerate(Mimir_onDist):
                    thesePix = np.where(dist == rad)

                    # Don't reject distances... we need every distance in the
                    # Miir_on and Mimir_off vectors to match
                    theseVals = StokesImg.arr[thesePix]
                    # Compute sigma-clipped-stats
                    mean, median, stddev = sigma_clipped_stats(theseVals)
                    # Grab the pixels within 3-sigma of the median
                    thesePix1  = np.where(np.abs(theseVals - median) < 3.0*stddev)
                    theseVals1 = theseVals[thesePix1]
                    # Check if there are a large number of samples to work with
                    if len(thesePix1[0]) > 100:
                        # Recompute the mean and uncertainty
                        thisMean = np.mean(theseVals1)
                        thisSig  = np.std(theseVals1)/np.sqrt(theseVals.size - 1)
                        if thisMean > 0:
                            # Store the off-target flux and uncertainty
                            fluxDict[rad] = (thisMean, thisSig)
                            # Store the on-target flux and uncertainty, too
                            Mimir_onDist1.append(rad)
                            Mimir_onFlux1.append(Mimir_onFlux[i])
                            Mimir_onSigma1.append(Mimir_onSigma[i])

                # Convert new Mimir on-target lists to arrays
                Mimir_onDist  = np.array(Mimir_onDist1)
                Mimir_onFlux  = np.array(Mimir_onFlux1)
                Mimir_onSigma = np.array(Mimir_onSigma1)

                # Cleanup temporary lists to save on memory
                del Mimir_onDist1, Mimir_onFlux1, Mimir_onSigma1

                # Separate the dictionary keys and values
                Mimir_offDist, Mimir_offFlux1 = fluxDict.keys(), fluxDict.values()
                Mimir_offDist  = np.array([key for key in Mimir_offDist])
                Mimir_offFlux  = np.array([val for val, sig in Mimir_offFlux1])
                Mimir_offSigma = np.array([sig for val, sig in Mimir_offFlux1])

                # Compute normalized non-nebular radial profile
                median_Mimir_offFlux      = np.median(Mimir_offFlux)
                normalized_Mimir_offFlux  = Mimir_offFlux / median_Mimir_offFlux
                normalized_Mimir_offSigma = Mimir_offSigma / median_Mimir_offFlux

                # Divide by normalized radial profile to get a flattened Mimir
                # nebular flux radial profile
                flattened_Mimir_onFlux  = Mimir_onFlux / normalized_Mimir_offFlux
                flattened_Mimir_onSigma = (np.abs(flattened_Mimir_onFlux) * np.sqrt(
                    (Mimir_onSigma/Mimir_onFlux)**2 +
                    (normalized_Mimir_offSigma/normalized_Mimir_offFlux)**2))

                # Smooth Mimir flux vector to math 2MASS PSF size
                if TMASS_PSFwidth > Mimir_PSFwidth:
                    smoothWidth = TMASS_PSFwidth - Mimir_PSFwidth
                    flattened_Mimir_onFlux = gaussian_filter1d(flattened_Mimir_onFlux, smoothWidth)
                elif TMASS_PSFwidth < Mimir_PSFwidth:
                    smoothWidth = Mimir_PSFwidth - TMASS_PSFwidth
                    TMASS_flux = gaussian_filter1d(TMASS_flux, smoothWidth)

                # Use the following linear function for fitting slopes to data.
                # This model will be used to test for increasing, decreasing,
                # or flat slopes in the flux vs. distance curve.
                # Set up ODR with the model and data.
                lineFunc = lambda B, x: B[0]*x + B[1]
                lineModel = Model(lineFunc)


                # Perform the ODR fitting for all bins in Mimir profile. Use
                # this to check if there are any "ramp-ups" in flux far from
                # the image center.
                distSize = Mimir_onDist.size
                for iStart in range(distSize-55, 0, -1):
                    thisDist  = Mimir_onDist[(iStart + 10*5):iStart:-5]
                    thisFlux  = flattened_Mimir_onFlux[(iStart + 10*5):iStart:-5]
                    thisSigma = flattened_Mimir_onSigma[(iStart + 10*5):iStart:-5]
                    data      = RealData(thisDist, thisFlux, sy = thisSigma)
                    odr       = ODR(data, lineModel, beta0=[0.0, 0.0])
                    lineFit   = odr.run()

                    # Check the slope and significance of the slope
                    if (np.abs(lineFit.beta[0]) / lineFit.sd_beta[0]) < 3.0:
                        bkgStart = Mimir_onDist[iStart + 5*5]
                        break

                # Cull the flattened_Mimir_onFlux to only include distances
                # that do not include any "ramp up" flux
                goodInds = np.where(Mimir_onDist < bkgStart)
                Mimir_onDist            = Mimir_onDist[goodInds]
                flattened_Mimir_onFlux  = flattened_Mimir_onFlux[goodInds]
                flattened_Mimir_onSigma = flattened_Mimir_onSigma[goodInds]

                # Now we need to find the background level in the 2MASS images
                # Fit these distance and flux values with a line
                # Perform the ODR fitting for all bins in TMASS profile
                for iStart in range(len(TMASS_dist)):
                    if TMASS_dist[iStart] < 200: continue
                    thisDist  = TMASS_dist[iStart:(iStart + 10*5):5]
                    thisFlux  = TMASS_flux[iStart:(iStart + 10*5):5]
                    thisSigma = TMASS_sigma[iStart:(iStart + 10*5):5]
                    data      = RealData(thisDist, thisFlux, sy = thisSigma)
                    odr       = ODR(data, lineModel, beta0=[0.0, 0.0])
                    lineFit   = odr.run()

                    # Check the slope and significance of the slope
                    if (np.abs(lineFit.beta[0]) / lineFit.sd_beta[0]) < 1.0:
                        bkgStart = TMASS_dist[iStart]
                        break

                # Estimate the 2MASS background level and subtract from 2MASS flux
                # Background is computed for a 50 arcsec annulus starting 50 arcsec
                # beyond the previously identified "start point"
                bkgInds = np.where(np.logical_and(TMASS_dist > (bkgStart + 50),
                                                  TMASS_dist < (bkgStart + 100)))
                TMASS_bkg              = np.mean(TMASS_flux[bkgInds])
                TMASS_bkg_sigma        = np.std(TMASS_flux[bkgInds])/np.sqrt(bkgInds[0].size - 1)
                subtracted_TMASS_flux  = TMASS_flux - TMASS_bkg
                subtracted_TMASS_sigma = np.sqrt(TMASS_sigma**2 + TMASS_bkg_sigma**2)

                # Select the fluxes at distances in both 2MASS and Mimir profiles
                common_dist        = list()
                common_TMASS_flux  = list()
                common_TMASS_sigma = list()
                common_Mimir_flux  = list()
                common_Mimir_sigma = list()
                # Loop through all distances in the 2MASS profile
                for i, dist in enumerate(TMASS_dist):
                    # If the distance is also sampled in the Mimir profile,
                    # then store the distance and both 2MASS and Mimir fluxes
                    if dist in Mimir_onDist:
                        common_dist.append(dist)
                        common_TMASS_flux.append(subtracted_TMASS_flux[i])
                        common_TMASS_sigma.append(subtracted_TMASS_sigma[i])

                        # Grab the corresponding index for the 2MASS distance
                        iDist2 = np.where(Mimir_onDist == dist)
                        common_Mimir_flux.append(flattened_Mimir_onFlux[iDist2])
                        common_Mimir_sigma.append(flattened_Mimir_onSigma[iDist2])

                # Convert these lists to arrays
                common_dist        = np.array(common_dist).flatten()
                common_TMASS_flux  = np.array(common_TMASS_flux).flatten()
                common_TMASS_sigma = np.array(common_TMASS_sigma).flatten()
                common_Mimir_flux  = np.array(common_Mimir_flux).flatten()
                common_Mimir_sigma = np.array(common_Mimir_sigma).flatten()

                # Solve for fitting scale factor and background subtraction
                def errFunc1(B):
                    # B[0] = scaling factor to convert Mimir flux to 2MASS flux
                    # B[1] = Mimir background level to subtract
                    Mimir_flux1 = B[0]*(common_Mimir_flux - B[1])
                    thisAbsDiff = (common_TMASS_flux - Mimir_flux1)**2
                    thisDiffVar = (common_TMASS_sigma**2 + (B[0]*common_Mimir_sigma)**2)
                    return np.sum(thisAbsDiff/thisDiffVar)

                fitResult1 = minimize(errFunc1,
                    (0.1, np.median(common_Mimir_flux)),
                    method='Nelder-Mead')

                # Grab the scale factor and Mimir background
                # (This background level is the number you've been looking for!)
                scaleFactor, MimirBkg = fitResult1.x

                # Compute the subtracted, scaled Mimir flux
                Mimir_flux = scaleFactor*(common_Mimir_flux - MimirBkg)

                # Now compute a radially varrying transmission flux
                def errFunc2(B):
                    # B[0] = Slope for "flattening line" on subtract, scaled Mimir flux
                    # B[1] = Intercept for "flatting line" on subtracted, scaled Mimir flux
                    flatLine    = B[0]*common_dist + B[1]
                    Mimir_flux1 = Mimir_flux / flatLine
                    thisAbsDiff = (common_TMASS_flux - Mimir_flux1)**2
                    thisDiffVar = (common_TMASS_sigma**2 + (common_Mimir_sigma)**2)
                    return np.sum(thisAbsDiff/thisDiffVar)

                fitResult2 = minimize(errFunc2,
                    (0.0, 0.0),
                    method='Nelder-Mead')

                # Grab the "flatt-line" slope and intercept and compute a
                # "flattening line"
                flatSlope, flatIntercept = fitResult2.x
                flatLine = flatSlope*common_dist + flatIntercept

                # Solve for fitting scale factor and background subtraction.
                # Include variable transmission in fitting procedure.
                def errFunc3(B):
                    # B[0] = scaling factor to convert Mimir flux to 2MASS flux
                    # B[1] = Mimir background level to subtract
                    Mimir_flux1 = B[0]*(common_Mimir_flux - B[1])
                    Mimir_flux2 = Mimir_flux1 / flatLine
                    thisAbsDiff = (common_TMASS_flux - Mimir_flux2)**2
                    thisDiffVar = (common_TMASS_sigma**2 + (B[0]*common_Mimir_sigma)**2)
                    return np.sum(thisAbsDiff/thisDiffVar)

                # Perform the actual fit
                fitResult3 = minimize(errFunc3,
                    (0.1, np.median(common_Mimir_flux)),
                    method='Nelder-Mead')

                # Grab the scale factor and MimirBkg
                scaleFactor, MimirBkg = fitResult3.x

                # # Plot to see that everything is working correctly.
                # Mimir_flux1 = (scaleFactor*(common_Mimir_flux - MimirBkg)) / flatLine
                # plt.ion()
                # plt.plot(common_dist, common_TMASS_flux,
                #          common_dist, Mimir_flux1)
                # plt.xlabel('Radial Distance from Center (arcsec)')
                # plt.ylabel('Scaled Counts')
                # pdb.set_trace()
                # plt.close('all')

                # Store the sky levels in dictionary
                thisTime = 0.5*(AimgTimes[HWP1_ind] + AimgTimes[HWP2_ind])
                skyLevelDict[thisTime] = MimirBkg

        # Interpolate sky levels to the Aimg times
        # Unpack the sky level dictionary and sort them
        skyTimes  = np.array([time for time in skyLevelDict.keys()])
        skyValues = np.array([sky for sky in skyLevelDict.values()])
        sortInds  = skyTimes.argsort()
        skyTimes  = skyTimes[sortInds]
        skyValues = skyValues[sortInds]

        # Grab the earliest time for the group and recompute image times
        # relative to that earliest image
        time0 = np.min(np.concatenate((AimgTimes, BimgTimes, skyTimes)))
        AimgTimes -= time0
        BimgTimes -= time0
        skyTimes  -= time0

        # Interpolate off-target background volues to on-target times
        # Loop through the on-target images (usually A-images) and extrapolate
        # or interpolate the expected sky brightness at Atime for each image.
        interpSkyTimes  = []
        interpSkyValues = []
        for imgNum, Atime in enumerate(AimgTimes):
            ################################################################
            #################### Interpolate Asky image ####################
            ################################################################
            # # Select the time and HWP for this img
            # Atime, A_HWP = AimgTimes[imgNum],  Agroup['HWP'].data[imgNum]

            # Figure out which B images are before or after the current Aimg.
            # Images near the current Aimg time will be used to interpolate
            # an Asky value.
            skyTimesBefore = ((skyTimes - Atime) < 0)
            skyTimesAfter  = ((skyTimes - Atime) > 0)
            numBefore      = np.sum(skyTimesBefore)
            numAfter       = np.sum(skyTimesAfter)

            # Skip any images which are NOT bracketed by a background estimate
            if (numBefore == 0) or (numAfter == 0): continue

            # There are some images before and after, so we can interpolate
            # between the estimated sky values
            beforeInd = (np.where(skyTimesBefore)[0])[numBefore-1:numBefore]
            afterInd  = (np.where(skyTimesAfter)[0])[0:1]
            skyInds  = np.concatenate((beforeInd, afterInd))

            # Grab the Bigs, Bsky, and BimgTimes values for the selected images
            thisBsky   = skyValues[skyInds]
            thisBtimes = skyTimes[skyInds]

            # Perform the interpolation/extrapolation to the expected sky value
            thisSlope = (thisBsky[1] - thisBsky[0])/(thisBtimes[1] - thisBtimes[0])
            thisIntercept = thisBsky[0] - thisSlope*thisBtimes[0]
            thisSky = thisSlope*Atime + thisIntercept

            # Store the interpolated sky value
            interpSkyTimes.append(Atime)
            interpSkyValues.append(thisSky)

        # Convert the sky values list into an array
        interpSkyTimes  = np.array(interpSkyTimes)
        interpSkyValues = np.array(interpSkyValues)

        # if obsType == 'rapidChop':
        #     # Handle the rapid-chop interpolation for comparison
        #     skyTimes1  = np.array([time for time in skyLevelDict1.keys()])
        #     skyValues1 = np.array([sky for sky in skyLevelDict1.values()])
        #     sortInds   = skyTimes1.argsort()
        #     skyTimes1  = skyTimes1[sortInds]
        #     skyValues1 = skyValues1[sortInds]
        #
        #     # Subtract to the same initial time
        #     skyTimes1 -= time0
        #
        #     # Do the interpolation
        #     interpSkyTimes1  = []
        #     interpSkyValues1 = []
        #
        #     for imgNum, Atime in enumerate(AimgTimes):
        #         ################################################################
        #         #################### Interpolate Asky image ####################
        #         ################################################################
        #         # # Select the time and HWP for this img
        #         # Atime, A_HWP = AimgTimes[imgNum],  Agroup['HWP'].data[imgNum]
        #
        #         # Figure out which B images are before or after the current Aimg.
        #         # Images near the current Aimg time will be used to interpolate
        #         # an Asky value.
        #         skyTimesBefore = ((skyTimes1 - Atime) < 0)
        #         skyTimesAfter  = ((skyTimes1 - Atime) > 0)
        #         numBefore      = np.sum(skyTimesBefore)
        #         numAfter       = np.sum(skyTimesAfter)
        #
        #         # Skip any images which are NOT bracketed by a background estimate
        #         if (numBefore == 0) or (numAfter == 0): continue
        #
        #         # There are some images before and after, so we can interpolate
        #         # between the estimated sky values
        #         beforeInd = (np.where(skyTimesBefore)[0])[numBefore-1:numBefore]
        #         afterInd  = (np.where(skyTimesAfter)[0])[0:1]
        #         skyInds   = np.concatenate((beforeInd, afterInd))
        #
        #         # Grab the Bigs, Bsky, and BimgTimes values for the selected images
        #         thisBsky   = skyValues1[skyInds]
        #         thisBtimes = skyTimes1[skyInds]
        #
        #         # Perform the interpolation/extrapolation to the expected sky value
        #         thisSlope = (thisBsky[1] - thisBsky[0])/(thisBtimes[1] - thisBtimes[0])
        #         thisIntercept = thisBsky[0] - thisSlope*thisBtimes[0]
        #         thisSky = thisSlope*Atime + thisIntercept
        #
        #         # Store the interpolated sky value
        #         interpSkyTimes1.append(Atime)
        #         interpSkyValues1.append(thisSky)
        #
        #     # Convert the lists to arrays
        #     interpSkyTimes1 = np.array(interpSkyTimes1)
        #     interpSkyValues1 = np.array(interpSkyValues1)

        # # Do a quick plotup for Dan
        # plt.ion()
        # fig = plt.figure()
        # ax  = fig.add_subplot(1,1,1)
        #
        # if obsType == 'slowChop':
        #     ax.plot(BimgTimes, BimgSky, '-ro', label='Off-target')
        #     ax.plot(interpSkyTimes1, interpSkyValues1, 'kx')
        # elif obsType == 'rapidChop':
        #     ax.plot(skyTimes, skyValues, '-bo', label='Profile-fitting')
        #     ax.plot(interpSkyTimes, interpSkyValues, 'kx')
        #
        # ax.plot(AimgTimes, AimgSky, '-go', label='On-target')
        # ax.set_xlabel('Time (sec)')
        # ax.set_ylabel('Sky Background (ADU)')
        # plt.legend()
        # plt.title(thisPPOLname)
        # #
        # #
        # # ax.plot(skyTimes, skyValues, 'r')
        # # ax.plot(skyTimes, skyValues, 'ko')
        #
        #
        # pdb.set_trace()
        # plt.close()

    # Now that the sky background levels have been determined, use those
    # background levels to scale the estimated "sky-image" and subtract
    # background from the on-target images.
    for thisTime, thisSky in zip(interpSkyTimes, interpSkyValues):
        # Find the AimgTime closest to THIS TIME
        deltaTime = np.abs(AimgTimes - thisTime)
        AgroupInd = (np.where(deltaTime == np.min(deltaTime)))[0][0]

        # Read in the Aimg associated with this time
        AimgFile = Agroup['Filename'].data[AgroupInd]
        Aimg     = AstroImage(AimgFile)

        # Find the location in fileIndex1 with this filename
        indexInd = np.where(fileIndex1['Filename'].data == AimgFile)

        if len(indexInd[0]) == 1:
            fileIndex1['Background'][indexInd] = thisSky
        else:
            print('Filename is not unique?! That is strange.')
            pdb.set_trace()

        # Grab the HWP for this image
        A_HWP = Agroup['HWP'].data[AgroupInd]

        # Construct the path to the background image
        bkgFile = os.path.join(bkgImagesDir,
            (thisPPOLname + '_HWP' + str(A_HWP) + '.fits'))

        # If the background image exists, then subtract and save
        if os.path.isfile(bkgFile):
            # Read in the background image
            bkgImg = AstroImage(bkgFile)

            # Perform the actual background subtraction
            bkgSubImg = Aimg - thisSky*bkgImg

            # Save the subtracted image to disk
            outFile = os.path.join(bkgSubDir, os.path.basename(Aimg.filename))
            bkgSubImg.write(outFile, dtype=np.float32)

# Save the new file index including background levels
# First recombine the file indices
finalFileIndex = vstack([fileIndex1, fileIndex2])

# Sort the file index by filename
sortInds = finalFileIndex['Filename'].data.argsort()
finalFileIndex = finalFileIndex[sortInds]

# Then save to disk
print('*************************************')
print('Writing all background levels to disk')
print('*************************************')
finalFileIndex.write(indexFile, format='csv')

print('Done!')
