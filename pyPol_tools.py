# This module provides some "image tools" specific to the Mimir_pyPol scripts

import numpy as np
import psutil
import warnings
import subprocess
import os
import sys
from astropy.io import fits
from wcsaxes import WCS
from astropy.wcs.utils import pixel_to_skycoord
from scipy.odr import *
from scipy.ndimage import map_coordinates
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.interpolate import griddata
from photutils import daofind, Background, detect_sources
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma
from astropy.modeling import models, fitting
from astropy.convolution import Gaussian2DKernel
from astropy.coordinates import SkyCoord, ICRS

import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage
import image_tools

def build_supersky(imgList, clipSigma = 3.0, iters=6):
    '''This function will find the stars in each level and mask them to produce
    a median-filtered-mean supersky image produced from the imgList.

    parameters:
    imgList        -- a list containing Image class objects.
    output         -- [MEAN', 'SUM']
                      the desired array to be returned be the function.
    bkgClipSigma   -- the level at which to trim outliers in the relatively
                      flat background (default = 6.0)
    '''

    # Count the number of images to be combined
    numImg = len(imgList)
    print('\nEntered averaging method')

    # Test for the correct number of bits in each pixel
    dataType    = imgList[0].dtype
    if dataType == np.int16:
        numBits = 16
    elif (dataType == np.int32) or (dataType == np.float32):
        numBits = 32
    elif (dataType == np.int64) or (dataType == np.float64):
        numBits = 64

    # Compute the number of pixels that fit under the memory limit.
    memLimit    = (psutil.virtual_memory().available/
                  (numBits*(1024**2)))
    memLimit    = int(10*np.floor(memLimit/10.0))
    numStackPix = memLimit*(1024**2)*8/numBits
    ny, nx      = imgList[0].arr.shape
    numRows     = int(np.floor(numStackPix/(numImg*nx)))
    if numRows > ny: numRows = ny
    numSections = int(np.ceil(ny/numRows))

    # Recompute the number of rows to be evenly spaced
    numRows = int(np.ceil(ny/numSections))

    # Compute "star mask" for this stack of images
    print('\nComputing masks for bright sources')

    # Check if this is a Mimir image
    MimirImage = (imgList[0].header['INSTRUME'] == 'Mimir Instrument')
    if MimirImage:
        # TODO use the basic astropy.io.fits routines to eliminate
        # dependence on "AstroImage" module
        kokopelliMask = (AstroImage('kokopelliMask.fits').arr != 0)

    # Compute kernel shape
    binX, binY = imgList[0].binning
    medianKernSize = np.ceil(np.sqrt((9.0/binX)*(9.0/binY)))

    # Initalize a final star mask
    starMask = np.zeros((numImg,ny,nx), dtype=bool)

    # We will need to mask out any stellar sources, so let's begin by
    # looping through the images and compute individual star masks.
    # Store the average background values for each image, too.
    bkgList      = []
    skyValueList = []
    for imgNum, img in enumerate(imgList):
        print('Building star mask for image {0:g}'.format(imgNum + 1))
        # Grab the image array
        thisArr = img.arr.copy()

        # Replace bad values with zeros
        badInds = np.where(np.logical_not(np.isfinite(thisArr)))
        thisArr[badInds] = 0

        # Trim the very edges for background estimation (this allows an integer
        # number of sub-regions for background subtraction)
        trimmedArr = thisArr[4:1024,2:1022]

        # Perform an estimate of the background
        bkg = Background(trimmedArr, (60, 60), filter_shape=(10, 10))

        # Store the median background value
        skyValueList.append(bkg.background_median)

        # Re-extend the background
        bkg = bkg.background
        bkg = np.vstack((
            bkg[0,:],
            bkg[0,:],
            bkg[0,:],
            bkg[0,:],
            bkg,
            bkg[1019,:],
            bkg[1019,:]
            ))
        bkg = np.hstack((
            bkg[:,0:1],
            bkg[:,0:1],
            bkg,
            bkg[:,1019:1020],
            bkg[:,1019:1020]
            ))

        bkgList.append(bkg)

        # Normalize this array by the background level
        thisArr /= bkg

        if MimirImage:
            # Compute a conservative starmask for the sketching regions of
            # the Mimir detector
            restrictiveMask = image_tools.build_starMask(thisArr,
                neighborThresh = 4, kernelSize = medianKernSize)

            # Store rows 511-513 for reinsertion after smoothing
            middleRows = restrictiveMask[511:514]

            # Store edges of image
            lfEdge = restrictiveMask[:,0:12]
            rtEdge = restrictiveMask[:,(nx-12):]
            btEdge = restrictiveMask[0:12,:]
            tpEdge = restrictiveMask[(ny-12):,:]

        # Compute a star mask (requiring at least two neighboring masked pixels)
        starMask1 = image_tools.build_starMask(thisArr,
            neighborThresh = 2, kernelSize = medianKernSize)

        if MimirImage:
            # Replace the consevrative mask in the middle rows
            starMask1[511:514] = middleRows

            # blank out the edges before smoothing
            starMask1[:,0:12] = False
            starMask1[:,(nx-12):] = False
            starMask1[0:12,:] = False
            starMask1[(ny-12):,:] = False

        # Dialate the star-mask to avoid any unwanted edge flux
        starMask1 = gaussian_filter(starMask1.astype(float), (4, 4))

        # Grab any pixels (and indices) above 0.05 value post-smoothing
        starMask1  = (starMask1 > 0.05)

        ##############################################################
        # Mask out the REALLY bright star in the NGC 1977 off position
        coords = SkyCoord(ra=82.776437, dec=-04.806723,
            frame = ICRS, unit='deg')
        if img.in_image(coords):
            # Build a quick star mask at that position.
            # Grab the WCS from the header
            wcs = WCS(img.header)

            # Convert coordinates to pixel positions
            x1, y1 = coords.to_pixel(wcs)

            # Generate pixel positions within mask
            yy, xx = np.mgrid[0:ny, 0:nx]

            # Generate distance based mask.
            thisMask = np.sqrt((yy-(y1-6))**2 + (xx - (x1-6))**2) < 38

            # Cross-hair construction.
            # First setup the lines that define the cross hairs
            cross1Ytop = +xx + (y1 - x1 + 8)
            cross1Ybot = +xx + (y1 - x1 - 8)
            cross2Ytop = -xx + (y1 + x1 + 8)
            cross2Ybot = -xx + (y1 + x1 - 8)

            # Construct each half of the cross-hair
            cross1 = np.logical_and(yy < cross1Ytop,
                yy > cross1Ybot)
            cross1 = np.logical_and(cross1,
                xx > x1 - 60)
            cross1 = np.logical_and(cross1,
                xx < x1 + 60)

            cross2 = np.logical_and(yy < cross2Ytop,
                yy > cross2Ybot)
            cross2 = np.logical_and(cross2,
                xx > x1 - 60)
            cross2 = np.logical_and(cross2,
                xx < x1 + 60)

            # Combine the star and cros-hair components
            cross = np.logical_or(cross1, cross2)
            thisMask = np.logical_or(thisMask, cross)
            # plt.ion()
            # plt.imshow(thisArr*(1-thisMask), vmin=0.99,vmax=1.01,cmap='gray',origin='lower')
            # pdb.set_trace()

            # Combine this mask with the star mask
            starMask1 = np.logical_or(starMask1, thisMask)

            # plt.ion()
            # plt.imshow(thisArr*(1-(thisMask + starMask1)), vmin = 0.99, vmax = 1.01,
            #     cmap='gray', origin='lower')
            # pdb.set_trace()
        #############################################################

        # Now mask Kokopelli crack if this is a Mimir image
        if MimirImage:
            starMask1 = np.logical_or(starMask1, kokopelliMask)

        # Compute another star mask, attempting to catch dimmer stars this round
        if np.sum(starMask1) > 0:
            # Begin by filling in the masked pixels
            thisArr[np.where(starMask1)] = 1.0

            # Smooth the image to enhance SMALL star features
            smoothedImg1 = median_filter(thisArr, size=(3,3))
            smoothedImg2 = median_filter(thisArr, size=(12,12))
            residual     = smoothedImg1 - smoothedImg2

            # Establish "threshold" for star detection is this image
            mean, median, stddev = sigma_clipped_stats(residual)
            threshold = median + 3.0*stddev

            # Use "detect_sources" to detect stars in the residual image
            sigma = 2.0 * gaussian_fwhm_to_sigma    # FWHM = 2.
            kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
            kernel.normalize()
            segm = detect_sources(residual, threshold,
                npixels=6, filter_kernel=kernel)
            starMask2 = segm.data > 0

            if MimirImage:
                # Replace the consevrative mask in the middle rows
                starMask2[511:514] = middleRows

                # blank out the edges before smoothing
                starMask2[:,0:12] = False
                starMask2[:,(nx-12):] = False
                starMask2[0:12,:] = False
                starMask2[(ny-12):,:] = False

            # Dialate the star-mask to avoid any unwanted edge flux
            starMask2 = gaussian_filter(starMask2.astype(float), (4, 4))

            # Grab any pixels (and indices) above 0.05 value post-smoothing
            starMask2  = (starMask2 > 0.05)

            # Combine the first and second pass masks
            starMask1 = np.logical_or(starMask1, starMask2)

        if MimirImage:
            # Replace conservative mask estimates after second pass/smoothing
            starMask1[:,0:12] = lfEdge
            starMask1[:,(nx-12):] = rtEdge
            starMask1[0:12,:] = btEdge
            starMask1[(ny-12):,:] = tpEdge
            starMask1 = np.logical_or(starMask1, kokopelliMask)

        # Accumulate these pixels into the final set of star masks
        starMask[imgNum,:,:] = starMask1

        # # Debugging code... check if all the stars were masked
        # plt.ion()
        # plt.imshow(thisArr*(1-0.1*starMask1.astype(int)), vmin = 0.98, vmax = 1.02,
        #     cmap='gray', origin='lower')
        # pdb.set_trace()
        # plt.close()
        # plt.ioff()

    # Cleanup temporary variables
    del thisArr, badInds

    # If there is more than one image in the list (and there generally should
    # be), then compute the supersky from the median-filtered-mean image.
    # Otherwise just take the star-mask and replace star pixels with NaNs.
    if numImg > 1:
        # Compute the number of subsections and display stats to user
        print('\nAiming to fit each stack into {0:g}MB of memory'.format(memLimit))
        print('\nBreaking stack of {0:g} images into {1:g} sections of {2:g} rows'
            .format(numImg, numSections, numRows))

        # Initalize an array to store the final averaged image
        outputImg = np.zeros((ny,nx))

        # Compute the stacked output of each section
        # Loop through each "row section" of the stack
        for thisSec in range(numSections):
            # Calculate the row numbers for this section
            thisRows = (thisSec*numRows,
                        min([(thisSec + 1)*numRows, ny]))

            # Stack the selected region of the images.
            secRows = thisRows[1] - thisRows[0]
            stack   = np.ma.zeros((numImg, secRows, nx), dtype = dataType)
            stack.mask = np.zeros((numImg, secRows, nx), dtype = bool)
            for i in range(numImg):
                # Grab this section of each image and remove the background
                tmpImg  = imgList[i].arr[thisRows[0]:thisRows[1],:]
                tmpImg /= (bkgList[i])[thisRows[0]:thisRows[1],:]

                # Store the relevant data in the stack variable.
                stack.data[i,:,:] = tmpImg
                stack.mask[i,:,:] = starMask[i,thisRows[0]:thisRows[1],:]

            # Catch and mask any NaNs or Infs (or -1e6 values)
            # before proceeding with the average
            NaNsOrInfs = np.logical_not(np.isfinite(stack.data))
            stack.mask = np.logical_or(stack.mask, NaNsOrInfs)
            stack.data[np.where(stack.mask)] = -1e6
            # Complement the NaNs search with bad pix value search
            badPix      = stack.data < -1e5
            stack.mask  = np.logical_or(NaNsOrInfs, badPix)

            # Now that the bad values have been saved,
            # replace them with signal "bad-data" values
            stack.data[np.where(stack.mask)] = -1e6

            print('\nAveraging rows {0[0]:g} through {0[1]:g}'.format(thisRows))

            # Iteratively clip outliers until answer converges.
            # Use the stacked median for first image estimate.
            outliers = np.zeros(stack.shape, dtype = bool)

            # This loop will iterate until the mask converges to an
            # unchanging state, or until clipSigma is reached.
            # Compute the starting point for the sigma clipping,
            # assuming that we will step in increments of 0.2 sigma.
            sigmaStart    = clipSigma - iters*0.2
            numPoints     = np.zeros((secRows, nx), dtype=int) + numImg
            scale         = sigmaStart*np.ones((secRows, nx), dtype=float)
            for iLoop in range(iters):
                print('\tProcessing section for {0:3.2g} sigma'.format(
                    sigmaStart + 0.2*iLoop))

                # Loop through the stack, and find the outliers.
                imgEstimate = np.ma.median(stack, axis = 0).data
                stackSigma  = np.ma.std(stack, axis = 0).data

                # Loop through each section of the stack and find outliers
                for j in range(numImg):
                    deviation       = np.absolute(stack.data[j,:,:] - imgEstimate)
                    outliers[j,:,:] = (deviation > scale*stackSigma)

                # Save the newly computed outliers to the mask
                stack.mask = np.logical_or(outliers, NaNsOrInfs)
                # Save the number of unmasked points along AXIS
                numPoints1 = numPoints
                # Total up the new number of unmasked pixels in each column
                numPoints  = np.sum(np.logical_not(stack.mask), axis = 0)
                # Figure out which pixel columns have improved results
                nextScale  = (numPoints != numPoints1).astype(int)

                if np.sum(nextScale) == 0:
                    # If there are no new data included, then break out of loop
                    break
                else:
                    # Otherwise increment scale where new data are included
                    scale += nextScale*0.2

            # Compute the mean of the unmasked values
            tmpOut = np.ma.mean(stack, axis = 0)

            # Find the pixels with no contribution.
            numSamples = np.sum(np.logical_not(stack.mask), axis = 0)

            # Replace the unsampled values with NaNs
            nanInds         = np.where(numSamples == 0)
            tmpOut[nanInds] = np.NaN

            # Place the output and uncertainty in their holders
            outputImg[thisRows[0]:thisRows[1],:] = tmpOut.data

        # Map the location of finite pixels
        finitePix = np.isfinite(outputImg)

        # Store the location of NaN pixels and replace with 1.0
        if np.sum(np.logical_not(finitePix)) > 0:
            nanInds = np.where(np.logical_not(np.isfinite(outputImg)))
            outputImg[nanInds] = 1.0

        # Re-normalize the supersky
        if np.sum(finitePix) > 0:
            goodInds = np.where(finitePix)
            mean, median, stddev = sigma_clipped_stats(outputImg[goodInds])
            outputImg /= mean
        else:
            print('No finite pixels means something is broken')
            pdb.set_trace()

        # Grab any "outliers" in the output image and set them to local median
        medImg = median_filter(outputImg, size=(9, 9))
        mean, median, stddev = sigma_clipped_stats(outputImg[goodInds])
        fixPix = np.abs((outputImg - medImg)/stddev) > 3.0
        if np.sum(fixPix) > 0:
            fixPixInds = np.where(fixPix)
            outputImg[fixPixInds] = medImg[fixPixInds]

        # Re-insert original NaN pixels
        if np.sum(np.logical_not(finitePix)) > 0:
            outputImg[nanInds] = np.NaN

        # # Debugging code... check if all the stars were masked
        # plt.ion()
        # plt.imshow(outputImg, vmin = 0.9949, vmax = 1.0065,
        #     cmap='gray', origin='lower')
        # pdb.set_trace()
        # plt.close()
        # plt.ioff()


        # Inpaint NaN values for the output image
        outputImg = image_tools.inpaint_nans(outputImg)

        # # Debugging code... check if all the stars were masked
        # plt.ion()
        # plt.imshow(outputImg1, vmin = 0.9949, vmax = 1.0065,
        #     cmap='gray', origin='lower')
        # pdb.set_trace()
        # plt.close()
        # plt.ioff()


        # Smooth the supersky with a median filter
        outputImg = median_filter(outputImg, size = (7,7))

        # # Debugging code... check if all the stars were masked
        # plt.ion()
        # plt.imshow(outputImg2, vmin = 0.9949, vmax = 1.0065,
        #     cmap='gray', origin='lower')
        # pdb.set_trace()
        # plt.close()
        # plt.ioff()


        # Bias the supersky back to the mean sky value and normalize
        # outputImg += np.mean(skyValueList)
        subSampled = (outputImg[0:ny:10, 0:nx:10]).flatten()
        mean, median, stddev = sigma_clipped_stats(subSampled)
        outputImg /= mean

        # Store the final result for output image
        # Get ready to return an AstroImage object to the user
        outImg     = imgList[0].copy()
        outImg.arr = outputImg

    else:
        # Copy the output image array
        outputImg = imgList[0].arr

        # Grab the starMask arrya
        starMask = starMask[0]

        # Map the location of finite pixels
        goodInds = np.where(np.logical_not(starMask))
        nanInds  = np.where(starMask)

        # Grab any "outliers" in the output image and set them to local median
        medImg = median_filter(outputImg, size=(9, 9))
        mean, median, stddev = sigma_clipped_stats(outputImg[goodInds])
        fixPix = np.abs((outputImg - medImg)/stddev) > 3.0
        if np.sum(fixPix) > 0:
            fixPixInds = np.where(fixPix)
            outputImg[fixPixInds] = medImg[fixPixInds]

        # Re-insert original NaN pixels
        if np.sum(starMask) > 0:
            outputImg[nanInds] = np.NaN

        # Inpaint NaN values for the output image
        outputImg = image_tools.inpaint_nans(outputImg)

        # Smooth the supersky with a median filter
        outputImg = median_filter(outputImg, size = (9,9))

        # Bias the supersky back to the mean sky value and normalize
        # outputImg += np.mean(skyValueList)
        subSampled = (outputImg[0:ny:10, 0:nx:10]).flatten()
        mean, median, stddev = sigma_clipped_stats(subSampled)
        outputImg /= mean

        # Store the final result for output image
        outImg = imgList[0].copy()
        outImg.arr = outputImg

    # Now that an average image has been computed,
    # Clear out the old astrometry data (may not be relevant anyway)
    if 'WCSAXES' in outImg.header.keys():
        del outImg.header['WCSAXES']
        del outImg.header['PC*']
        del outImg.header['CDELT*']
        del outImg.header['CUNIT*']
        del outImg.header['*POLE']
        outImg.header['CRPIX*'] = 1.0
        outImg.header['CRVAL*'] = 1.0
        outImg.header['CTYPE*'] = 'Linear Binned ADC Pixels'

    # Make sure the output image has the proper shape information in its header
    outImg.header['NAXIS1'] = outImg.arr.shape[1]
    outImg.header['NAXIS2'] = outImg.arr.shape[0]

    # Finally return the result
    return outImg

def test_for_gradients(img, sig_thresh = 10.0):
    '''This function takes an AstroImage object and tests it for the presence of
    a gradient along the upper edge and right edge. If gradients are significant
    along BOTH edges, then the test returns boolean True, otherwise False.
    '''
    # Grab the image array and its shape
    arr = img.arr
    ny, nx = arr.shape

    # Define a simple linear model to fit using ODR
    lineFunc = lambda B, x: B[0]*x + B[1]

    # Define the model to use for all regions (simple line should do)
    lineModel = Model(lineFunc)

    ################################# TOP ZONE #################################
    # Sample the upper edge
    xTop   = []
    zTop   = []
    s_zTop = []
    # Loop through 16 sample points along top 1/4 of image
    for ix in range(16):
        # Cut out this zone and compute statistics
        lf, rt    = ix*64, (ix + 1)*64
        bt, tp    = ny - 257, ny - 1
        zone      = arr[bt:tp, lf:rt]
        mean, median, stddev = sigma_clipped_stats(zone)

        # Store in lists
        xTop.append((ix + 0.5)*64)
        zTop.append(median)
        s_zTop.append(stddev/np.sqrt(zone.size - 1))

    # Define the data
    topData = RealData(xTop, zTop,
        sx = np.repeat(32, 16), sy = s_zTop)

    # Define the ODR object and run it
    topODR = ODR(topData, lineModel, beta0=[0.0, np.median(zTop)])
    topOut = topODR.run()

    # Test for the presence of a slope in the top zone
    topSlopeBool   = (np.abs(topOut.beta[0]/topOut.sd_beta[0]) > 6.0 and
                      topOut.beta[0] < -0.06)

    ################################ RIGHT ZONE ################################
    # Sample the right edge
    yRight   = []
    zRight   = []
    s_zRight = []
    # Loop through 16 sample points along the right 1/4 of image
    for iy in range(16):
        # Cut out this zone and compute statistics
        lf, rt    = nx - 257, nx - 2
        bt, tp    = 1 + iy*64, 1 + (iy + 1)*64
        zone      = arr[bt:tp, lf:rt]
        mean, median, stddev = sigma_clipped_stats(zone)

        # Store in lists
        yRight.append(1 + (iy + 0.5)*64)
        zRight.append(median)
        s_zRight.append(stddev/np.sqrt(zone.size - 1))

    # Define the data
    rightData = RealData(yRight, zRight,
        sx = np.repeat(32, 16), sy = s_zRight)

    # Define the ODR object and run it
    rightODR = ODR(rightData, lineModel, beta0=[0.0, np.median(zRight)])
    rightOut = rightODR.run()

    # test for the presence of a solpe in the right zone
    rightSlopeBool = (np.abs(rightOut.beta[0]/rightOut.sd_beta[0]) > 6.0 and
                      rightOut.beta[0] < -0.06)

    ############################## DIAGONAL 1 ZONE #############################
    # Sample bottom left to top right diagonal zone
    posDiag1 = []
    zDiag1   = []
    s_zDiag1 = []
    for iDiag in range(16):
        # Cut out this zone and compute statistics
        lf, rt    = iDiag*64, (iDiag + 1)*64
        bt, tp    = 1 + iDiag*64, 1 + (iDiag + 1)*64
        zone      = arr[bt:tp, lf:rt]
        mean, median, stddev = sigma_clipped_stats(zone)

        # Store in lists
        posDiag1.append(np.sqrt(2)*(iDiag + 0.5)*64)
        zDiag1.append(median)
        s_zDiag1.append(stddev/np.sqrt(zone.size - 1))

    # Treat the bottom left to top right diagonal
    # Define the data
    diag1Data = RealData(posDiag1, zDiag1,
        sx = np.repeat(np.sqrt(2)*32, 16), sy = s_zDiag1)

    # Define the ODR object and run it
    diag1ODR = ODR(diag1Data, lineModel, beta0=[0.0, np.median(zDiag1)])
    diag1Out = diag1ODR.run()

    # Test for the presence of a slope along diagonal 1
    # (bottom left to upper right)
    diag1SlopeBool  = (np.abs(diag1Out.beta[0]/diag1Out.sd_beta[0]) > 10.0 and
                      diag1Out.beta[0] < -0.1)
    ############################## DIAGONAL 2 ZONE #############################
    # Sample bottom left to top right diagonal zone
    posDiag2 = []
    zDiag2   = []
    s_zDiag2 = []
    for iDiag in range(0,8):
        # Cut out this zone and compute statistics
        lf, rt    = iDiag*64, (iDiag + 1)*64
        bt, tp    = ny - (iDiag + 1)*64 - 2, ny - iDiag*64 - 2
        zone      = arr[bt:tp, lf:rt]
        mean, median, stddev = sigma_clipped_stats(zone)

        # Store in lists
        posDiag2.append(np.sqrt(2)*(iDiag + 0.5)*64)
        zDiag2.append(median)
        s_zDiag2.append(stddev/np.sqrt(zone.size - 1))

    # Define the data
    diag2Data = RealData(posDiag2, zDiag2,
        sx = np.repeat(32, 8), sy = s_zDiag2)

    # Define the ODR object and run it
    diag2ODR = ODR(diag2Data, lineModel, beta0=[0.0, np.median(zDiag2)])
    diag2Out = diag2ODR.run()

    # Test for the presence of a slope along diagonal 2
    # (top left to center)
    diag2SlopeBool  = (np.abs(diag2Out.beta[0]/diag2Out.sd_beta[0]) > 20.0 and
                      diag2Out.beta[0] < -0.2)

    ############################## DIAGONAL 3 ZONE #############################
    # Sample bottom left to top right diagonal zone
    posDiag3 = []
    zDiag3   = []
    s_zDiag3 = []
    for iDiag in range(12,16):
        # Cut out this zone and compute statistics
        lf, rt    = iDiag*64, (iDiag + 1)*64
        bt, tp    = ny - (iDiag + 1)*64 - 2, ny - iDiag*64 - 2
        zone      = arr[bt:tp, lf:rt]
        mean, median, stddev = sigma_clipped_stats(zone)

        # Store in lists
        posDiag3.append(np.sqrt(2)*(iDiag + 0.5)*64)
        zDiag3.append(median)
        s_zDiag3.append(stddev/np.sqrt(zone.size - 1))

    # Define the data
    diag3Data = RealData(posDiag3, zDiag3,
        sx = np.repeat(32, 8), sy = s_zDiag3)

    # Define the ODR object and run it
    diag3ODR = ODR(diag3Data, lineModel, beta0=[0.0, np.median(zDiag3)])
    diag3Out = diag3ODR.run()

    # Test for the presence of a slope along diagonal 2
    # (center to bottom right)
    diag3SlopeBool  = (np.abs(diag3Out.beta[0]/diag3Out.sd_beta[0]) > 20.0 and
                       diag3Out.beta[0] > 0.2)

    ############################## RETURN DECISION #############################
    # If all three of theses tests pass, then this is a type 1 gradient
    #(bright bottom left, dim top right)
    if (topSlopeBool and rightSlopeBool and diag1SlopeBool):
        return 1

    # If the two extra diagonal tests pass, then this is a type 2 gradient
    if (diag2SlopeBool and diag3SlopeBool):
        return 2

    # If none of these tests passed at all, then
    else:
        return 0

def flatten_gradients(img, gradientType=1, mask=None):
    '''This function performs a polynomial fit to a sample of grid-points from
    the image array and subtracts that fit from the image to flatten gradients.
    '''
    # Grab the image shape
    ny, nx = img.arr.shape

    # Check if a mask is provided and build a null mask if needed
    if mask is None:
        mask = np.zeros((ny, nx), dtype = bool)
    elif mask.shape != img.arr.shape:
        raise ValueError('"mask" must have same shape as "img"')

    # Parse the gradient type parameter
    if gradientType == 1:
        polyDeg = 1
    elif gradientType ==2:
        polyDeg = 2

    # Loop through the sample points and compute coordinates and median values
    x = []
    y = []
    z = []
    for ix in range(0, 16):
        for iy in range(0, 16):
            # Compute position centroid
            thisX, thisY = np.int((ix + 0.5)*64), np.int((iy + 0.5)*64 + 1)

            # Test if this is a masked portion of the image
            if not mask[thisY, thisX]:
                # Compute this region boundaries
                lf, rt = ix*64, (ix + 1)*64
                bt, tp = (iy*64 + 1), ((iy + 1)*64 + 1)

                # Store the median value of this zone
                x.append(thisX)
                y.append(thisY)
                z.append(np.median(img.arr[bt:tp,lf:rt]))

    # Convert the coordinate and median lists into arrays
    x  = np.array(x)
    y  = np.array(y)
    z  = np.array(z)

    # Prepare to use griddata
    y1, x1 = np.mgrid[0:ny, 0:nx]

    # Fit the data using astropy.modeling
    p_init = models.Polynomial2D(degree=polyDeg)
    fit_p = fitting.LevMarLSQFitter()

    # Do a fitting (ignore warning on linearity)
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x, y, z)

    # Compute fitted values
    grid_z = p(x1, y1)

    # Return a copy of the image with the gradient subtracted.
    bkgLevel   = np.mean(z)
    grid_z1    = grid_z - bkgLevel
    outImg     = img.copy()
    outImg.arr = img.arr - grid_z1

    return outImg
