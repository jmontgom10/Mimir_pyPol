# -*- coding: utf-8 -*-
"""
Reads and aligns the averaged polAng images for each (target, filter) pair.
Uses the 'StokesParameters' class to compute Stokes and polarization images.
"""

# Core imports
import os
import copy
import sys

# Scipy/numpy imports
import numpy as np

# Astropy imports
from astropy.table import Table
import astropy.units as u
from astropy.stats import sigma_clipped_stats

# Import astroimage
import astroimage as ai

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# # This is a list of targets which have a hard time with the "cross_correlate"
# # alignment method, so use "wcs" method instead
# wcsAlignmentList = ['NGC7023', 'NGC2023']
#
# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201612\\'

# Specify which (target, filter) pairs to process
# targetsToProcess = ['NGC2023', 'NGC7023', 'NGC891', 'NGC4565']
targetsToProcess = ['NGC891', 'NGC4565', 'NGC2023', 'NGC7023']
filtersToProcess = ['V', 'R']

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')

polAngDir = os.path.join(polarimetryDir, 'polAngImgs')

stokesDir = os.path.join(polarimetryDir, 'stokesImgs')
if (not os.path.isdir(stokesDir)):
    os.mkdir(stokesDir, 0o755)

# Read in the polarization calibration constants
polCalFile      = os.path.join(pyPol_data, 'polCalConstants.csv')
polCalConstants = Table.read(polCalFile)

# Define a set of polAng values for reference within the loops
polAngs = [0, 200, 400, 600]

# Define a corresponding set of IPPA values for each polAng value
IPPAs = [0, 45, 90, 135]

# Build a list of dictionary keys for these IPPAs
IPPAkeys = ['I_' + str(IPPA) for IPPA in IPPAs]

# Loop through each target
for thisTarget in targetsToProcess:
    # Update the user on progress
    print('Processing files for')
    print('Target : {}'.format(thisTarget))

    ######
    # Check if the files have already been processed.
    ######
    # Initalize a blank list and dict to store all the Stokes file names
    stokesFileList = []
    stokesFileDict = {}

    # Loop through each filter for the current target
    for thisFilter in filtersToProcess:
        # Construct the expected output names filenames for this
        # (target, filter) pair
        stokesIfilename = '_'.join([thisTarget, thisFilter, 'I']) + '.fits'
        stokesIfilename = os.path.join(stokesDir, stokesIfilename)
        stokesQfilename = '_'.join([thisTarget, thisFilter, 'Q']) + '.fits'
        stokesQfilename = os.path.join(stokesDir, stokesQfilename)
        stokesUfilename = '_'.join([thisTarget, thisFilter, 'U']) + '.fits'
        stokesUfilename = os.path.join(stokesDir, stokesUfilename)

        # Store the output Stokes filenames in the list and dictionary
        thisFilterStokesFiles = [stokesIfilename, stokesQfilename, stokesUfilename]
        stokesFileList.extend(thisFilterStokesFiles)
        stokesFileDict[thisFilter] = thisFilterStokesFiles

    # Check if all the Stokes files for this target have already been processed.
    # If they have been processed, then skip this target
    if all([os.path.isfile(f) for f in stokesFileList]):
        print('\tFile for {0} already exists...'.format(thisTarget))
        continue

    # Initalize lists and dictionaries for storing each IPPA filename and image
    # for the current target.
    ippaImgList = []
    ippaImgIndexDict = {}
    # Loop back through each filter and construct the IPPA filenames
    for thisFilter in filtersToProcess:
        # Construct the expected input filenames for this (target, filter) pair
        thisFilterFilenames = [
            os.path.join(
                polAngDir,
                '_'.join([thisTarget, thisFilter, str(polAng)]) + '.fits'
            )
            for polAng in polAngs
        ]

        # Test if all of the expected polAng images exist
        if not all([os.path.isfile(f) for f in thisFilterFilenames]):
            raise ValueError('Could not locate all of the expected polAng images for {0}_{1}'.format(
                thisTarget, thisFilter
            ))

        # Read in the IPPA images for this filter
        thisFilterImgList = [ai.reduced.ReducedScience.read(f) for f in thisFilterFilenames]

        ########################################################################
        # HACK: read in a list of uncertainty arrays and store them...
        from astropy.io import fits
        thisFilterUncertList = [fits.open(f)[1].data for f in thisFilterFilenames]
        thisFilterImgList1 = []
        for img, uncert in zip(thisFilterImgList, thisFilterUncertList):
            img.uncertainty = uncert
            thisFilterImgList1.append(img)
        ########################################################################

        # Store the IPPA images in the ippaImgDict
        thisFilterIndexList = [
            ind for ind in range(
                len(ippaImgList), len(ippaImgList) + len(thisFilterImgList),
            )
        ]

        # Store the IPPA file names in the list
        ippaImgList.extend(thisFilterImgList)

        ippaImgIndexDict[thisFilter] = thisFilterIndexList

    # Place the images in an ImageStack for alignment
    ippaImgStack = ai.utilitywrappers.ImageStack(copy.deepcopy(ippaImgList))

    # Allign ALL images for this stack
    ippaImgStack.align_images_with_cross_correlation(
        subPixel=True,
        satLimit=22e3
    )

    # Grab the reference "median" image
    referenceImage = ippaImgStack.build_median_image()

    # Trigger a re-solving of the image astrometry. Start by clearing relevant
    # astrometric data from the header.
    referenceImage.clear_astrometry()
    tmpHeader = referenceImage.header
    del tmpHeader['POLPOS']
    referenceImage.header = tmpHeader

    print('\tSolving astrometry for the reference image.')
    # Clear out the filename so that the solver will use a TEMPORARAY file
    # to find and solve the astrometry
    referenceImage.filename = ''

    # Create an AstrometrySolver instance and run it.
    astroSolver = ai.utilitywrappers.AstrometrySolver(referenceImage)
    referenceImage, success = astroSolver.run(clobber=True)

    if not success:
        raise RuntimeError('Failed to solve astrometry of the reference image.')

    # Loop through ALL the images and assign the solved astrometry to them
    imgList = []
    for img in ippaImgStack.imageList:
        img.astrometry_to_header(referenceImage.wcs)
        imgList.append(img)

    # Recreate the IPPA image stack from this updated list of images
    ippaImgStack = ai.utilitywrappers.ImageStack(imgList)

    # Now look ONE-MORE-TIME through each filter and align all its images to
    # THIS REFERENCE IMAGE
    for iFilter, thisFilter in enumerate(filtersToProcess):
        print('\tFilter : {0}'.format(thisFilter))

        # Store the polAngImgs in a dictonary with IPPA keys
        polAngImgDict = dict(zip(
            IPPAkeys,
            ippaImgStack.imageList[iFilter*4:(iFilter+1)*4]
        ))

        # Set the calibration constants for this waveband
        # ########################################################################
        # # HACK: to get through old calibration file designations
        # polCalRowInd = np.where(polCalConstants['Waveband'] == thisFilter)
        # polCalRow    = polCalConstants[polCalRowInd]
        # polCalDict   = {}
        # for key, properKey in zip(['PE', 's_PE', 'PAsign', 'dPA', 's_dPA'], ['PE', 's_PE', 'PAsign', 'D_PA', 's_D_PA']):
        #     if key in ['dPA', 's_dPA']:
        #         polCalDict[properKey] = float(polCalRow[key])*u.deg
        #     else:
        #         polCalDict[properKey] = float(polCalRow[key])
        # ########################################################################
        polCalRowInd = np.where(polCalConstants['FILTER'] == thisFilter)
        polCalRow    = polCalConstants[polCalRowInd]
        polCalDict   = {}
        for key in ['PE', 's_PE', 'PAsign', 'D_PA', 's_D_PA']:
            if key in ['D_PA', 's_D_PA']:
                polCalDict[key] = float(polCalRow[key])*u.deg
            else:
                polCalDict[key] = float(polCalRow[key])

        # Set the polarization calibration constants for this filter
        ai.utilitywrappers.StokesParameters.set_calibration_constants(polCalDict)

        # Construct the StokesParameters object
        stokesParams = ai.utilitywrappers.StokesParameters(polAngImgDict)

        # Compute the Stokes parameter images
        stokesParams.compute_stokes_parameters(resolveAstrometry=False)

        # Update the astrometry in the image headers
        stokesI = stokesParams.I.copy()
        stokesI.astrometry_to_header(referenceImage.wcs)
        stokesQ = stokesParams.Q.copy()
        stokesQ.astrometry_to_header(referenceImage.wcs)
        stokesU = stokesParams.U.copy()
        stokesU.astrometry_to_header(referenceImage.wcs)

        # Grab the filenames back from the original storage point.
        stokesIfilename, stokesQfilename, stokesUfilename = (
            stokesFileDict[thisFilter]
        )

        # Write the stokes parameter images to disk
        stokesI.write(stokesIfilename, clobber=True)
        stokesQ.write(stokesQfilename, clobber=True)
        stokesU.write(stokesUfilename, clobber=True)

print('Done processing images!')
