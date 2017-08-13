# -*- coding: utf-8 -*-
"""
Reads and aligns the averaged polAng images for each (target, filter) pair.
Uses the 'StokesParameters' class to compute Stokes and polarization images.
"""

# Core imports
import os
import copy
import sys
from datetime import datetime

# Scipy/numpy imports
import numpy as np

# Astropy imports
from astropy.table import Table
import astropy.units as u
from astropy.stats import sigma_clipped_stats

# Import astroimage
import astroimage as ai
ai.set_instrument('Mimir')

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# Specify the location of the PPOL *software*
# (where calibration constants are saved)
PPOLsoftwareDir = 'C:\\Users\\Jordan\\IDL8_MSP_Workspace\\MSP_PPOL'
Hpol_calDir     = os.path.join(PPOLsoftwareDir, 'H-Pol_Pinst_Default')
Kpol_calDir     = os.path.join(PPOLsoftwareDir, 'K-Pol_Pinst_Default')

# # This is a list of targets which have a hard time with the "cross_correlate"
# # alignment method, so use "wcs" method instead
# wcsAlignmentList = ['NGC7023', 'NGC2023']
#
# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\'

# Specify which (target, filter) pairs to process
targetFilterDict = {
    'NGC2023':['H', 'Ks'],
    'NGC7023':['H', 'Ks'],
    'M78':['H', 'Ks']
}

# These are the directories where polarimetry data are stored
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
IPPAdir = os.path.join(polarimetryDir, 'IPPAimages')

stokesDir = os.path.join(polarimetryDir, 'stokesImages')
if (not os.path.isdir(stokesDir)):
    os.mkdir(stokesDir, 0o755)

################################################################################
# Write a quick function to parse the P_inst file
def parsePinstFile(filename):
    """Reads the supplied file and returns the information as a dictionary"""
    try:
        # Open the file to read its contents
        pinstFile = open(filename, 'r')

        # Initalize a dictionary to return
        PinstDict = {}

        # Loop through each line of the file
        for line in pinstFile:
            # Check if this line has some valuable information
            if (line[0:3] != 'HIST') and ('=' in line):
                # Split the line at the equal sign
                splitLine = line.split('=')

                # Parse the split line
                key = splitLine[0].strip()

                # Split on the forward slash
                value, comment = splitLine[1].split('/')

                # Convert the value into a float (ignore comment for now)
                value = float(value)

                # Add this key, value pair to the dictionary
                PinstDict[key] = value

    except:
        raise

    return PinstDict
################################################################################

# Read in the polarization calibration constants
HpolCalFile = os.path.join(Hpol_calDir, 'P_inst_values.dat')
HpolCalDict = parsePinstFile(HpolCalFile)
KpolCalFile = os.path.join(Kpol_calDir, 'P_inst_values.dat')
KpolCalDict = parsePinstFile(KpolCalFile)
polCalDict  = {'H':HpolCalDict, 'Ks':KpolCalDict}

# Grab the year-blocks from the Pinst files
Hpol_yrBlockKeys = []
Hpol_yrBlockVals = []
for key, value in HpolCalDict.items():
    if 'ENDYR' in key:
        Hpol_yrBlockKeys.append(key)
        Hpol_yrBlockVals.append(value)
Hpol_yrBlockKeys = np.array(Hpol_yrBlockKeys)
Hpol_yrBlockVals = np.array(Hpol_yrBlockVals)
Hpol_yrSortInds  = Hpol_yrBlockVals.argsort()
Hpol_yrBlockKeys = Hpol_yrBlockKeys[Hpol_yrSortInds]
Hpol_yrBlockVals = Hpol_yrBlockVals[Hpol_yrSortInds]

Kpol_yrBlockKeys = []
Kpol_yrBlockVals = []
for key, value in KpolCalDict.items():
    if 'ENDYR' in key:
        Kpol_yrBlockKeys.append(key)
        Kpol_yrBlockVals.append(value)
Kpol_yrBlockKeys = np.array(Kpol_yrBlockKeys)
Kpol_yrBlockVals = np.array(Kpol_yrBlockVals)
Kpol_yrSortInds  = Kpol_yrBlockVals.argsort()
Kpol_yrBlockKeys = Kpol_yrBlockKeys[Kpol_yrSortInds]
Kpol_yrBlockVals = Kpol_yrBlockVals[Kpol_yrSortInds]

yrBlockDict = {
    'H':  dict(zip(Hpol_yrBlockKeys, Hpol_yrBlockVals)),
    'Ks': dict(zip(Kpol_yrBlockVals, Kpol_yrBlockVals))
}
yrBlockKeys = {
    'H':  Hpol_yrBlockKeys,
    'Ks': Kpol_yrBlockKeys
}
yrBlockVals = {
    'H':  Hpol_yrBlockVals,
    'Ks': Kpol_yrBlockVals
}

# Define a corresponding set of IPPA values for each polAng value
IPPAs = [0, 45, 90, 135]

# Build a list of dictionary keys for these IPPAs. These will be useed to
# create a StokesParameters object from which to compute polarization maps.
IPPAkeys = ['I_' + str(IPPA) for IPPA in IPPAs]

# Initalize a dictionary to store all the IPPA images for this target

# Loop through each target-filter pairing
for thisTarget, filters in targetFilterDict.items():
    # Quickly loop through filters and check if this target has already been done
    stokesFileList = []
    stokesFileDict = {}
    for thisFilter in filters:
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
        print('\tTraget {0} has already been processed... skipping'.format(thisTarget))
        continue

    # Now that it's been confirmed that some processing remains to be done, loop
    # back through the filters and read in all the images
    IPPAimgList = []
    for thisFilter in filters:
        # Loop through each IPPA
        for IPPA in IPPAs:
            # Construct the expected IPPA image names for this pairing
            thisIPPAfile = os.path.join(
                IPPAdir,
                '{}_{}_{}.fits'.format(thisTarget, thisFilter, str(IPPA))
            )

            # Check if that file already exists (it *REALLY* must!)
            if not os.path.isfile(thisIPPAfile):
                print('Could not find file {}'.format(thisIPPAfile))
                import pdb; pdb.set_trace()

            # If the file *does* exist, then simply read it in
            thisIPPAimg = ai.reduced.ReducedScience.read(thisIPPAfile)

            # Store this image in the IPPA image list for this filter
            IPPAimgList.append(thisIPPAimg)

    # Place the images in an ImageStack for alignment
    IPPAimgStack = ai.utilitywrappers.ImageStack(copy.deepcopy(IPPAimgList))

    # Allign ALL images for this stack
    IPPAimgStack.align_images_with_cross_correlation(
        subPixel=True,
        satLimit=16e3
    )

    # Grab the reference "median" image
    referenceImage = IPPAimgStack.build_median_image()

    # Trigger a re-solving of the image astrometry. Start by clearing relevant
    # astrometric data from the header.
    referenceImage.clear_astrometry()
    tmpHeader = referenceImage.header
    del tmpHeader['HWP*']
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
    for img in IPPAimgStack.imageList:
        img.astrometry_to_header(referenceImage.wcs)
        imgList.append(img)

    # Recreate the IPPA image stack from this updated list of images
    IPPAimgStack = ai.utilitywrappers.ImageStack(imgList)

    # Now look ONE-MORE-TIME through each filter and align all its images to
    # THIS REFERENCE IMAGE
    for iFilter, thisFilter in enumerate(filters):
        print('\tFilter : {0}'.format(thisFilter))

        # Grab the IPPA images associated with this filter
        # NOTE: there is a bit of guesswork here assuming that the images are in
        # such an order that each group of 4 IPPA images is associated with a
        # single filter. This might be resolved in the future, but it works for now.
        IPPAimgDict = dict(zip(
            IPPAkeys,
            IPPAimgStack.imageList[iFilter*4:(iFilter+1)*4]
        ))

        # Compute the date of observation for this image
        thisDatetime = IPPAimgDict['I_0'].datetime
        Yr0          = datetime(thisDatetime.year, 1, 1, 0, 0, 0)
        secondsInYr  = 365.25*24*60*60
        dateAfterYr0 = (thisDatetime - Yr0).total_seconds()/secondsInYr
        thisFracYr   = thisDatetime.year + dateAfterYr0

        # Find the year-block corresponding to this date
        yrBlockInd   = np.max(np.where(yrBlockVals[thisFilter] < thisFracYr))
        timeBlockKey = yrBlockKeys[thisFilter][yrBlockInd]
        timeBlockNum = timeBlockKey.split('_')[-1]
        D_PA_key     = 'P_OFF_' + timeBlockNum
        s_D_PA_key   = 'SP_OF_' + timeBlockNum

        # Grab the PA offset and uncertainty in that offset for these values
        D_PA   = polCalDict[thisFilter][D_PA_key]*u.degree
        s_D_PA = polCalDict[thisFilter][s_D_PA_key]*u.degree

        # Grab the other calibrating constants straight from the dictionary
        PE   = polCalDict[thisFilter]['P_EFFIC']
        s_PE = polCalDict[thisFilter]['S_P_EFF']

        # Construct the polarization calibration dictionary for these images
        thisPolCalDict = {
            'PE': PE,
            's_PE': s_PE,
            'PAsign': -1,
            'D_PA': D_PA,
            's_D_PA': s_D_PA
        }

        # Set the polarization calibration constants for this filter
        ai.utilitywrappers.StokesParameters.set_calibration_constants(
            thisPolCalDict
        )

        # Construct the StokesParameters object for this filter
        stokesParams = ai.utilitywrappers.StokesParameters(IPPAimgDict)

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
