import os
import sys
import subprocess
import datetime
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.wcs import WCS
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from scipy.stats import norm, f
from scipy.ndimage.filters import median_filter, gaussian_filter
from scipy.ndimage.interpolation import zoom
from photutils import detect_sources, Background

# For debugging
import matplotlib.pyplot as plt
import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage
import image_tools

# This script will run the image averaging step of the pyPol reduction. All
# on-target images for each target and waveband will be combined to produce
# 1) average HWP images
# 2) average IPPA images
# 3) average stokes (I, U, Q) images and associated uncertainties
# *** HOW ARE UNCERTAINTIES TO BE HANDLED?

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================
# This is the location of all PPOL reduction directory
PPOL_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_reduced'

# This is the location where PPOL was installed.
# This directory contains information about the Mimir calibration procedure.
PPOLcodeDir = 'C:\\Users\\Jordan\\IDL8_MSP_Workspace\\MSP_PPOL'

# Now build the location of the S6 scale factor data files
S6dir = os.path.join(PPOL_dir, 'S6_Scaling')

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_data'

# Setup the directory where the supersky images are saved
bkgSubDir = os.path.join(pyPol_data, 'bkgSubtracted')

# Setup new directory for polarimetry data
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

stokesDir = os.path.join(polarimetryDir, 'stokesImgs')
if (not os.path.isdir(stokesDir)):
    os.mkdir(stokesDir, 0o755)

# Setup Mimir detector properties
read_noise = 19.0 # electrons
effective_gain = 8.21 # electrons/ADU

# Read in the Kokopelli mask
if os.path.isfile('kokopelliMask.fits'):
    kokopelliMask = AstroImage('kokopelliMask.fits')
else:
    print('Kokopelli Mask is not built... go back and build it with step 2.')
    pdb.set_trace()
################################################################################
# READ IN CALIBRATION DATA
################################################################################
# Read in the instrumental U and Q images (and associated uncertainties)
# First do H-band
# Q images
H_calDir      = os.path.join(PPOLcodeDir, 'H-Pol_Pinst_Default')
H_Q_instFile  = os.path.join(H_calDir, 'q_inst.fits')
H_sQ_instFile = os.path.join(H_calDir, 'q_sig_inst.fits')
H_Q_instImg   = AstroImage(H_Q_instFile)
H_sQ_instImg  = AstroImage(H_sQ_instFile)
H_Q_instImg.sigma = H_sQ_instImg.arr.copy()

# U images
H_U_instFile  = os.path.join(H_calDir, 'u_inst.fits')
H_sU_instFile = os.path.join(H_calDir, 'u_sig_inst.fits')
H_U_instImg   = AstroImage(H_U_instFile)
H_sU_instImg  = AstroImage(H_sU_instFile)
H_U_instImg.sigma = H_sU_instImg.arr.copy()

del H_sQ_instImg, H_sU_instImg

# Then do K-band
# Q images
K_calDir = os.path.join(PPOLcodeDir, 'K-Pol_Pinst_Default')
K_Q_instFile  = os.path.join(K_calDir, 'q_inst.fits')
K_sQ_instFile = os.path.join(K_calDir, 'q_sig_inst.fits')
K_Q_instImg   = AstroImage(K_Q_instFile)
K_sQ_instImg  = AstroImage(K_sQ_instFile)
K_Q_instImg.sigma = K_sQ_instImg.arr.copy()

# U images
K_calDir = os.path.join(PPOLcodeDir, 'K-Pol_Pinst_Default')
K_U_instFile  = os.path.join(K_calDir, 'u_inst.fits')
K_sU_instFile = os.path.join(K_calDir, 'u_sig_inst.fits')
K_U_instImg   = AstroImage(K_U_instFile)
K_sU_instImg  = AstroImage(K_sU_instFile)
K_U_instImg.sigma = K_sU_instImg.arr.copy()


del K_sQ_instImg, K_sU_instImg

# Scale these down by a factor of 100 to match raw U and Q computations
H_Q_instImg   = H_Q_instImg/100.0
H_U_instImg   = H_U_instImg/100.0
K_Q_instImg   = K_Q_instImg/100.0
K_U_instImg   = K_U_instImg/100.0


# Read in the information in the Hpol and Kpol correction factors
# First do H-band
H_calFile = os.path.join(H_calDir, 'P_inst_values.dat')
H_dPAlist = []
H_s_dPAlist = []
H_endDate = []
f = open(H_calFile, 'r')
for line in f:
    # Skip comments
    if line[0:7] == 'HISTORY': continue
    # If the line contains a key value pair, parse it
    if "=" in line:
        key, val = line.split('=')
        key = key.strip()
        val, comment = val.split('/')
        try:
            val = int(val)
        except ValueError:
            val = float(val)

        # Now decide what to to with the key value pair
        if key == 'P_EFFIC': H_PolEff     = val
        if key == 'S_P_EFF': sig_H_PolEff = val
        if key[0:5] == 'P_OFF': H_dPAlist.append(val)
        if key[0:5] == 'SP_OF': H_s_dPAlist.append(val)
        if key[0:5] == 'ENDYR': H_endDate.append(val)

del f

# Convert lists to arrays
H_dPAlist  = np.array(H_dPAlist)
H_s_dPAlist = np.array(H_s_dPAlist)
H_endDate = np.array(H_endDate)

# Repeat the same for K-band
K_calDir = os.path.join(PPOLcodeDir, 'K-Pol_Pinst_Default')
K_calFile = os.path.join(K_calDir, 'P_inst_values.dat')
K_dPAlist = []
K_s_dPAlist = []
K_endDate = []
f = open(K_calFile, 'r')
for line in f:
    # Skip comments
    if line[0:7] == 'HISTORY': continue
    # If the line contains a key value pair, parse it
    if "=" in line:
        key, val = line.split('=')
        key = key.strip()
        val, comment = val.split('/')
        try:
            val = int(val)
        except ValueError:
            val = float(val)

        # Now decide what to to with the key value pair
        if key == 'P_EFFIC': K_PolEff     = val
        if key == 'S_P_EFF': sig_K_PolEff = val
        if key[0:5] == 'P_OFF': K_dPAlist.append(val)
        if key[0:5] == 'SP_OF': K_s_dPAlist.append(val)
        if key[0:5] == 'ENDYR': K_endDate.append(val)

del f

# Convert lists to arrays
K_dPAlist  = np.array(K_dPAlist)
K_s_dPAlist = np.array(K_s_dPAlist)
K_endDate = np.array(K_endDate)

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

# Determine which parts of the fileIndex pertain to science images
useFiles = fileIndex['Use'] == 1
useFiles = np.logical_and(useFiles, fileIndex['Background'] > 0)
useFiles = np.logical_and(useFiles, fileIndex['Background Cut'] == 0)
useFiles = np.where(useFiles)

# Cull the file index to only include files selected for use
fileIndex = fileIndex[useFiles]

# Add a temporary IPPA column to the fileIndex
# Grab the HWPs
HWPs  = fileIndex['HWP'].data
IPPAs = np.zeros_like(HWPs)

for IPPA in range(1,5):
    matchingHWPs = HWPs == IPPA
    matchingHWPs = np.logical_or(matchingHWPs, HWPs == IPPA + 4)
    matchingHWPs = np.logical_or(matchingHWPs, HWPs == IPPA + 8)
    matchingHWPs = np.logical_or(matchingHWPs, HWPs == IPPA + 12)
    IPPAinds     = np.where(matchingHWPs)
    IPPAs[IPPAinds] = IPPA

# Add this column to the fileIndex
fileIndex.add_column(Column(name='IPPA', data=IPPAs))

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
# 4. HWP Angle
# TODO I need to group by "groupName", not by "Target".
# TODO This will allow us to access PPOL files easily.
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither'])

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])

    if thisWaveband == 'Ks': thisWaveband = 'K'

    if thisTarget == 'M78' or thisTarget == 'NGC1977': continue
    # if thisTarget != 'M78': continue

    # Divide the group up into its A and B images
    Ainds  = np.where(group['ABBA'].data == 'A')
    Agroup = group[Ainds]

    numImgs = len(Agroup)
    print('\nProcessing images for')
    print('\tTarget   : {0}'.format(thisTarget))
    print('\tWaveband : {0}'.format(thisWaveband))

    # Construct paths to PPOL S6 scaling factors data files
    S6file = '_'.join([thisTarget, thisWaveband, 'meta_meta_scale.dat'])
    S6path = os.path.join(S6dir, S6file)

    # Check if scaling factors file exists
    if not os.path.isfile(S6path):
        print('Could not find scaling factors for this group')
        pdb.set_trace()

    # # Read in PPOL S6 scaling factors
    # S6data = ascii.read(S6path, comment=';',
    #     names = ['Filename', 'HWPA', 'Total CF', 'Xoffset', 'Yoffset',
    #     'Dither', 'UseFlag', 'Secular CF', 'UQ CF', 'Field Pol', 'Sky CF'])

    # Read in PPOL S6 scaling factors (from meta-grouping)
    S6data = ascii.read(S6path, comment=';',
        names = ['Filename', 'HWPA', 'Total CF', 'Xoffset', 'Yoffset'])
    S6files = np.array(['_'.join(file1.split('_')[0:2])
        for file1 in S6data['Filename'].data])

    # Group this metagroup by HWP number
    IPPAgroups = Agroup.group_by('IPPA')

    # Create a dictionary to store the images for each IPPA (and positions, too)
    IPPAimgDict       = dict()
    IPPA_xPosSumDict  = dict()
    IPPA_yPosSumDict  = dict()
    IPPA_posCountDict = dict()

    # Loop through each HWP number and perform image combination for that HWP
    for IPPAgroup in IPPAgroups.groups:
        # Parse which HWP is being processed
        thisIPPA = str(np.unique(IPPAgroup['IPPA'].data)[0])
        ippa     = int(thisIPPA)

        # Read in all the images which have existing files
        imgList = []
        xPosList = []
        yPosList = []
        scaleFactors = []
        backgroundLevels = []
        for file1, bkg in zip(IPPAgroup['Filename'].data, IPPAgroup['Background'].data):
            # Costruct the path to each file
            fileBase   = os.path.basename(file1)
            bkgSubFile = os.path.join(bkgSubDir, fileBase)

            # If the file exists, then append it to the imgList
            if os.path.isfile(bkgSubFile):
                # Grab the scale factor for this image
                thisFileBasename = '.'.join(fileBase.split('.')[0:2])
                thisImgInd = np.where(S6files == thisFileBasename)
                if len(thisImgInd[0]) > 0:
                    thisScale  = (S6data['Total CF'][thisImgInd]).data[0]
                else:
                    continue

                # If the scale factor is non-zero, then incorporate it into the
                # list of images and scale factors to be used
                if thisScale != 0:
                    # Read in this image
                    tmpImg = AstroImage(bkgSubFile)

                    # Compute uncertainty array for this image
                    tmpImg.sigma = np.sqrt((bkg + tmpImg.arr)/effective_gain)

                    # Create a copy of the temporary image and replace its array
                    # with the x and y positions of the Mimir pixels
                    yImg   = tmpImg.copy()
                    xImg   = tmpImg.copy()
                    ny, nx = tmpImg.arr.shape
                    yImg.arr, xImg.arr = np.mgrid[0:ny, 0:nx]

                    # Apply the Kokopelli mask to the image
                    tmpImg.arr[np.where(kokopelliMask.arr)] = np.nan

                    # Crop all images (and position images) to avoid edges
                    tmpImg.crop(12,1012, 13, 1013)
                    xImg.crop(12, 1012, 13, 1013)
                    yImg.crop(12, 1012, 13, 1013)

                    # Append each image to its respective list
                    imgList.append(tmpImg)
                    xPosList.append(xImg)
                    yPosList.append(yImg)

                    # Append scale factor and background level to their lists
                    scaleFactors.append(thisScale)
                    backgroundLevels.append(bkg)

        # # Loop through each image in the image list and compute its uncertainty
        # imgList1 = imgList.copy()
        # for imgNum, img in enumerate(imgList):
        #     imgList1[imgNum].sigma = np.sqrt((backgroundLevels[imgNum] + img.arr)/effective_gain)
        #
        # imgList = imgList1.copy()
        # del imgList1

        # Now apply the scale factors to the images in imgList
        imgList = [CF*img for CF, img in zip(scaleFactors, imgList)]

        # Now that the background free images are stored, combine them
        # Align the images
        print('\tAligning the background removed images for IPPA {0}'.format(ippa))
        imgList = image_tools.align_images(imgList, mode='wcs', padding=np.nan)

        # Perform identical alignments for the xPos and yPos images
        xPosList = image_tools.align_images(xPosList, mode='wcs', padding=-1e6)
        yPosList = image_tools.align_images(yPosList, mode='wcs', padding=-1e6)

        # Loop through the position lists and cull any stupid numbers
        for imgNum in range(len(xPosList)):
            # Create the temporary position arrays with masks
            tmpXarr      = np.ma.array(xPosList[imgNum].arr)
            tmpYarr      = np.ma.array(yPosList[imgNum].arr)
            tmpXarr.mask = np.zeros_like(tmpXarr, dtype=bool)
            tmpYarr.mask = tmpXarr.mask.copy()

            # Find the positions of the bad values
            badInds     = np.where(np.logical_or(tmpXarr < 0, tmpYarr < 0))
            if len(badInds[0]) > 0:
                # Mask these bad values (to be filtered later)
                tmpXarr.mask[badInds] = True
                tmpYarr.mask[badInds] = True

                # Place the updated masked arrays into the image list
                xPosList[imgNum].arr = tmpXarr
                yPosList[imgNum].arr = tmpYarr

        # Combine the images
        print('\tCombining the aligned images for IPPA {0}'.format(ippa))
        IPPAimg = image_tools.combine_images(imgList, weighted_mean = True)

        # Now that the astrometry has been resolved, let's store this image
        # in the dictionary for later use
        IPPAimgDict[ippa] = IPPAimg

        # Store the sum of the position values and the count of the good positions
        xPosSum  = np.ma.array(np.zeros_like(IPPAimg.arr, dtype=int))
        yPosSum  = xPosSum.copy()
        posCount = yPosSum.data.copy()
        for imgNum in range(len(xPosList)):
            # Accumulate the x and y position values
            xPosSum += xPosList[imgNum].arr
            yPosSum += yPosList[imgNum].arr

            # Accumulate the number of samples in each pixel
            posCount += (1 - np.logical_and(xPosList[imgNum].arr.mask,
                                            yPosList[imgNum].arr.mask).astype(int))

        # Make a copy of the final image and use it to store the x and y
        # position information and accumulation of number of pixel samples
        tmpImg = IPPAimg.copy()
        tmpImg.arr = xPosSum
        IPPA_xPosSumDict[ippa] = tmpImg.copy()
        tmpImg.arr = yPosSum
        IPPA_yPosSumDict[ippa] = tmpImg.copy()
        tmpImg.arr = posCount
        IPPA_posCountDict[ippa] = tmpImg.copy()

    # Now that we have the IPPA images and uncertainties for this target/waveband
    # compute I, U, Q and their uncertainties.
    # First we need to align the images before combining them
    alignedImgs = image_tools.align_images(
        [IPPAimgDict[1], IPPAimgDict[2], IPPAimgDict[3], IPPAimgDict[4]],
        mode='cross_correlate', subPixel=True)

    # Apply a similar alignment to positinos (but we can't use cross_correlate)
    # The WCS based alignment should suffice since this is really just an
    # AVERAGE detector position anyway
    aligned_xPos = image_tools.align_images(
        [IPPA_xPosSumDict[1], IPPA_xPosSumDict[2],
        IPPA_xPosSumDict[3], IPPA_xPosSumDict[4]],
        mode = 'wcs')
    aligned_yPos = image_tools.align_images(
        [IPPA_yPosSumDict[1], IPPA_yPosSumDict[2],
        IPPA_yPosSumDict[3], IPPA_yPosSumDict[4]],
        mode = 'wcs')
    aligned_posCount = image_tools.align_images(
        [IPPA_posCountDict[1], IPPA_posCountDict[2],
        IPPA_posCountDict[3], IPPA_posCountDict[4]],
        mode = 'wcs')

    # Compute the final intensity image first
    # Combine the two stokes I estimates to get an average stokes I image
    I_13 =  alignedImgs[0] + alignedImgs[2]
    I_24 =  alignedImgs[1] + alignedImgs[3]
    Iimg = 0.5*(I_13 + I_24)

    # Solve the astrometry of this image in order to properly map mask
    # positions Aimg positions
    Iimg.filename = 'tmp.fits'
    Iimg.write()
    Iimg, success = image_tools.astrometry(Iimg, override = True)
    if success:
        print('Astrometry for initial combination solved')
        # Delete the temporary file
        if 'win' in sys.platform:
            delCmd = 'del '
            shellCmd = True
        else:
            delCmd = 'rm '
            shellCmd = False

        # Perform the actual deletion
        rmProc = subprocess.Popen(delCmd + 'tmp.fits', shell=shellCmd)
        rmProc.wait()
        rmProc.terminate()
    else:
        print('Astrometry for inital combination not successful')
        pdb.set_trace()

    # Now take the mutually aligned images and compute Stokes parameters.
    # Also transfer the astrometry in the header of the Iimg.
    Q        = (alignedImgs[0] - alignedImgs[2])/I_13
    Q.header = Iimg.header.copy()

    # Compute the average detector positions
    Q_posSum = (aligned_posCount[0] + aligned_posCount[2])
    Q_xPos   = (aligned_xPos[0] + aligned_xPos[2])/Q_posSum
    Q_yPos   = (aligned_yPos[0] + aligned_yPos[2])/Q_posSum
    # Round positions to nearest integer
    Q_xPos.arr = (Q_xPos.arr.round()).astype(int)
    Q_yPos.arr = (Q_yPos.arr.round()).astype(int)

    # Compute the U Stokes parameter and transfer astrometry in Iimg header.
    U        = (alignedImgs[1] - alignedImgs[3])/I_24
    U.header = Iimg.header.copy()

    # Compute the average detector positinos
    U_posSum = (aligned_posCount[1] + aligned_posCount[3])
    U_xPos   = (aligned_xPos[1] + aligned_xPos[3])/U_posSum
    U_yPos   = (aligned_yPos[1] + aligned_yPos[3])/U_posSum
    # Round positions to nearest integer
    U_xPos.arr = (U_xPos.arr.round()).astype(int)
    U_yPos.arr = (U_yPos.arr.round()).astype(int)

    # Figure out where to crop everything (only positions with maximal sampling)
    sampleCount = Q_posSum + U_posSum
    midX, midY  = sampleCount.arr.shape[0]//2, sampleCount.arr.shape[1]//2
    goodPix     = sampleCount.arr == np.max(sampleCount.arr)
    goodRow     = (np.where(goodPix[midY, :]))[0]
    goodCol     = (np.where(goodPix[:, midX]))[0]
    lf, rt      = goodRow.min(), goodRow.max()
    bt, tp      = goodCol.min(), goodCol.max()

    # Apply the actual crops
    Q.crop(lf, rt, bt, tp)
    U.crop(lf, rt, bt, tp)
    Q_xPos.crop(lf, rt, bt, tp)
    Q_yPos.crop(lf, rt, bt, tp)
    U_xPos.crop(lf, rt, bt, tp)
    U_yPos.crop(lf, rt, bt, tp)
    Iimg.crop(lf, rt, bt, tp)

    # Now apply the instrumental corrections
    # Start by subtracting Qinst and Uinst
    if thisWaveband == 'H':
        Qinst  = H_Q_instImg.copy()
        Uinst  = H_U_instImg.copy()
    elif thisWaveband == 'K':
        Qinst  = K_Q_instImg.copy()
        Uinst  = K_U_instImg.copy()

    # Map the average instrumental contributions and subtract them
    Qinst.arr         = Qinst.arr[Q_yPos.arr.flatten(), Q_xPos.arr.flatten()]
    Qinst.arr.shape   = Q.arr.shape
    Qinst.sigma       = Qinst.sigma[Q_yPos.arr.flatten(), Q_xPos.arr.flatten()]
    Qinst.sigma.shape = Q.sigma.shape

    Uinst.arr         = Uinst.arr[U_yPos.arr.flatten(), U_xPos.arr.flatten()]
    Uinst.arr.shape   = U.arr.shape
    Uinst.sigma       = Uinst.sigma[U_yPos.arr.flatten(), U_xPos.arr.flatten()]
    Uinst.sigma.shape = U.sigma.shape

    ####
    ####
    # Skip this for now and see if it makes a difference (it definitely should!)
    ####
    ####
    # Perform the actual subtraction
    Q1 = Q - Qinst
    U1 = U - Uinst
    # Q1 = Q.copy()
    # U1 = U.copy()

    # Now correct U and Q  for polarization efficiency
    # First we need the date of the observation.
    imgDate = (Iimg.header['Date'].split('T'))[0]
    dt      = datetime.datetime.strptime(imgDate, '%Y-%m-%d')
    tt      = dt.timetuple()
    yrFrac  = float(tt.tm_year) + float(tt.tm_yday)/365.0

    # Grab the appropriate dates, delta PA, and PE values for these images
    if thisWaveband == 'H':
        endDates  = H_endDate.copy()
        dPAlist   = H_dPAlist.copy()
        s_dPAlist = H_s_dPAlist.copy()
        PE        = H_PolEff
        s_PE      = sig_H_PolEff
    elif thisWaveband =='K':
        endDates  = K_endDate.copy()
        dPAlist   = K_dPAlist.copy()
        s_dPAlist = K_s_dPAlist.copy()
        PE        = K_PolEff
        s_PE      = sig_K_PolEff

    # See if there are ANY dates after the time of observation.
    dPA_ind = np.where(endDates > yrFrac)
    if dPA_ind[0] > 0:
        # Grab the earliest end-date after the observations
        dPA_ind = np.min(dPA_ind)
    else:
        # Otherwise just grab the latest possible date
        dPA_ind = endDates.size - 1

    # Select the best matching delta PA and uncertainty
    dPA   = dPAlist[dPA_ind]
    s_dPA = s_dPAlist[dPA_ind]

    # Add the dPA and s_dPA values into the image headers
    Q1.header['DELTAPA']   = dPA
    Q1.header['S_DPA']     = s_dPA
    U1.header['DELTAPA']   = dPA
    U1.header['S_DPA']     = s_dPA
    Iimg.header['DELTAPA'] = dPA
    Iimg.header['S_DPA']   = s_dPA

    # Scale the Stokes values and apply the uncertainty
    Qarr = Q1.arr/PE
    Qsig = np.abs(Qarr)*np.sqrt((Q1.sigma/Q1.arr)**2 + (s_PE/PE)**2)
    Uarr = U1.arr/PE
    Usig = np.abs(Uarr)*np.sqrt((U1.sigma/U1.arr)**2 + (s_PE/PE)**2)

    # Load these scaled values into images and save to disk
    Q2 = Q1.copy()
    U2 = U1.copy()
    Q2.arr   = Qarr
    Q2.sigma = Qsig
    U2.arr   = Uarr
    U2.sigma = Usig

    # Build the output filenames
    Ifile = os.path.join(stokesDir,
        '_'.join([thisTarget, thisWaveband, 'I']) + '.fits')
    Qfile = os.path.join(stokesDir,
        '_'.join([thisTarget, thisWaveband, 'Q']) + '.fits')
    Ufile = os.path.join(stokesDir,
        '_'.join([thisTarget, thisWaveband, 'U']) + '.fits')

    # Write the files to disk
    Iimg.write(Ifile)
    Q2.write(Qfile)
    U2.write(Ufile)

print('\nDone computing average images!')
