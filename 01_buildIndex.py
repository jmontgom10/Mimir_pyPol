# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:23:45 2015

@author: jordan
"""

#Import whatever modules will be used
import os
import sys
import time
import pdb
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
import astropy.coordinates as coord
import astropy.units as u
from astropy.io import fits
from scipy import stats

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage

################################################################################
# Define a recursive file search which takes a parent directory and returns all
# the FILES (not DIRECTORIES) beneath that node.
def recursive_file_search(parentDir, exten='', fileList=[]):
    # Query the elements in the directory
    subNodes = os.listdir(parentDir)

    # Loop through the nodes...
    for node in subNodes:
        # If this node is a directory,
        thisPath = os.path.join(parentDir, node)
        if os.path.isdir(thisPath):
            # then drop down recurse the function
            recursive_file_search(thisPath, exten, fileList)
        else:
            # otherwise test the extension,
            # and append the node to the fileList
            if len(exten) > 0:
                # If an extension was defined,
                # then test if this file is the right extension
                exten1 = (exten[::-1]).upper()
                if (thisPath[::-1][0:len(exten1)]).upper() == exten1:
                    fileList.append(thisPath)
            else:
                fileList.append(thisPath)

    # Return the final list to the user
    return fileList
################################################################################

#Setup the path delimeter for this operating system
delim = os.path.sep

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

# The Montgomery-Clemens reflection nebula project used the pattern
# A (on-target), B (off-target), B (off-target), A (on-target)
# To indicate a group with the opposite pattern
# A (off-target), B (on-target), B (on-target), A (off-target)
# simply include the name of the OBJECT header keyword
ABBAswap = []

# Construct a list of all the reduced BDP files (recursively suearching)
fileList = recursive_file_search(S3dir, exten='.fits')

#Sort the fileList
fileNums = [''.join((file.split(delim).pop().split('.'))[0:2]) for file in fileList]
fileNums = [file.split('_')[0] for file in fileNums]

sortInds = np.argsort(np.array(fileNums, dtype = np.int64))
fileList = [fileList[ind] for ind in sortInds]

#==============================================================================
# ***************************** INDEX *****************************************
# Build an index of the file type and binning, and write it to disk
#==============================================================================
# Check if a file index already exists... if it does then just read it in
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')

# Record a list of HWP angles to be checked
HWPlist  = np.array([34,    4556,  2261,  6784,  9011,  13534, 11306, 15761,
                    18056, 22511, 20284, 24806, 27034, 31556, 29261, 33784])
HWPlist.sort()

# Loop through each night and test for image type
print('\nCategorizing files by groups.\n')
startTime = time.time()

# Begin by initalizing some arrays to store the image classifications
telRA    = []
telDec   = []
name     = []
waveBand = []
HWPang   = []
binType  = []
expTime  = []
night    = []
fileCounter = 0
percentage  = 0

#Loop through each file in the fileList variable
for file in fileList:
    # Read in the image
    tmpImg = AstroImage(file)

    # Grab the RA and Dec from the header
    # Parse the pointing for this file
    tmpRA    = coord.Angle(tmpImg.header['TELRA'], unit=u.hour)
    tmpDec   = coord.Angle(tmpImg.header['TELDEC'], unit=u.degree)
    telRA.append(tmpRA.degree)
    telDec.append(tmpDec.degree)

    # Classify each file type and binning
    tmpName = tmpImg.header['OBJECT']
    if len(tmpName) < 1:
        tmpName = 'blank'
    name.append(tmpName)

    # Parse the HWP number
    tmpHWP  = round(100*tmpImg.header['HWP_ANG'])
    HWPdiff = np.abs(HWPlist - tmpHWP)
    tmpHWP  = (np.where(HWPdiff == np.min(HWPdiff)))[0][0] + 1
    HWPang.append(tmpHWP)

    # Parse the waveband
    waveBand.append(tmpImg.header['FILTNME2'])

    # Test the binning of this file
    binTest  = tmpImg.header['CRDELT*']
    if binTest[0] == binTest[1]:
        binType.append(int(binTest[0]))

    # Grab the night of this observation
    tmpNight = (tmpImg.header['DATE-OBS'])[0:10]
    tmpNight = tmpNight.translate({ord(i):None for i in '-'})
    night.append(tmpNight)

    # Grab the exposure time of this observation
    tmpExpTime = tmpImg.header['EXPTIME']
    expTime.append(tmpExpTime)

    # Count the files completed and print update progress message
    fileCounter += 1
    percentage1  = np.floor(fileCounter/len(fileList)*100)
    if percentage1 != percentage:
        print('completed {0:3g}%'.format(percentage1), end='\r')
    percentage = percentage1

endTime  = time.time()
numFiles = len(fileList)
print(('\n{0} File processing completed in {1:g} seconds'.
       format(numFiles, (endTime -startTime))))

# Write the file index to disk
fileIndex = Table([fileList, telRA, telDec, name, waveBand, HWPang, binType,
   expTime, night], names = ['Filename', 'RA', 'Dec', 'Name', 'Waveband',
   'HWP', 'Binning', 'Exp Time', 'Night'])
fileIndex.add_column(Column(name='Use',
                            data=np.ones((numFiles)),
                            dtype=np.int),
                            index=0)

# Group by "Name"
groupFileIndex = fileIndex.group_by('Name')

# Grab the file-number orderd indices for the groupFileIndex
fileIndices = np.argsort(groupFileIndex['Filename'])

# Loop through each "Name" and assign it a "Target" value
targetList   = []
ditherList   = []
PPOLnameList = []
for group in groupFileIndex.groups:
    # Select this groups properties
    thisName     = np.unique(group['Name'].data)
    thisWaveband = np.unique(group['Waveband'].data)
    thisExpTime  = np.unique(group['Exp Time'].data)

    # Test if the group name truely is unique
    if len(thisName) == 1:
        thisName = str(thisName[0])
    else:
        print('There is more than one name in this group!')
        pdb.set_trace()

    # Test if the waveband is truely uniqe name truely is unique
    if len(thisWaveband) == 1:
        thisWaveband = str(thisWaveband[0])
    else:
        print('There is more than one waveband in this group!')
        pdb.set_trace()

    # Test if the exposure time is truely uniqe name truely is unique
    if len(thisExpTime) == 1:
        thisExpTime = str(thisExpTime[0])
    else:
        print('There is more than one exposure time in this group!')
        pdb.set_trace()

    # Grab the file numbers for this group
    thisGroupFileNums = []
    for thisFile in group['Filename']:
        # Parse the file number for this file
        thisFileNum = os.path.basename(thisFile)
        thisFileNum = thisFileNum.split('_')[0]
        thisFileNum = int(''.join(thisFileNum.split('.')))
        thisGroupFileNums.append(thisFileNum)

    # Sort the file numbers from greatest to least
    thisGroupFileNums = sorted(thisGroupFileNums)

    # Grab the first and last image numbers
    firstImg = str(min(thisGroupFileNums))
    firstImg = firstImg[0:8] + '.' + firstImg[8:]
    lastImg  = str(max(thisGroupFileNums))
    lastImg  = lastImg[0:8] + '.' + lastImg[8:]

    # Count the number of elements in this group
    groupLen = len(group)

    # Print diagnostic information
    print('\nProcessing {0} images for'.format(groupLen))
    print('\tGroup      : {0}'.format(thisName))
    print('\tWaveband   : {0}'.format(thisWaveband))
    print('\tExptime    : {0}'.format(thisExpTime))
    print('\tFirst Img  : {0}'.format(firstImg))
    print('\tLast Img   : {0}'.format(lastImg))
    print('')

    # Add the "PPOLname"
    # thisPPOLname = input('\nEnter the PPOL name for group "{0}": '.format(thisName))
    #
    # Add the "Target" column to the fileIndex
    # thisTarget = input('\nEnter the target for group "{0}": '.format(thisName))
    #
    # Ask the user to supply the dither pattern for this group
    # thisDitherEntered = False
    # while not thisDitherEntered:
    #     # Have the user select option 1 or 2
    #     print('\nEnter the dither patttern for group "{0}": '.format(thisName))
    #     thisDither = input('[1: ABBA, 2: HEX]: ')
    #
    #     # Test if the numbers 1 or 2 were entered
    #     try:
    #         thisDither = np.int(thisDither)
    #         if (thisDither == 1) or (thisDither == 2):
    #             # If so, then reassign as a string
    #             thisDither = ['ABBA', 'HEX'][(thisDither-1)]
    #             thisDitherEntered = True
    #     except:
    #         print('Response not recognized')


    # Use the following code to skip over manual entry (comment out lines above)
    thisPPOLname = thisName
    thisTarget   = (thisName.split('_'))[0]
    thisDither   = "ABBA"

    # Add these elements to the target list
    PPOLnameList.extend([thisPPOLname]*groupLen)
    targetList.extend([thisTarget]*groupLen)
    ditherList.extend([thisDither]*groupLen)

# Add the "PPOL name" "Target" and "Dither columns"
groupFileIndex.add_column(Column(name='Target',
                            data = np.array(targetList)),
                            index = 2)
groupFileIndex.add_column(Column(name='PPOL Name',
                            data = np.array(PPOLnameList)),
                            index = 3)
groupFileIndex.add_column(Column(name='Dither',
                            data = np.array(ditherList)),
                            index = 7)

# Re-sort by file-number
fileSortInds = np.argsort(groupFileIndex['Filename'])
fileIndex1   = groupFileIndex[fileSortInds]

#==============================================================================
# ************************** ABBA PARSER **************************************
# Loop through all the groups in the index and parse the ABBA dithers
#==============================================================================
ABBAlist = np.repeat('X', len(fileIndex1))

fileIndexByName = fileIndex1.group_by(['PPOL Name'])
ABBAlist = []
for key, group in zip(fileIndexByName.groups.keys, fileIndexByName.groups):
    print('\nParsing ABBA values for PPOL group ', key['PPOL Name'])
    # For each of each group file, we will need two pieces of information
    # 1) The FILENUMBER (essentially the date plus the nightly file number)
    # 2) The HWP_ANGLE (the rotation of the HWP)
    # Using this information, we can parse which files are A vs. B

    # For later reference, let's group the group name
    thisName = np.unique(group['Name'].data)

    # Grab the file numbers for this group
    thisGroupFileNums = [''.join((file.split(delim).pop().split('.'))[0:2])
        for file in group['Filename'].data]
    thisGroupFileNums = [int(file.split('_')[0]) for file in thisGroupFileNums]
    thisGroupFileNums = np.array(thisGroupFileNums)

    # Grab the HWP, RA, and Decs for this group
    thisGroupHWPs     = group['HWP'].data
    thisGroupRAs      = group['RA'].data
    thisGroupDecs     = group['Dec'].data

    # Compute the incremental step for each image in the sequence
    numIncr = thisGroupFileNums - np.roll(thisGroupFileNums, 1)
    numIncr[0] = 1

    # Check if the mean increment is about 2, and skip if it is...
    meanIncr = (10.0*np.mean(numIncr))/10.0
    skipGroup = False
    if (meanIncr >= 1.85) and (meanIncr <= 2.15):
        print(key['Name'] + 'appears to already have been parsed')
        skipGroup = True

    if skipGroup: continue

    # Find where the HWP changes
    # This is not quite the right algorithm anymore...
    HWPshifts = (thisGroupHWPs != np.roll(thisGroupHWPs, 1))
    HWPshifts[0] = False

    if np.sum(HWPshifts) < 0.5*len(group):
        #****************************************************
        # This group has  the 16(ABBA) dither type.
        #****************************************************
        print('Dither type 16(ABBA)')
        # Find places where the HWP change corresponds to a numIncr of 1
        ABBArestart = np.logical_and(HWPshifts, (numIncr == 1))

        if np.sum(ABBArestart) > 0:
            # Check for the index of these ABBA restart points
            ABBArestartInd = np.where(ABBArestart)
            # Find the index of the first restart
            firstRestartInd = np.min(ABBArestartInd)
            # Find the amount of shift needed to coincide ABBAinds with ABBArestart
            numSkips  = round(np.sum(numIncr[0:firstRestartInd])) - firstRestartInd
            ABBAshift = (64 - (firstRestartInd + numSkips)) % 4
            ABBAinds  = (thisGroupFileNums - thisGroupFileNums[0] + ABBAshift) % 4
        else:
            print('The HWP shifts are not well mapped, so a solution is not possible.')
            pdb.set_trace()

        # Setup the dither pattern array
        if thisName in ABBAswap:
            # Setup the reverse ABBA array
            print('Using reverse ABBA values for this group')
            ABBAarr = np.array(['B','A','A','B'])
        else:
            # Setup the normal ABBA array
            ABBAarr = np.array(['A','B','B','A'])

        # Grab the ABBA values for each file
        thisGroupABBAs = ABBAarr[ABBAinds]

        # Parse the indices for A images and B images
        Ainds = np.where(thisGroupABBAs == 'A')
        Binds = np.where(thisGroupABBAs == 'B')

    else:
        #****************************************************
        # This group has  the (16A, 16B, 16B, 16A) dither type.
        #****************************************************
        print('Dither type (16A, 16B, 16B, 16A)')

        # Setup the group dither pattern array (16*A, 16*B, 16*B, 16*A)
        As = np.repeat('A', 16)
        Bs = np.repeat('B', 16)

        if thisName in ABBAswap:
            # Setup the reverse ABBA array
            print('Using reverse ABBA values for this group')
            ABBAarr = np.array([Bs, As, As, Bs]).flatten()
        else:
            # Setup the normal ABBA array
            ABBAarr = np.array([As, Bs, Bs, As]).flatten()

        # Figure out if any of the first images were dropped
        HWPorder = np.array([1,3,2,4,5,7,6,8,9,11,10,12,13,15,14,16])
        firstHWP = (np.where(HWPorder == thisGroupHWPs[0]))[0][0]

        # Determine which ABBAinds to use
        HWPdiff     = np.abs(HWPlist - thisGroupHWPs[0])
        firstHWPind = np.where(HWPdiff == np.min(HWPdiff))
        ABBAinds    = thisGroupFileNums - thisGroupFileNums[0] + firstHWP

        # Grab the ABBA values for each file
        thisGroupABBAs = ABBAarr[ABBAinds]

        # Parse the indices for A images and B images
        Ainds = np.where(thisGroupABBAs == 'A')
        Binds = np.where(thisGroupABBAs == 'B')

    # Double check that the pointing for each group is correct.
    outliersPresent = True
    while outliersPresent:
        # Compute the median pointings for A and B dithers
        A_medRA  = np.median(thisGroupRAs[Ainds])
        A_medDec = np.median(thisGroupDecs[Ainds])
        B_medRA  = np.median(thisGroupRAs[Binds])
        B_medDec = np.median(thisGroupDecs[Binds])

        # Compute the (RA, Dec) offsets from the median pointings
        A_delRA  = thisGroupRAs[Ainds] - A_medRA
        A_delDec = thisGroupDecs[Ainds] - A_medDec
        B_delRA  = thisGroupRAs[Binds] - B_medRA
        B_delDec = thisGroupDecs[Binds] - B_medDec

        # Search for outliers in either RA **OR** Dec
        # (more than 1 arcmin off median pointing).
        A_RA_out  = np.abs(A_delRA) > 1.0/60.0
        A_Dec_out = np.abs(A_delDec) > 1.0/60.0
        B_RA_out  = np.abs(B_delRA) > 1.0/60.0
        B_Dec_out = np.abs(B_delDec) > 1.0/60.0

        # Set a flag to determine if there are still any outliers
        outliersPresent = (np.sum(np.logical_or(A_RA_out, A_Dec_out)) +
                           np.sum(np.logical_or(B_RA_out, B_Dec_out)) > 0)

        # If there **DO** still seem to be outliers present,
        # then swap offending images between groups.
        if outliersPresent:
            print('Repairing pointing outliers')
            pdb.set_trace()
            # First identify offending images from each group
            A_out = np.logical_or(A_RA_out, A_Dec_out)
            B_out = np.logical_or(B_RA_out, B_Dec_out)

            # Now identify which of the Aind and Binds need to be swapped
            if np.sum(A_out) > 0:
                AswapInds = Ainds[np.where(A_out)]
                AkeepInds = Ainds[np.where(np.logical_not(A_out))]
            if np.sum(B_out) > 0:
                BswapInds = Binds[np.where(B_out)]
                BkeepInds = Binds[np.where(np.logical_not(B_out))]

            # Reconstruct the Ainds and Binds arrays
            Ainds = np.concatenate([AkeepInds, BswapInds])
            Binds = np.concatenate([BkeepInds, AswapInds])

            # Sort the newly constructed Ainds and Binds arrays
            AsortArr = SORT(Ainds)
            Ainds    = Ainds[AsortArr]
            BsortArr = SORT(Binds)
            Binds    = Binds[BsortArr]

            # Count the number of images in each group
            AimgCount = N_ELEMENTS(Ainds)
            BimgCount = N_ELEMENTS(Binds)

    # *************************************
    # Now that we have checked for errors,
    # add these ABBA values to the ABBAlist
    # *************************************
    ABBAlist.extend(thisGroupABBAs)

# Now that we have the indices for A and B images for this group,
# we need to add them to the column to be added to the file index
fileIndexByName.add_column(Column(name='ABBA',
                           data=np.array(ABBAlist)),
                           index = 8)

# Re-sort by file-number
fileSortInds = np.argsort(fileIndexByName['Filename'])
fileIndex1   = fileIndexByName[fileSortInds]

#==============================================================================
# ********************* Write the file to disk ********************************
# Now that all the information for this dataset has been parsed,
# write the full index to disk.
#==============================================================================
print('')
print('***************************')
print('Writing final index to disk')
print('***************************')
fileIndex1.write(indexFile, format='csv')
