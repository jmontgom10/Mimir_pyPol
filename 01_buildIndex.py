# -*- coding: utf-8 -*-
"""
Restructures the rawFileIndex from PRISM_pyBDP to contain ONLY the science
images, and break those up into individual groups based on changes in

1) OBJECT (object name)
2) FILTER (optical filter value)
3) EXPTIME (the exposure time of the images)
4) Pointing changes (more than 1.5 degrees of chang is considered a new group)

Attempts to associate each group with a target in the 'targetList' variable on
the basis of the string in the OBJECT column of that group.

Saves the index file with a USE and GROUP_ID columns added to the table.
"""

#Import whatever modules will be used
import os
import sys
import time
import re
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats

# Add the AstroImage class
import astroimage as ai

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================

# This is the location of all pyBDP data (index, calibration images, reduced...)
BDP_data='C:\\Users\\Jordan\\FITS_data\\Mimir_data\\BDP_Data\\201611'

# Set the directory for the PPOL reduced data
PPOL_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_Reduced\\201611\\'

# This is the location where all pyPol data will be saved
pyPol_data='C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\'
if (not os.path.isdir(pyPol_data)):
    os.mkdir(pyPol_data, 0o755)

# Set the filename for the reduced data indexFile
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')

# Compose a list of expected targets. All groups will be assigned to ONE of
# these targets within a given tolerance. If no match is found for a group
# within this list of targets, then an error will be raised.
targetList = [
    'M78',
    'NGC7023',
    'NGC2023',
    'Orion_Cal',
    'Cyg_OB2',
    'Elias'
]

# Create a dictionary with known group name problems. They keys should be the
# name of the group as it is currently, and the value should be the name of the
# group as it *ought* to be.
problematicGroupNames = {'NGC723_H3': 'NGC7023_H3'}

# Force all the targets to be upper case to remove ambiguity
targetList = [t.upper() for t in targetList]

# ################################################################################
# # Define a recursive file search which takes a parent directory and returns all
# # the FILES (not DIRECTORIES) beneath that node.
# def recursive_file_search(parentDir, exten='', fileList=[]):
#     # Query the elements in the directory
#     subNodes = os.listdir(parentDir)
#
#     # Loop through the nodes...
#     for node in subNodes:
#         # If this node is a directory,
#         thisPath = os.path.join(parentDir, node)
#         if os.path.isdir(thisPath):
#             # then drop down recurse the function
#             recursive_file_search(thisPath, exten, fileList)
#
#         else:
#             fileDirName = os.path.basename(
#                 os.path.normpath(
#                     os.path.dirname(thisPath)
#                 )
#             )
#             # *** ONLY ACCEPT FILES IN THE 'polarimetry' DIRECTORIES ***
#             if fileDirName ==  'polarimetry':
#                 # otherwise test the extension,
#                 # and append the node to the fileList
#                 if len(exten) > 0:
#                     # If an extension was defined,
#                     # then test if this file is the right extension
#                     exten1 = (exten[::-1]).upper()
#                     if (thisPath[::-1][0:len(exten1)]).upper() == exten1:
#                         fileList.append(thisPath)
#                 else:
#                     fileList.append(thisPath)
#             else: pass
#
#     # Return the final list to the user
#     return fileList



# Generate a list of files in the 'polarimetry' directories
# fileList = np.array(recursive_file_search(BDP_data, exten='.fits'))
S3_dir   = os.path.join(PPOL_data, 'S3_Astrometry')
fileList = np.array(os.listdir(S3_dir))

#Sort the fileList
fileNums = [''.join((os.path.basename(f).split('.'))[0:2]) for f in fileList]
fileNums = np.array([f.split('_')[0] for f in fileNums], dtype=np.int64)
sortInds = fileNums.argsort()
fileList = fileList[sortInds]

# Test for image type
print('\nCategorizing files into TARGET, HWP, BAAB\n')
startTime = time.time()
# Begin by initalizing some arrays to store the image classifications
OBJECT   = []
OBSTYPE  = []
FILTER   = []
TELRA    = []
TELDEC   = []
EXPTIME  = []
HWP      = []
AB       = []
NIGHT    = []
percentage = 0

#Loop through each file in the fileList variable
numberOfFiles = len(fileList)
for iFile, filename in enumerate(fileList):
    # Read in the file
    thisHDU    = fits.open(os.path.join(S3_dir, filename))
    thisHeader = thisHDU[0].header

    # Grab the OBJECT header value
    tmpOBJECT = thisHeader['OBJECT']
    if len(tmpOBJECT) < 1:
        tmpOBJECT = 'blank'
    OBJECT.append(tmpOBJECT)

    # Grab the OBSTYPE header value
    OBSTYPE.append(thisHeader['OBSTYPE'])

    # Grab the FILTNME3 header value
    FILTER.append(thisHeader['FILTNME2'])

    # Grab the TELRA header value
    try:
        TELRA.append(thisHeader['TELRA'])
    except:
        TELRA.append(0)

    try:
        TELDEC.append(thisHeader['TELDEC'])
    except:
        TELDEC.append(0)

    # Grab the HWP header value
    HWP.append(thisHeader['HWP'])

    # Search for the A-pos B-pos value
    if 'COMMENT' in thisHeader:
        # This is SUPER lazy, but it gets the job done
        for thisComment in thisHeader['COMMENT']:
            if 'HWP' in thisComment and 'posn' in thisComment:
                thisAB = thisComment[-6]
            else:
                thisAB = 'A'
    else:
        thisAB = 'A'

    # Append the AB value to the list
    if thisAB == 'A' or thisAB == 'B':
        AB.append(thisAB)
    else:
        # Oops... something went wrong. You should only have As or Bs
        import pdb; pdb.set_trace()

    # Grab the EXPTIME value from the header
    EXPTIME.append(thisHeader['EXPTIME'])

    # Assign a NIGHT value for this image
    NIGHT.append(''.join((os.path.basename(filename).split('.'))[0]))

    # Count the files completed and print update progress message
    percentage1  = np.floor(100*iFile/numberOfFiles)
    if percentage1 != percentage:
        print('completed {0:3g}%'.format(percentage1), end="\r")
    percentage = percentage1

print('completed {0:3g}%'.format(100), end="\r")
endTime = time.time()
print('\nFile processing completed in {0:g} seconds'.format(endTime - startTime))

# Query the user about the targets of each group...
# Write the file index to disk
reducedFileIndex = Table([ fileList,   NIGHT,   OBSTYPE,   OBJECT,   FILTER,
                    TELRA,   TELDEC,   EXPTIME,   HWP,   AB],
          names = ['FILENAME', 'NIGHT', 'OBSTYPE', 'OBJECT', 'FILTER',
                   'TELRA', 'TELDEC', 'EXPTIME', 'HWP', 'AB'])

# Remap the filenames to be the reduced filenames
fileBasenames    = [os.path.basename(f) for f in reducedFileIndex['FILENAME']]
# reducedFilenames = [os.path.join(pyBDP_reducedDir, f) for f in fileBasenames]
# reducedFileIndex['FILENAME'] = reducedFilenames

# Find the breaks in observation procedure. These are candidates for group
# boundaries.
# 1) OBJECT changes
objectChange = (reducedFileIndex['OBJECT'][1:] != reducedFileIndex['OBJECT'][0:-1])

# 2) OBSTYPE changes
obstypeChange = (reducedFileIndex['OBSTYPE'][1:] != reducedFileIndex['OBSTYPE'][0:-1])

# 3) FILTER changes
filterChange = (reducedFileIndex['FILTER'][1:] != reducedFileIndex['FILTER'][0:-1])

# 4) EXPTIME changes
expTimeChange = (reducedFileIndex['EXPTIME'][1:] != reducedFileIndex['EXPTIME'][0:-1])

# 5) Pointing changes
# Look for any pointing differences 1.5 degree (or more) for further separations
allPointings   = SkyCoord(
    reducedFileIndex['TELRA'],
    reducedFileIndex['TELDEC'],
    unit=(u.hour, u.degree)
)
medianDecs     = 0.5*(allPointings[1:].ra.to(u.rad) + allPointings[0:-1].ra.to(u.rad))
deltaDec       = allPointings[1:].dec - allPointings[0:-1].dec
deltaRA        = (allPointings[1:].ra - allPointings[0:-1].ra)*np.cos(medianDecs)
deltaPointing  = np.sqrt(deltaRA**2 + deltaDec**2)
pointingChange = deltaPointing > (1.5*u.deg)

# Identify all changes
allChanges = objectChange
allChanges = np.logical_or(allChanges, obstypeChange)
allChanges = np.logical_or(allChanges, filterChange)
allChanges = np.logical_or(allChanges, expTimeChange)
allChanges = np.logical_or(allChanges, pointingChange)

# Assign a GROUP_ID for each group
groupBoundaries = np.hstack([0, np.where(allChanges)[0] + 1, allChanges.size])
groupIDs        = []
for i in range(groupBoundaries.size - 1):
    # Find the start and end indices of the group
    groupStartInd = groupBoundaries[i]
    groupEndInd   = groupBoundaries[i+1]

    # Build the gorup ID number
    groupID = i + 1

    # Count the number of images in this group
    numberOfImages = groupEndInd - groupStartInd

    # Build the list of ID numbers for THIS group and append it to the full list
    thisGroupID = numberOfImages*[groupID]
    groupIDs.extend(thisGroupID)

# Fill in the final entry
groupIDs.append(groupID)

# Store the groupID number in the reducedFileIndex
groupIDcolumn = Column(name='GROUP_ID', data=groupIDs)
reducedFileIndex.add_column(groupIDcolumn, index=2)

# Now remove any GROUPS with less than 8 images
groupIndex = reducedFileIndex.group_by('GROUP_ID')
goodGroupInds = []
groupInds = groupIndex.groups.indices
for startInd, endInd in zip(groupInds[:-1], groupInds[+1:]):
    # Count the number of images in this group and test if it's any good.
    if (endInd - startInd) >= 8:
        goodGroupInds.extend(range(startInd, endInd))


# Cull the reducedFileIndex to only include viable groups
goodGroupInds    = np.array(goodGroupInds)
reducedFileIndex = reducedFileIndex[goodGroupInds]

# Match a dither type for each group ("ABBA" or "HEX")
groupIndex = reducedFileIndex.group_by('GROUP_ID')
ditherType = []
for group in groupIndex.groups:
    # Count the number of images in this group
    numberOfImages = len(group)

    # Test if this is an ABBA or HEX dither
    if ('A' in group['AB']) and ('B' in group['AB']):
        ditherType.extend(numberOfImages*['ABBA'])
    if ('A' in group['AB']) and not ('B' in group['AB']):
        ditherType.extend(numberOfImages*['HEX'])

# Store the ditherNames number in the reducedFileIndex
ditherTypeColumn = Column(name='DITHER_TYPE', data=ditherType)
groupIndex.add_column(ditherTypeColumn, index=10)

# Identify meta-groups pointing at a single target with a single dither style.
targets = []
for group in groupIndex.groups:
    # Count the number of images in this group
    numberOfImages = len(group)

    # Get the group name
    groupName = np.unique(group['OBJECT'])[0]

    # Capitalize the group name to remove ambiguity
    groupName = groupName.upper()

    # Rename the group if it needs to be renamed
    if groupName in problematicGroupNames:
        groupName = problematicGroupNames[groupName]

    # Test it a target name occurs in this group name
    for target in targetList:
        if target in groupName:
            targets.extend(numberOfImages*[target])

            break
    else:
        import pdb; pdb.set_trace()
        raise ValueError('Gorup {} found no match in the target list'.format(groupName))

# Add the target identifications to the groupIndex table
targetColumn = Column(name='TARGET', data=targets)
groupIndex.add_column(targetColumn, index=5)

# Re-order by filename. Start by getting the sorting array
sortInds = groupIndex['FILENAME'].data.argsort()
reducedFileIndex = groupIndex[sortInds]

# Finally, add a column of "use" flags at the first index
useColumn = Column(name='USE', data=np.ones((len(reducedFileIndex),), dtype=int))
reducedFileIndex.add_column(useColumn, index=0)

# Save the index to disk.
reducedFileIndex.write(indexFile, format='ascii.csv', overwrite=True)

print('Done!')
