# Allows the user to view the constructed HWP bacxground images. Shows all four
# HWP rotations associated with a single IPPA angle
#

import os
import sys
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from photutils import detect_threshold, detect_sources
from scipy.ndimage.filters import median_filter, gaussian_filter
import matplotlib.pyplot as plt

# Add the AstroImage class
import astroimage as ai

# Add the header handler to the BaseImage class
from Mimir_header_handler import Mimir_header_handler
ai.reduced.ReducedScience.set_header_handler(Mimir_header_handler)
ai.set_instrument('mimir')

# This is the location of all PPOL reduction directory
PPOL_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_reduced\\201611'

# Build the path to the S3_Asotrometry files
S3_dir = os.path.join(PPOL_dir, 'S3_Astrometry')

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\'

# Build the path to the supersky directory
bkgImagesDir = os.path.join(pyPol_data, 'bkgImages')

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

################################################################################
# Determine which parts of the fileIndex pertain to science images
useFiles = np.where(fileIndex['USE'] == 1)

# Cull the file index to only include files selected for use
fileIndex = fileIndex[useFiles]

# Group the index by GROUP_ID and IPPA
fileIndexByGroup = fileIndex.group_by(['GROUP_ID', 'AB'])


# Define a dictionary for translating HWP rotation into IPPA
HWPlist     = (
    4*(np.arange(16, dtype=int).reshape((4,4))%4) +
    np.arange(4,dtype=int).reshape((4,1)) + 1
)
IPPAlist    = np.array([0, 45, 90, 135])
IPPA_to_HWP = dict(zip(IPPAlist, HWPlist))

groupDict = {}
# Loop through each group
for group in fileIndexByGroup.groups:
    # Test if it was an ABBA/BAAB dither
    thisDither = str(np.unique(group['DITHER_TYPE'].data)[0])
    if thisDither != 'ABBA': continue

    thisGroupName = str(np.unique(group['OBJECT'].data)[0])
    thisGroupID   = str(np.unique(group['GROUP_ID'].data)[0])
    ippaDict = {}
    # Loop through each IPPA/HWP pairing in this group
    for ippa, hwps in IPPA_to_HWP.items():
        # Loop through all the HWPs in this group and build a list of filenames
        hwpImageFiles = []
        for hwp in hwps:
            hwpFile = os.path.join(
                bkgImagesDir,
                '{}_G{}_HWP{}.fits'.format(thisGroupName, thisGroupID, str(hwp))
            )
            hwpImageFiles.append(hwpFile)

        # Store the HWPs in the corresponding ippaDict entry
        ippaDict[ippa] = hwpImageFiles

    # Add the IPPA dictionary to the groupDict
    groupKey = '{}_G{}'.format(thisGroupName, thisGroupID)
    groupDict[groupKey] = ippaDict

# Grab all the keys and alphabetically sort them for consistency
groupKeyList = list(groupDict.keys())
groupKeyList.sort()

################################################################################
# Define a function to handle what sholud be done whenever a key is pressed
################################################################################
def on_key(event):
    global groupDict, groupKeyList, IPPAlist, fig, IPPA_num, group_num
    global HWP_0_img, HWP_1_img, HWP_2_img, HWP_3_img
    global HWP_0_AxImg, HWP_1_AxImg, HWP_2_AxImg, HWP_3_AxImg
    # global prevTarget, thisTarget, nextTarget
    global HWP_0_Label, HWP_1_Label, HWP_2_Label, HWP_3_Label

    # Increment the image number
    if event.key == 'right' or event.key == 'left':
        if event.key == 'right':
            #Advance to the next IPPA
            IPPA_num += 1

            # If there are no more IPPAs left in this group, then move to the
            # next group.
            if IPPA_num > IPPAlist.size - 1:
                IPPA_num   = 0
                group_num += 1

                # If there are no more groups left, reloop back to zero
                if group_num > (len(groupKeyList) - 1):
                    group_num = 0

                if group_num < (1 - len(groupKeyList)):
                    group_num = 0

            # Having incremented the group index and the IPPA index, it's time
            # to grab the current groupDict entry
            thisGroupKey = groupKeyList[group_num]
            thisIPPA     = IPPAlist[IPPA_num]
            thisHWPfiles = groupDict[thisGroupKey][thisIPPA]

            # Loop through all the for this group/IPPA and read them in
            try:
                HWP_0_img = ai.reduced.ReducedScience.read(thisHWPfiles[0])
            except:
                HWP_0_img = ai.reduced.ReducedScience(np.ones(HWP_0_img.shape))

            try:
                HWP_1_img = ai.reduced.ReducedScience.read(thisHWPfiles[1])
            except:
                HWP_1_img = ai.reduced.ReducedScience(np.ones(HWP_1_img.shape))

            try:
                HWP_2_img = ai.reduced.ReducedScience.read(thisHWPfiles[2])
            except:
                HWP_2_img = ai.reduced.ReducedScience(np.ones(HWP_2_img.shape))

            try:
                HWP_3_img = ai.reduced.ReducedScience.read(thisHWPfiles[3])
            except:
                HWP_3_img = ai.reduced.ReducedScience(np.ones(HWP_3_img.shape))

        if event.key == 'left':
            #Advance to the next IPPA
            IPPA_num -= 1

            # If there are no more IPPAs left in this group, then move to the
            # next group.
            if IPPA_num < 0:
                IPPA_num   = IPPAlist.size - 1
                group_num -= 1

            # Having incremented the group index and the IPPA index, it's time
            # to grab the current groupDict entry
            thisGroupKey = groupKeyList[group_num]
            thisIPPA     = IPPAlist[IPPA_num]
            thisHWPfiles = groupDict[thisGroupKey][thisIPPA]

            # Loop through all the for this group/IPPA and read them in
            try:
                HWP_0_img = ai.reduced.ReducedScience.read(thisHWPfiles[0])
            except:
                HWP_0_img = ai.reduced.ReducedScience(np.ones(HWP_0_img.shape))

            try:
                HWP_1_img = ai.reduced.ReducedScience.read(thisHWPfiles[1])
            except:
                HWP_1_img = ai.reduced.ReducedScience(np.ones(HWP_1_img.shape))

            try:
                HWP_2_img = ai.reduced.ReducedScience.read(thisHWPfiles[2])
            except:
                HWP_2_img = ai.reduced.ReducedScience(np.ones(HWP_2_img.shape))

            try:
                HWP_3_img = ai.reduced.ReducedScience.read(thisHWPfiles[3])
            except:
                HWP_3_img = ai.reduced.ReducedScience(np.ones(HWP_3_img.shape))

        ###############################
        # Update the displayed images
        ###############################
        # Display the new images
        HWP_0_AxImg.set_data(HWP_0_img.data)
        HWP_1_AxImg.set_data(HWP_1_img.data)
        HWP_2_AxImg.set_data(HWP_2_img.data)
        HWP_3_AxImg.set_data(HWP_3_img.data)

        # Update the annotation
        thisTitle =  fig.suptitle('{}: IPPA {}'.format(thisGroupKey, thisIPPA))

        # Construct the label strings
        HWP_0_str, HWP_1_str, HWP_2_str, HWP_3_str = (
            ['HWP {}'.format(hwp) for hwp in IPPA_to_HWP[thisIPPA]]
        )

        # Update the labels
        HWP_0_Label.set_text(HWP_0_str)
        HWP_1_Label.set_text(HWP_1_str)
        HWP_2_Label.set_text(HWP_2_str)
        HWP_3_Label.set_text(HWP_3_str)

        ###############################
        # Update time series plot
        ###############################
        # Now plot the timeseries for this dataset
        # Grab the group ID for the current group/IPPA
        thisGroupID  = int(thisGroupKey.split('_')[-1][1:])

        # Locate the MJD and background values for the current group/IPPA
        thisGroupIPPAbool = np.logical_and(
            fileIndex['GROUP_ID'] == thisGroupID,
            fileIndex['IPPA'] == thisIPPA
        )
        thisAbool = np.logical_and(
            thisGroupIPPAbool,
            fileIndex['AB'] == 'A'
        )
        thisBbool = np.logical_and(
            thisGroupIPPAbool,
            fileIndex['AB'] == 'B'
        )
        thisAinds = np.where(thisAbool)
        thisBinds = np.where(thisBbool)
        thisAmjd  = fileIndex['MJD'][thisAinds]
        thisBmjd  = fileIndex['MJD'][thisBinds]

        # Make an estimate of the first time stamp
        try:
            mjd0 = np.min([np.min(thisAmjd), np.min(thisBmjd)])
        except:
            mjd0 = 0

        thisAmjd -= mjd0
        thisBmjd -= mjd0
        thisAmjd *= 24*60*60
        thisBmjd *= 24*60*60
        thisAbkg  = fileIndex['BACKGROUND'][thisAinds]
        thisBbkg  = fileIndex['BACKGROUND'][thisBinds]

        # Identify filter
        thisFilter = np.unique(fileIndex[thisGroupIPPAbool]['FILTER'].data)[0]
        if thisFilter == 'H':
            ylims = (600, 2100)
        if thisFilter == 'Ks':
            ylims = (400, 1000)

        # Plot the background values
        ax4.cla()
        ax4.plot(thisBmjd, thisBbkg, marker='o', color='b')#, facecolor='b', edgecolor='k')
        ax4.plot(thisAmjd, thisAbkg, marker='o', color='r')#, facecolor='r', edgecolor='k')
        plt.setp(ax4.get_xticklabels(), fontsize = 6)
        plt.setp(ax4.get_yticklabels(), fontsize = 6)
        ax4.set_ylim((ylims))
        ax4.set_ylabel('Background Counts [ADU]')
        ax4.set_xlabel('Time [sec]')

        # Update the display
        fig.canvas.draw()


#******************************************************************************
# This script will run the mask building step of the pyPol reduction
#******************************************************************************
fig = plt.figure(figsize=(18,9))

# Create the first axis and make the x-axis labels invisible
ax0 = plt.subplot(2,4,1)
plt.setp(ax0.get_xticklabels(), visible = False)
plt.setp(ax0.get_yticklabels(), fontsize = 6)

# Create the second axis and make the x- and y-axis labels invisible
ax1 = plt.subplot(2,4,2, sharey=ax0, sharex=ax0)
plt.setp(ax1.get_xticklabels(), visible = False)
plt.setp(ax1.get_yticklabels(), visible = False)

# Create the third axis and make both axis labels visible
ax2 = plt.subplot(2,4,5, sharey=ax0, sharex=ax0)
plt.setp(ax2.get_xticklabels(), fontsize = 6)
plt.setp(ax2.get_yticklabels(), fontsize = 6)

# Create the fourth axis and make y-axis labels invisible
ax3 = plt.subplot(2,4,6, sharey=ax0, sharex=ax0)
plt.setp(ax3.get_xticklabels(), fontsize = 6)
plt.setp(ax3.get_yticklabels(), visible = False)

# Create the final plot for the time-series
ax4 = plt.subplot(1,2,2)
ax4.yaxis.set_label_position('right')
ax4.tick_params(axis='y',
     labelleft=False, labelright=True,
     )

# Rescale the figure and setup the spacing between images
plt.subplots_adjust(left = 0.04, bottom = 0.04, right = 0.95, top = 0.96,
    wspace = 0.02, hspace = 0.02)

axarr = [ax0, ax1, ax2, ax3, ax4]

# Initalize the group and IPPA index at zero
IPPA_num, group_num = 0, 0

# Start by grabbing the corresponding group names and IPPAs for those indices
thisGroupKey = groupKeyList[group_num]
thisIPPA     = IPPAlist[IPPA_num]
thisHWPfiles = groupDict[thisGroupKey][thisIPPA]

# Loop through all the for this group/IPPA and read them in
HWP_0_img = ai.reduced.ReducedScience.read(thisHWPfiles[0])
HWP_1_img = ai.reduced.ReducedScience.read(thisHWPfiles[1])
HWP_2_img = ai.reduced.ReducedScience.read(thisHWPfiles[2])
HWP_3_img = ai.reduced.ReducedScience.read(thisHWPfiles[3])

# Populate each axis with its image
HWP_0_AxImg = HWP_0_img.show(axes = axarr[0], cmap='viridis',
                                        vmin = 0.95, vmax = 1.05, noShow = True)
HWP_1_AxImg = HWP_1_img.show(axes = axarr[1], cmap='viridis',
                                        vmin = 0.95, vmax = 1.05, noShow = True)
HWP_2_AxImg = HWP_2_img.show(axes = axarr[2], cmap='viridis',
                                        vmin = 0.95, vmax = 1.05, noShow = True)
HWP_3_AxImg = HWP_3_img.show(axes = axarr[3], cmap='viridis',
                                        vmin = 0.95, vmax = 1.05, noShow = True)

# Now plot the timeseries for this dataset
# Grab the group ID for the current group/IPPA
thisGroupID  = int(thisGroupKey.split('_')[-1][1:])

# Locate the MJD and background values for the current group/IPPA
thisGroupIPPAbool = np.logical_and(
    fileIndex['GROUP_ID'] == thisGroupID,
    fileIndex['IPPA'] == thisIPPA
)
thisAbool = np.logical_and(
    thisGroupIPPAbool,
    fileIndex['AB'] == 'A'
)
thisBbool = np.logical_and(
    thisGroupIPPAbool,
    fileIndex['AB'] == 'B'
)
thisAinds = np.where(thisAbool)
thisBinds = np.where(thisBbool)
thisAmjd  = fileIndex['MJD'][thisAinds]
thisBmjd  = fileIndex['MJD'][thisBinds]
mjd0      = np.min([np.min(thisAmjd), np.min(thisBmjd)])
thisAmjd -= mjd0
thisBmjd -= mjd0
thisAmjd *= 24*60*60
thisBmjd *= 24*60*60
thisAbkg  = fileIndex['BACKGROUND'][thisAinds]
thisBbkg  = fileIndex['BACKGROUND'][thisBinds]

# Identify filter
thisFilter = np.unique(fileIndex[thisGroupIPPAbool]['FILTER'].data)[0]
if thisFilter == 'H':
    ylims = (600, 2100)
if thisFilter == 'Ks':
    ylims = (400, 1000)

# Plot the background values
ax4.plot(thisBmjd, thisBbkg, marker='o', color='b')#, facecolor='b', edgecolor='k')
ax4.plot(thisAmjd, thisAbkg, marker='o', color='r')#, facecolor='r', edgecolor='k')
plt.setp(ax4.get_xticklabels(), fontsize = 6)
plt.setp(ax4.get_yticklabels(), fontsize = 6)
ax4.set_ylim((ylims))

# Add timeseries axis labels
ax4.set_ylabel('Background Counts [ADU]')
ax4.set_xlabel('Time [sec]')

# Add some labels to each plot
HWP_0_str, HWP_1_str, HWP_2_str, HWP_3_str = (
    ['HWP {}'.format(hwp) for hwp in IPPA_to_HWP[thisIPPA]]
)

# Add some figure annotation
thisTitle = fig.suptitle('{}: IPPA {}'.format(thisGroupKey, thisIPPA))

# Construct the label strings
HWP_0_str, HWP_1_str, HWP_2_str, HWP_3_str = (
    ['HWP {}'.format(hwp) for hwp in IPPA_to_HWP[thisIPPA]]
)

# Update the labels
HWP_0_Label = axarr[0].text(20, 875, HWP_0_str,
    color = 'black', backgroundcolor = 'white', size = 'medium')
HWP_1_Label = axarr[1].text(20, 875, HWP_1_str,
    color = 'black', backgroundcolor = 'white', size = 'medium')
HWP_2_Label = axarr[2].text(20, 875, HWP_2_str,
    color = 'black', backgroundcolor = 'white', size = 'medium')
HWP_3_Label = axarr[3].text(20, 875, HWP_3_str,
    color = 'black', backgroundcolor = 'white', size = 'medium')

# Connect the event manager...
cid1 = fig.canvas.mpl_connect('key_press_event', on_key)

# NOW show the image (without continuing execution)
# plt.ion()
plt.show()
# plt.ioff()
#
# pdb.set_trace()
# Disconnect the event manager and close the figure
fig.canvas.mpl_disconnect(cid1)

# Close the plot
plt.close()

print('Done!')
