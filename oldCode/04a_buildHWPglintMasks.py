# -*- coding: utf-8 -*-
"""
Launches a GUI mask builder for each file associated with  the targets specidied
by the 'targets' variable.

When the script is launched, a three pane plot of the science images will be
displayed. The center pane in this triptych is the active pane for which you are
build a mask. The following look-up table indicates which button clicks and keys
presses are associated with a given action in the GUI.

--------------------------------------------------------------------------------
Event	    | Effect
--------------------------------------------------------------------------------
Left Click	| Apply a circular aperture mask to the clicked region

Right Click	| Delete a circular aperture mask from the clicked region

1 - 6	    | Set the size of the circular aperture

Enter	    | Save the current mask for the center pane to disk

Backspace	| Reset the current mask to a blank slate

Left/Right	| Change the active pane to the previous/next image
--------------------------------------------------------------------------------

To build a mask, simply click on the regions of the active pane which need to be
masked. The mask will be displayed in the center pane as a semi-opaque white
outline. You can delete regions of the mask with right clicks. Once you are
satisfied with the mask you've build, you can save it to disk with a single
stroke of the Enter key. Press the left or right arrow keys to scroll through
the images for the specified target(s), and press the Backspace key to clear the
current mask to be completely blank. Simply close the GUI window plot to end the
script.
"""

# Imports
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from astropy.table import Table, Column
from astropy.visualization import ZScaleInterval

# TODO: build a "MaskBuilder" class to manage all these variables and actions.
# Define the mask directory as a global variable
global maskDir

# Add the AstroImage class
import astroimage as ai

# Add the header handler to the BaseImage class
from Mimir_header_handler import Mimir_header_handler
ai.reduced.ReducedScience.set_header_handler(Mimir_header_handler)
ai.set_instrument('mimir')

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================
# Define the location of the PPOL reduced data to be read and worked on
PPOL_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_Reduced\\201611\\'
S3_dir    = os.path.join(PPOL_data, 'S3_Astrometry')

# This is the location where all pyPol data will be saved
pyPol_data='C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611'

# The user can speed up the process by defining the "Target" values from
# the fileIndex to be considered for masking.
# Masks can onlybe produced for targets in this list.
targets = ['NGC7023']

# Setup new directory for masks
maskDir = os.path.join(pyPol_data, 'Masks')
maskDir = os.path.join(maskDir, 'artifactMasks')
if (not os.path.isdir(maskDir)):
    os.mkdir(maskDir, 0o755)

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='ascii.csv')

# Determine which parts of the fileIndex pertain to on-target science images
useFiles = np.logical_and(
    fileIndex['USE'] == 1,
    fileIndex['DITHER_TYPE'] == 'ABBA'
)
useFiles = np.logical_and(
    useFiles,
    fileIndex['AB'] == 'A'
)

# Further restrict the selection to only include the selected targets
targetFiles = np.zeros((len(fileIndex),), dtype=bool)
for target in targets:
    targetFiles = np.logical_or(
        targetFiles,
        fileIndex['TARGET'] == target
    )

# Cull the fileIndex to ONLY include the specified targets
goodTargetRows = np.logical_and(useFiles, targetFiles)
targetRowInds  = np.where(goodTargetRows)
fileIndex      = fileIndex[targetRowInds]

#******************************************************************************
# Define the event handlers for clicking and keying on the image display
#******************************************************************************

def on_click(event):
    global xList, yList, xx, yy
    global fig, brushSize, axarr, maskImg, thisAxImg

    x, y = event.xdata, event.ydata
#    xList.append(x)
#    yList.append(y)

    # Compute distances from the click and update mask array
    dist     = np.sqrt((xx - x)**2 + (yy - y)**2)
    maskInds = np.where(dist < brushSize*5)
    if event.button == 1:
        tmpData = maskImg.data
        tmpData[maskInds] = 1
        maskImg.data = tmpData
    if (event.button == 2) or (event.button == 3):
        tmpData = maskImg.data
        tmpData[maskInds] = 0
        maskImg.data = tmpData

    # Update contour plot (clear old lines redo contouring)
    axarr[1].collections = []
    axarr[1].contour(xx, yy, maskImg.data, levels=[0.5], colors='white', alpha = 0.2)

    # Update the display
    fig.canvas.draw()

def on_key(event):
    global fileList, targetList, fig, imgNum, brushSize
    global maskDir, maskImg
    global prevImg,   thisImg,   nextImg
    global prevAxImg, thisAxImg, nextAxImg
    global prevTarget, thisTarget, nextTarget
    global prevMin,   thisMin,   nextMin
    global prevMax,   thisMax,   nextMax
    global prevLabel, thisLabel, nextLabel

    # Handle brush sizing
    if event.key == '1':
        brushSize = 1
    elif event.key == '2':
        brushSize = 2
    elif event.key == '3':
        brushSize = 3
    elif event.key == '4':
        brushSize = 4
    elif event.key == '5':
        brushSize = 5
    elif event.key == '6':
        brushSize = 6

    # Increment the image number
    if event.key == 'right' or event.key == 'left':
        if event.key == 'right':
            #Advance to the next image
            imgNum += 1

            # Read in the new files
            prevImg = thisImg
            thisImg = nextImg
            nextImg = ai.reduced.ReducedScience.read(fileList[(imgNum + 1) % len(fileList)])

            # Update target info
            prevTarget = thisTarget
            thisTarget = nextTarget
            nextTarget = targetList[(imgNum + 1) % len(fileList)]

            # Build the image scaling intervals
            zScaleGetter = ZScaleInterval()

            # Compute new image display minima
            prevMin = thisMin
            thisMin = nextMin
            nextMin, _ = zScaleGetter.get_limits(nextImg.data)

            # Compute new image display maxima
            prevMax = thisMax
            thisMax = nextMax
            _, nextMax = zScaleGetter.get_limits(nextImg.data)

        if event.key == 'left':
            #Move back to the previous image
            imgNum -= 1

            # Read in the new files
            nextImg = thisImg
            thisImg = prevImg
            prevImg = ai.reduced.ReducedScience.read(fileList[(imgNum - 1) % len(fileList)])

            # Update target info
            nextTarget = thisTarget
            thisTarget = prevTarget
            prevTarget = targetList[(imgNum - 1) % len(fileList)]

            # Build the image scaling intervals
            zScaleGetter = ZScaleInterval()

            # Compute new image display minima
            nextMin = thisMin
            thisMin = prevMin
            prevMin, _ = zScaleGetter.get_limits(prevImg.data)

            # Compute new image display maxima
            nextMax = thisMax
            thisMax = prevMax
            _, prevMax = zScaleGetter.get_limits(prevImg.data)

        #*******************************
        # Update the displayed mask
        #*******************************

        # Check which mask files might be usable...
        prevMaskFile = os.path.join(maskDir,
            os.path.basename(prevImg.filename))
        thisMaskFile = os.path.join(maskDir,
            os.path.basename(thisImg.filename))
        nextMaskFile = os.path.join(maskDir,
            os.path.basename(nextImg.filename))
        if os.path.isfile(thisMaskFile):
            # If the mask for this file exists, use it
            print('using this mask: ',os.path.basename(thisMaskFile))
            maskImg = ai.reduced.ReducedScience.read(thisMaskFile)
        elif os.path.isfile(prevMaskFile) and (prevTarget == thisTarget):
            # Otherwise check for the mask for the previous file
            print('using previous mask: ',os.path.basename(prevMaskFile))
            maskImg = ai.reduced.ReducedScience.read(prevMaskFile)
        elif os.path.isfile(nextMaskFile) and (nextTarget == thisTarget):
            # Then check for the mask of the next file
            print('using next mask: ',os.path.basename(nextMaskFile))
            maskImg = ai.reduced.ReducedScience.read(nextMaskFile)
        else:
            # If none of those files exist, build a blank slate
            # Build a mask template (0 = not masked, 1 = masked)
            maskImg       = thisImg.copy()
            maskImg.filename = thisMaskFile
            maskImg = maskImg.astype(np.int16)
            # Make sure the uncertainty array is removed from the image
            try:
                del maskImg.uncertainty
            except:
                pass

        # Update contour plot (clear old lines redo contouring)
        axarr[1].collections = []
        axarr[1].contour(xx, yy, maskImg.data, levels=[0.5], colors='white', alpha = 0.2)

        # Reassign image display limits
        prevAxImg.set_clim(vmin = prevMin, vmax = prevMax)
        thisAxImg.set_clim(vmin = thisMin, vmax = thisMax)
        nextAxImg.set_clim(vmin = nextMin, vmax = nextMax)

        # Display the new images
        prevAxImg.set_data(prevImg.data)
        thisAxImg.set_data(thisImg.data)
        nextAxImg.set_data(nextImg.data)

        # Update the annotation
        axList = fig.get_axes()
        axList[1].set_title(os.path.basename(thisImg.filename))

        prevStr   = (str(prevImg.header['OBJECT']) + '\n' +
                     str(prevImg.header['FILTNME2'] + '\n' +
                     str(prevImg.header['HWP'])))
        thisStr   = (str(thisImg.header['OBJECT']) + '\n' +
                     str(thisImg.header['FILTNME2'] + '\n' +
                     str(thisImg.header['HWP'])))
        nextStr   = (str(nextImg.header['OBJECT']) + '\n' +
                     str(nextImg.header['FILTNME2'] + '\n' +
                     str(nextImg.header['HWP'])))
        prevLabel.set_text(prevStr)
        thisLabel.set_text(thisStr)
        nextLabel.set_text(nextStr)

        # Update the display
        fig.canvas.draw()

    # Save the generated mask
    if event.key == 'enter':
        # Make sure the header has the right values
        maskImg.header = thisImg.header

        # TODO: make sure the mask ONLY has what it needs
        # i.e., remove uncertainty and convert to np.ubyte type.

        # Write the mask to disk
        maskBasename = os.path.basename(thisImg.filename)
        maskFullname = os.path.join(maskDir, maskBasename)
        print('Writing mask for file {}'.format(maskBasename))
        maskImg.write(maskFullname, clobber=True)

    # Clear out the mask values
    if event.key == 'backspace':
        # Clear out the mask array
        maskImg.data = maskImg.data * np.byte(0)

        # Update contour plot (clear old lines redo contouring)
        axarr[1].collections = []
        axarr[1].contour(xx, yy, maskImg.data, levels=[0.5], colors='white', alpha = 0.2)

        # Update the display
        fig.canvas.draw()

#******************************************************************************

#******************************************************************************
# This is the main script that will load in file names and prepare for plotting
#******************************************************************************

# Declare global variables
#global xList, yList
global xx, yy
global fileList, targetList, fig, imgNum, maskImg
global prevImg,   thisImg,   nextImg
global prevTarget, thisTarget, nextTarget
global prevAxImg, thisAxImg, nextAxImg
global prevMin,   thisMin,   nextMin
global prevMax,   thisMax,   nextMax
global prevLabel, thisLabel, nextLabel
xList     = []
yList     = []
imgNum    = 0      # This number will be the FIRST image to be displayed center...
brushSize = 3      # (5xbrushSize pix) is the size of the region masked

#******************************************************************************
# This script will run the mask building step of the pyPol reduction
#******************************************************************************

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
# 4. Polaroid Angle
fileIndexByTarget = fileIndex.group_by(
    ['TARGET', 'FILTER']
)

# Add the information to the fileList and targetList variables
fileList   = fileIndexByTarget['FILENAME'].data.tolist()
fileList   = [os.path.join(S3_dir, f) for f in fileList]
targetList = fileIndexByTarget['TARGET'].data.tolist()

#*************************************
# Now prepare to plot the first images
#*************************************

# Read in an image for masking
prevImg = ai.reduced.ReducedScience.read(fileList[imgNum - 1])
thisImg = ai.reduced.ReducedScience.read(fileList[imgNum])
nextImg = ai.reduced.ReducedScience.read(fileList[imgNum + 1])

# Log the targets of the curent panes
prevTarget = targetList[imgNum - 1]
thisTarget = targetList[imgNum]
nextTarget = targetList[imgNum + 1]

###
# For some reason the prevTarget, thisTarget, and nextTaret
# variables are not accessible from the event managers the way that
# prevImg, thisImg, and nextImg are.
# I have definitely declared them to be global variables...
# Perhaps they're getting treated as local variables
# because they are modified elsewhere???


# Test if a mask has already been generated for this images
maskFile = os.path.join(maskDir, os.path.basename(thisImg.filename))
if os.path.isfile(maskFile):
    # If the mask file exists, use it
    maskImg = ai.reduced.ReducedScience.read(maskFile)
else:
    # If the mask file does not exist, build a blank slate
    # Build a mask template (0 = not masked, 1 = masked)
    maskImg       = thisImg.copy()
    maskImg.filename = maskFile
    maskImg = maskImg.astype(np.int16)

# Generate 2D X and Y position maps
maskShape = maskImg.shape
grids     = np.mgrid[0:maskShape[0], 0:maskShape[1]]
xx        = grids[1]
yy        = grids[0]

# Build the image displays
# Start by preparing a 1x3 plotting area
fig, axarr = plt.subplots(1, 3, sharey=True)

# Build the image scaling intervals
zScaleGetter = ZScaleInterval()

# Compute image count scaling
prevMin, prevMax = zScaleGetter.get_limits(prevImg.data)
thisMin, thisMax = zScaleGetter.get_limits(thisImg.data)
nextMin, nextMax = zScaleGetter.get_limits(nextImg.data)

# prevMin = np.median(prevImg.data) - 0.25*np.std(prevImg.data)
# prevMax = np.median(prevImg.data) + 2*np.std(prevImg.data)
# thisMin = np.median(thisImg.data) - 0.25*np.std(thisImg.data)
# thisMax = np.median(thisImg.data) + 2*np.std(thisImg.data)
# nextMin = np.median(nextImg.data) - 0.25*np.std(nextImg.data)
# nextMax = np.median(nextImg.data) + 2*np.std(nextImg.data)

# Populate each axis with its image
prevAxImg = prevImg.show(axes = axarr[0], cmap='viridis',
                                        vmin = prevMin, vmax = prevMax, noShow = True)
thisAxImg = thisImg.show(axes = axarr[1], cmap='viridis',
                                        vmin = thisMin, vmax = thisMax, noShow = True)
nextAxImg = nextImg.show(axes = axarr[2], cmap='viridis',
                                        vmin = nextMin, vmax = nextMax, noShow = True)

# Add a contour of the mask array
maskContour = axarr[1].contour(xx, yy, maskImg.data,
                               levels=[0.5], origin='lower', colors='white', alpha = 0.2)

# Rescale the figure and setup the spacing between images
#fig.set_figheight(5.575, forward=True)
#fig.set_figwidth(17.0, forward=True)
fig.set_size_inches(17, 5.675, forward=True)
plt.subplots_adjust(left = 0.04, bottom = 0.04, right = 0.98, top = 0.96,
                    wspace = 0.02, hspace = 0.02)

# Add some figure annotation
thisTitle = axarr[1].set_title(os.path.basename(thisImg.filename))
prevStr   = (str(prevImg.header['OBJECT']) + '\n' +
             str(prevImg.header['FILTNME2'] + '\n' +
             str(prevImg.header['HWP'])))
thisStr   = (str(thisImg.header['OBJECT']) + '\n' +
             str(thisImg.header['FILTNME2'] + '\n' +
             str(thisImg.header['HWP'])))
nextStr   = (str(nextImg.header['OBJECT']) + '\n' +
             str(nextImg.header['FILTNME2'] + '\n' +
             str(nextImg.header['HWP'])))
prevLabel = axarr[0].text(20, 875, prevStr,
                          color = 'white', size = 'medium')
thisLabel = axarr[1].text(20, 875, thisStr,
                          color = 'white', size = 'medium')
nextLabel = axarr[2].text(20, 875, nextStr,
                          color = 'white', size = 'medium')
thisShape = thisImg.shape
redLines  = axarr[1].plot([thisShape[0]/2, thisShape[0]/2], [0, thisShape[1]],
                          '-r',
                          [0, thisShape[0]], [thisShape[1]/2, thisShape[1]/2],
                          '-r', alpha = 0.4)
#********************************************
#log this for future use!
#********************************************
# A more standard way to handle mouse clicks?
#xyList = fig.ginput(n=-1, timeout=-30, show_clicks=True,
#                    mouse_add=1, mouse_pop=3, mouse_stop=2)
#********************************************

# Connect the event manager...
cid1 = fig.canvas.mpl_connect('button_press_event', on_click)
cid2 = fig.canvas.mpl_connect('key_press_event', on_key)

# NOW show the image (without continuing execution)
# plt.ion()
plt.show()
# plt.ioff()
#
# pdb.set_trace()
# Disconnect the event manager and close the figure
fig.canvas.mpl_disconnect(cid1)
fig.canvas.mpl_disconnect(cid2)

# Close the plot
plt.close()
