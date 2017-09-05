# Allows the user to view the constructed HWP bacxground images. Shows all four
# HWP rotations associated with a single IPPA angle
#

import os
import sys
import glob
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from photutils import detect_threshold, detect_sources
from scipy.ndimage.filters import median_filter, gaussian_filter
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval

# Add the AstroImage class
import astroimage as ai

# Add the header handler to the BaseImage class
from Mimir_header_handler import Mimir_header_handler
ai.reduced.ReducedScience.set_header_handler(Mimir_header_handler)
ai.set_instrument('mimir')

# This is the location where all pyPol data will be saved
pyPol_data     = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\'
polarimetryDir = os.path.join(pyPol_data, 'Polarimetry')
stokesDir      = os.path.join(polarimetryDir, 'stokesImages')

################################################################################
# Determine which parts of the fileIndex pertain to science images
fileList = glob.glob(os.path.join(stokesDir, '*I.fits'))
imgList  = [ai.reduced.ReducedScience.read(f) for f in fileList]
imgList  = np.array(imgList)

# Loop through each image and construct a list of pixel positions
xxList = []
yyList = []
for img in imgList:
    ny, nx = img.shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    xxList.append(xx)
    yyList.append(yy)

#******************************************************************************
# Define the event handlers for clicking and keying on the image display
#******************************************************************************
def on_click(event):
    global xxList, yyList, imgList, imgNum
    global fig, brushSize, ax, maskImg, axImg

    # Grab the position of the click
    x, y = event.xdata, event.ydata

    # Rtreieve the image pixel positions
    yy, xx = yyList[imgNum], xxList[imgNum]

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
    ax.collections = []
    ax.contour(xx, yy, maskImg.data, levels=[0.5], colors='white', alpha = 0.2)

    # Update the display
    fig.canvas.draw()

################################################################################
# Define a function to handle what sholud be done whenever a key is pressed
################################################################################
def on_key(event):
    global xxList, yyList, imgList, imgNum
    global fig, brushSize, maskImg
    global stokesDir

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

            # If there are no more images, then loop back to begin of list
            if imgNum > imgList.size - 1:
                imgNum   = 0

        if event.key == 'left':
            #Move back to the previous image
            imgNum -= 1

            # If there are no more images, then loop back to begin of list
            if imgNum < 0:
                imgNum   = imgList.size - 1

        # Build the image scaling intervals
        img              = imgList[imgNum]
        zScaleGetter     = ZScaleInterval()
        thisMin, thisMax = zScaleGetter.get_limits(img.data)
        thisMax         *= 10

        #*******************************
        # Update the displayed mask
        #*******************************

        # Check which mask files might be usable...
        baseFile = os.path.basename(img.filename).split('_I')[0]
        maskFile = os.path.join(stokesDir,
            baseFile + '_mask.fits')
        if os.path.isfile(maskFile):
            # If the mask for this file exists, use it
            print('using this mask: ',os.path.basename(maskFile))
            maskImg = ai.reduced.ReducedScience.read(maskFile)
        else:
            # If none of those files exist, build a blank slate
            # Build a mask template (0 = not masked, 1 = masked)
            maskImg = ai.reduced.ReducedScience(
                (img.data*0).astype(np.int16),
                header =  img.header
            )
            maskImg.filename = maskFile

        # Grab the pixel positons
        yy, xx, = yyList[imgNum], xxList[imgNum]

        # Update contour plot (clear old lines redo contouring)
        ax.collections = []
        ax.contour(xx, yy, maskImg.data, levels=[0.5], colors='white', alpha = 0.2)

        # Reassign image display limits
        axImg.set_clim(vmin = thisMin, vmax = thisMax)

        # Display the new images and update extent
        axImg.set_data(img.data)
        axImg.set_extent((xx.min(), xx.max(), yy.min(), yy.max()))

        # Update the annotation
        ax.set_title(os.path.basename(img.filename))

        # Update the display
        fig.canvas.draw()

    # Save the generated mask
    if event.key == 'enter':
        # Write the mask to disk
        print('Writing mask for file {}'.format(maskImg.filename))
        maskImg.write(clobber=True)

    # Clear out the mask values
    if event.key == 'backspace':
        try:
            # Clear out the mask array
            maskImg.data = (maskImg.data*0).astype(np.int16)

            # Update contour plot (clear old lines redo contouring)
            ax.collections = []
            ax.contour(xx, yy, maskImg.data, levels=[0.5], colors='white', alpha = 0.2)

            # Update the display
            fig.canvas.draw()
        except:
            pass

#******************************************************************************
# This script will run the mask building step of the pyPol reduction
#******************************************************************************
fig = plt.figure(figsize=(10,9))

# Create the first axis and make the x-axis labels invisible
ax = plt.subplot(111)
plt.setp(ax.get_xticklabels(), fontsize = 12)
plt.setp(ax.get_yticklabels(), fontsize = 12)

# Rescale the figure and setup the spacing between images
plt.subplots_adjust(left = 0.04, bottom = 0.04, right = 0.95, top = 0.96,
    wspace = 0.02, hspace = 0.02)

# Initalize the image number and brush size
imgNum = 0
brushSize = 3

# Start by grabbing the corresponding group names and IPPAs for those indices
img     = imgList[imgNum]

# Build (or read) an initial mask
baseFile = os.path.basename(img.filename).split('_I')[0]
maskFile = os.path.join(stokesDir,
    baseFile + '_mask.fits')
if os.path.isfile(maskFile):
    # If the mask for this file exists, use it
    print('using this mask: ',os.path.basename(maskFile))
    maskImg = ai.reduced.ReducedScience.read(maskFile)
else:
    # If none of those files exist, build a blank slate
    # Build a mask template (0 = not masked, 1 = masked)
    maskImg          = ai.reduced.ReducedScience(
        (img.data*0).astype(np.int16),
        header =  img.header
    )
    maskImg.filename = maskFile

# Populate each axis with its image
axImg = img.show(axes = ax, cmap='viridis', noShow = True)
ax.set_title(os.path.basename(img.filename))


# Connect the event manager...
cid1 = fig.canvas.mpl_connect('button_press_event',on_click)
cid2 = fig.canvas.mpl_connect('key_press_event', on_key)

# NOW show the image (without continuing execution)
# plt.ion()
plt.show()
# plt.ioff()
#
# import pdb; pdb.set_trace()
# Disconnect the event manager and close the figure
fig.canvas.mpl_disconnect(cid1)
fig.canvas.mpl_disconnect(cid2)

# Close the plot
plt.close()

print('Done!')
