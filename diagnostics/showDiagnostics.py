# Short script to read in and display all the images in the diagnostic dir

import glob
import numpy as np
import time
from astropy.stats import sigma_clipped_stats, freedman_bin_width
from scipy import ndimage
import astroimage as ai
ai.set_instrument('Mimir')

fileList = glob.glob('*.fits')
fileList = np.array(fileList)
fileList.sort()

# Define a quick mode function
def mode(array):
    """An estimate of the statistical mode of this image"""
    # SUPER fast and sloppy mode estimate:
    mean, median, std = sigma_clipped_stats(array)
    quickModeEst = 3*median - 2*mean

    # Compute an approximately 3-sigma range about this
    modeRegion = quickModeEst + std*np.array([-1.5, +1.5])

    # Now compute the number of bins to generate in this range
    binWidth = freedman_bin_width(array.flatten())
    bins     = np.arange(modeRegion[0], modeRegion[1], binWidth)

    # Loop through larger and larger binning until find unique solution
    foundMode = False
    while not foundMode:
        # Generate a histogram of the flat field
        hist, flatBins = np.histogram(array.flatten(), bins=bins)

        # Locate the histogram maximum
        maxInds = (np.where(hist == np.max(hist)))[0]
        if maxInds.size == 1:
            # Grab the index of the maximum value and shrink
            maxInd = maxInds[0]
            foundMode = True
        else:
            # Shrink the NUMBER of bins to help find a unqiue maximum
            numBins *= 0.9
            bins     = np.linspace(modeRegion[0], modeRegion[1], numBins)

    # Estimate flatMode from histogram maximum
    flatMode = np.mean(flatBins[maxInd:maxInd+2])

    return flatMode

# Loop through and display the files
for file1 in fileList:
    img = ai.reduced.ReducedScience.read(file1)
    tmpMode = mode(img.data.flatten())
    tmpData = ndimage.median_filter(img.data, 3)
    img.data = tmpData
    img.show(vmin=0.9*tmpMode, vmax=1.1*tmpMode)
    img.image.figure.suptitle(file1)
    img.image.figure.colorbar(img.image)
    img.image.figure.canvas.draw()
    time.sleep(0.5)



import pdb; pdb.set_trace()
