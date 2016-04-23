# This script will construct one average background image per waveplate angle,
# per waveband, per night. This assumes that the supersky structure is constant
# throughout the night. This is a reasonable assumption for the actual sky
# background contribution, but  the telescope contribution may change. To check
# if the telescope contribution changes through th night, the frist and last
# off-target "B image" from each night will be displayed to the user for
# examination. along with a residual different image. If the residual imge shows
# significant sky structure, then it will be necessary to identify when during
# the night the telescope background changed.
#

import os
import sys
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from photutils import detect_sources, Background
from scipy.ndimage.filters import median_filter, gaussian_filter

import pdb

# Add the AstroImage class
sys.path.append("C:\\Users\\Jordan\\Libraries\\python\\AstroImage")
from AstroImage import AstroImage
import image_tools
import pyPol_tools

# This is the location of all PPOL reduction directory
PPOL_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_reduced'

# Build the path to the S3_Asotremtry files
S3dir = os.path.join(PPOL_dir, 'S3_Astrometry')

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_data'

# Build the path to the supersky directory
bkgImagesDir = os.path.join(pyPol_data, 'bkgImages')
if (not os.path.isdir(bkgImagesDir)):
    os.mkdir(bkgImagesDir, 0o755)

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

# Determine which parts of the fileIndex pertain to science images
useFiles = np.where(fileIndex['Use'] == 1)

# Cull the file index to only include files selected for use
fileIndex = fileIndex[useFiles]

# Group the fileIndex by...
# 1. Waveband
# 2. Night
# 3. Dither (pattern)
# 4. HWP Angle
# 5. ABBA value

# fileIndexByGroup = fileIndex.group_by(['Waveband', 'Night',
#     'Dither', 'HWP', 'ABBA'])
fileIndexByGroup = fileIndex.group_by(['PPOL Name', 'HWP', 'ABBA'])

# Loop through each grouping
for group in fileIndexByGroup.groups:
    # Check if we're dealing with the A or B positions
    thisABBA = str(np.unique(group['ABBA'].data)[0])

    # Skip over the A images
    if thisABBA == 'A': continue

    # Grab the current target information
    thisPPOLname = str(np.unique(group['PPOL Name'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    thisHWP      = str(np.unique(group['HWP'].data)[0])

    # Test if this target-waveband-HWPang combo was previously processed
    outFile = os.path.join(bkgImagesDir,
        (thisPPOLname + '_HWP' + thisHWP + '.fits'))

    if os.path.isfile(outFile):
        print('File ' + os.path.basename(outFile) +
            ' already exists... skipping to next group')
        continue

    numImgs = len(group)
    print('\nProcessing {0} images for'.format(numImgs))
    print('\tPPOL Name : {0}'.format(thisPPOLname))
    print('\tWaveband  : {0}'.format(thisWaveband))
    print('\tHWP       : {0}'.format(thisHWP))

    # Read in all the relevant images for constructing this HWP image
    imgList = [AstroImage(file1) for file1 in group['Filename'].data]

    # # Loop through the images and check if there is a gradient to subtract.
    # imgList1  = imgList.copy()
    # flatCount = 0
    # for imgNum, img in enumerate(imgList):
    #     # Look for test files....
    #
    #     # Type 1
    #     # if os.path.basename(img.filename) != '20150205.550_LDFC.fits': continue
    #     # if os.path.basename(img.filename) != '20150205.551_LDFC.fits': continue
    #
    #     # Type 2
    #     # if os.path.basename(img.filename) != '20150212.235_LDFC.fits': continue
    #     # if os.path.basename(img.filename) != '20150206.599_LDFC.fits': continue
    #
    #     # Perform the gradient test
    #     gradientType = pyPol_tools.test_for_gradients(img, sig_thresh = 5.0)
    #
    #     # If there is a gradient, then subtract it.
    #     if gradientType != 0:
    #         flatCount += 1
    #         print('Flattening image {0}'.format(os.path.basename(img.filename)))
    #         imgList1[imgNum] = pyPol_tools.flatten_gradients(img,
    #             gradientType = gradientType)
    #         pdb.set_trace()
    #
    # # Cleanup temporary variable
    # imgList = imgList1.copy()
    # del imgList1
    #
    # if flatCount == 0: continue

    # Construct a basic supersky image
    thisSupersky = pyPol_tools.build_supersky(imgList)

    # The goal of the script is complete. Write the image to disk.
    thisSupersky.write(outFile, dtype=np.float32)

print('Done!')
