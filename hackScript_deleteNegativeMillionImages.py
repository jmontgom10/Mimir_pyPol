# Quick cheat script to move files out of the S3 directory if they have skijumps
#

import os
import glob
import numpy as np
import astroimage as ai
ai.set_instrument('Mimir')

# Set the bkgFreeImagesDir
bkgFreeImagesDir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\bkgFreeHWPimages'

# Find the files in the bkgFreeImagesDir
fileList = glob.glob(os.path.join(bkgFreeImagesDir, '*.fits'))

# loop through the files
for file1 in fileList:
    # Read the file in
    img = ai.reduced.ReducedScience.read(file1)

    # Test if the majority of pixels are marked as bad
    if np.abs(img.header['HWP_ANG']) < 2:
        print('Removing file {}'.format(os.path.basename(file1)))
        os.remove(file1)

print('Done!')
