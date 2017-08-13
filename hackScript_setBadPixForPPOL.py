# hack script to set the bad pixels to the right value for PPOL
import os
import glob
import numpy as np
import astroimage as ai
ai.set_instrument('Mimir')

bkgFreeDir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\bkgFreeHWPimages'
fileList = glob.glob(
    os.path.join(bkgFreeDir, '*.fits')
)

for file1 in fileList:
    print('processing {}'.format(os.path.basename(file1)))
    img = ai.reduced.ReducedScience.read(file1)

    # Capture NaNs and bad values and set them to -1e6 so that PPOL will
    # know what to do with those values.
    tmpData = img.data
    badPix = np.logical_not(np.isfinite(tmpData))
    tmpData[np.where(badPix)] = -1e6
    badPix = np.abs(tmpData) > 1e5
    tmpData[np.where(badPix)] = -1e6

    # Store the data in the image object
    img.data = tmpData

    # Write back to disk
    img.write(file1, dtype=np.float32, clobber=True)

print('Done!')
