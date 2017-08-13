# A script to ensure that every HWP flat-field is normalized by its full fram MFM
import os
import glob
from astropy.stats import sigma_clipped_stats
import astroimage as ai

# Add the header handler to the BaseImage class
# from Mimir_header_handler import Mimir_header_handler
# ai.reduced.ReducedScience.set_header_handler(Mimir_header_handler)
ai.set_instrument('mimir')


# This is the location of all PPOL reduction directory
PPOL_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_reduced\\201611'

# Build the path to the S3_Asotrometry files
S3_dir = os.path.join(PPOL_dir, 'S3_Astrometry')

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\'

# Build the path to the supersky directory
bkgImagesDir = os.path.join(pyPol_data, 'bkgImages')

# Find all the HWP background images
bkgImgFileList = glob.glob(os.path.join(bkgImagesDir, '*.fits'))

# Loop through all the files and renormalize
numberOfFiles = len(bkgImgFileList)
for iFile, bkgImgFile in enumerate(bkgImgFileList):
    # Read in the file
    tmpImg = ai.reduced.ReducedScience.read(bkgImgFile)

    # Force normalization by the median
    _, median, _ = sigma_clipped_stats(tmpImg.data)
    tmpImg = tmpImg / (median*tmpImg.unit)

    # Resave file
    tmpImg.write(clobber=True)
    print('{0:3.1%} complete'.format(iFile/numberOfFiles), end='\r')

print('100% complete', end='\n\n')

print('Done!')
