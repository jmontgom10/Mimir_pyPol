# PPOL step 5 requires at least 1x HWP 0 image. However, some groups do not have
# any successful HWP 0 image. Thus, to "fake" these images, copies of some other
# HWP with identicy IPPA value will be created and used in the place of the lost
# HWP images.

import os
import sys
import glob
import warnings
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from astropy.modeling import models, fitting
from photutils import detect_threshold, detect_sources
from scipy.ndimage.filters import median_filter, gaussian_filter
from photutils import Background2D

# Add the AstroImage class
import astroimage as ai
ai.set_instrument('Mimir')

# Add the header handler to the BaseImage class
from Mimir_header_handler import Mimir_header_handler

# This is the location of the *Raw* images for this observing run. These images
# will be used to retrieve information such as the observing date-time value.
# *** NOTE: This is a network drive, so you need to be connected to that drive
# *** to be able to run this script.
RAW_dir = 'Q:\\Mimir_Data\\Raw_Data\\201611_Jordan_Raw_Data'

# This is the BDP directory
BDP_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\BDP_Data\\201611\\12_Science_Images'

# This is the location of all PPOL reduction directory
PPOL_dir = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\PPOL_reduced\\201611\\notPreFlattened'

# Build the path to the S3_Asotrometry files
S3_dir = os.path.join(PPOL_dir, 'S3_Astrometry')

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\'

# This is the location of the (pre-computed) star masks for the B-images
maskDir = os.path.join(pyPol_data, 'Masks')
starMaskDir = os.path.join(maskDir, 'starMasks')

# Build the path to the supersky directory
bkgFreeImagesDir = os.path.join(pyPol_data, 'bkgFreeHWPimages')
if (not os.path.isdir(bkgFreeImagesDir)):
    os.mkdir(bkgFreeImagesDir, 0o755)

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')

# Get the HWPs fromm the flat list
HWPstepList    = np.array([
    0, 33, 67, 100,
    133, 167, 200, 233,
    267, 300, 333, 367,
    400, 433, 467, 500
])
HWPlist         = np.arange(16, dtype=int) + 1
IPPAlist        = np.array(4*[0, 45, 90, 135])
HWPstep_to_HWP  = dict(zip(HWPstepList, HWPlist))
HWP_to_IPPA     = dict(zip(HWPlist, IPPAlist))
IPPA0_HWPs      = 4*np.arange(4) + 1

# fileIndexByGroup = fileIndex.group_by(['FILTER', 'Night',
#     'Dither', 'HWP', 'ABBA'])
fileIndexByGroup = fileIndex.group_by(['GROUP_ID'])

# Loop through each grouping
for group in fileIndexByGroup.groups:
    # Grab the current target information
    thisGroupName = str(np.unique(group['OBJECT'].data)[0])
    thisTarget    = str(np.unique(group['TARGET'].data)[0])
    thisGroupID   = str(np.unique(group['GROUP_ID'].data)[0])
    thisFilter    = str(np.unique(group['FILTER'].data)[0])

    # if thisTarget != 'M78' or thisFilter != 'Ks': continue
    # if thisGroupName != 'M78_K3': continue

    print('\nProcessing images for')
    print('\tOBJECT : {0}'.format(thisGroupName))
    print('\tFILTER : {0}'.format(thisFilter))

    # Search for HWP 0 files
    HWP0_and_A = np.logical_and(
        group['HWP'] == 1,
        group['AB'] == 'A'
    )

    # Find the indices of the expected HWP 0 and on-target frames
    HWP0_and_Ainds  = np.where(HWP0_and_A)[0]
    HWP0_and_Afiles = group['FILENAME'][HWP0_and_Ainds]

    HWP0_and_AfilesExist = [
        os.path.isfile(
            os.path.join(bkgFreeImagesDir, f)
        ) for f in HWP0_and_Afiles
    ]

    # Check if at least one HWP 0 file was found
    if (np.sum(HWP0_and_AfilesExist) > 0):
        print('\tHWP 0 file exists... continuing')
        continue

    # Construct the filename of the expected HWP 0 file
    try:
        HWP0file = os.path.join(
            bkgFreeImagesDir,
            HWP0_and_Afiles[HWP0_and_Ainds.argmin()]
        )
    except:
        HWP0file = None

    # No HWP 0 images were found, so try other IPPAs
    for pseudoHWP0 in IPPA0_HWPs[1:]:
        # For the given pseudoHWP0 value, learch for viable files
        pseudoHWP0_and_A = np.logical_and(
            group['HWP'] == pseudoHWP0,
            group['AB'] == 'A'
        )
        pseudoHWP0_and_Ainds  = np.where(pseudoHWP0_and_A)[0]
        pseudoHWP0_and_Afiles = group['FILENAME'][pseudoHWP0_and_Ainds]

        pseudoHWP0_and_AfilesExist = [
            os.path.isfile(
                os.path.join(bkgFreeImagesDir, f)
            ) for f in pseudoHWP0_and_Afiles
        ]

        # Check if at least one pseudoHWP0 file was found
        if np.sum(pseudoHWP0_and_AfilesExist) > 0:
            # You found a pseudo HWP 0 file!!!
            print('\tUsing HWP {} as an HWP 0 substitute'.format(pseudoHWP0))

            # Locate the indices of the existing S3 images
            pseudoHWP0_and_Ainds = np.where(pseudoHWP0_and_AfilesExist)[0]

            # Grab this file path
            pseudoHWP0basename = pseudoHWP0_and_Afiles[pseudoHWP0_and_Ainds.argmin()]
            pseudoHWP0file = os.path.join(
                bkgFreeImagesDir,
                pseudoHWP0basename
            )

            # Construct the path to the expected HWP 0 image
            HWP0_night        = pseudoHWP0basename.split('.')[0]
            pseudoHWP0_number = int(pseudoHWP0basename.split('.')[1].split('_')[0])
            HWP0_number       = pseudoHWP0_number - 4*(pseudoHWP0 - 1)
            HWP0basename = '{}.{}_LDFC.fits'.format(
                HWP0_night, HWP0_number
            )
            HWP0file = os.path.join(
                bkgFreeImagesDir,
                HWP0basename
            )

            # Construct the path to the corresponding RAW image
            HWP0rawBasename = HWP0basename.split('_')[0] + '.fits'
            HWP0rawFile = os.path.join(
                os.path.join(
                    os.path.join(
                        RAW_dir,
                        HWP0_night
                    ), 'CDS_RAW'
                ), HWP0rawBasename
            )

            # Read in the substitute file
            tmpPPOLimg = ai.reduced.ReducedScience.read(pseudoHWP0file)
            tmpRAWimg  = ai.reduced.ReducedScience.read(HWP0rawFile)

            # # Modify the header to have HWP 0 and file number correct
            # # I think there are probably some other things I need to take care of...
            tmpPPOLhead = tmpPPOLimg.header.copy()
            tmpRAWhead  = tmpRAWimg.header.copy()
            tmpPPOLhead.update(tmpRAWhead)
            tmpPPOLhead['BUNIT']  = tmpPPOLhead['BUNIT']
            tmpPPOLhead['OBJECT'] = tmpPPOLimg.header['OBJECT']
            tmpPPOLhead['MB_HWP'] = 0.0

            # Create a ReducedScience instance to store data and header
            outImg = ai.reduced.ReducedScience(
                tmpPPOLimg.data.astype(np.float32),
                header=tmpPPOLhead
            )

            # Double check that astrometry is not lost
            outImg.astrometry_to_header(tmpPPOLimg.wcs)

            # Finally, construct the BDP path for this file
            HWP0BDPfile = os.path.join(
                os.path.join(
                    os.path.join(
                        BDP_dir,
                        HWP0_night
                    ), 'polarimetry'
                ), HWP0basename
            )

            # Write the image to disk
            try:
                print('Writing file {0}'.format(os.path.basename(HWP0file)))
                outImg.write(HWP0file, clobber=True)
                if not os.path.isfile(HWP0BDPfile):
                    outImg.write(HWP0BDPfile)
            except:
                print('Failed to write file {0}'.format(os.path.basename(HWP0file)))
                import pdb; pdb.set_trace()
            break

    else:
        print('\tNo suitable HWP 0 substitute was found for this group')

print('Done!')
