# -*- coding: utf-8 -*-
"""
Estimate the PSF FWHM for each of the reduced science images, and append the
'reducedFileIndex.csv' with a column containing that information. The PSF FWHM
values will be used to cull data to only include good seeing conditions.
"""

#Import whatever modules will be used
import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, Column
from astropy.coordinates import SkyCoord
import astropy.units as u
from scipy import stats

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

# Set the filename for the reduced data indexFile and read it in
reducedFileIndexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
reducedFileIndex     = Table.read(reducedFileIndexFile)

# Set the filename for the PSF backup index file. This file will be saved
# separately, in addition to the modified reduced file index. This way, if the
# user invokes 01_buildIndex.py again, it will not OVERWRITE the PSF data.
PSFindexFile = os.path.join(pyPol_data, 'PSFindex.csv')

# Finally, loop through EACH image and compute the PSF
PSFwidths = []
sigm2FWHM = 2*np.sqrt(2*np.log(2))
numberOfFiles = len(reducedFileIndex)
for iFile, filename in enumerate(reducedFileIndex['FILENAME'].data):
    if reducedFileIndex['AB'][iFile] == 'B':
        PSFwidths = [-1]
        continue

    # Construct the PPOL file name
    PPOL_file = os.path.join(S3_dir, filename)

    # Read in the image
    thisImg = ai.reduced.ReducedScience.read(PPOL_file)

    # Construct a Photometry analyzer object
    thisPhotAnalyzer = ai.utilitywrappers.PhotometryAnalyzer(thisImg)

    # Estimate the PSF for this image
    PSFstamp, PSFparams = thisPhotAnalyzer.get_psf()

    # Check if non-null values were returned from the get_psf method
    if (PSFparams['sminor'] is None) or (PSFparams['smajor'] is None):
        PSFwidths.append(0)
        continue

    # Estimate a binning-normalized "width" parameter for the PSF
    thisBinning = np.sqrt(np.array(thisImg.binning).prod())
    thisWidth   = np.sqrt(PSFparams['sminor']*PSFparams['smajor'])*thisBinning
    PSFwidths.append(sigm2FWHM*thisWidth)

    # Compute the percentage done and show it to the user
    percentage = np.round(100*iFile/numberOfFiles, 2)
    print('File : {0} ... completed {1:3.2f}%'.format(os.path.basename(filename), percentage), end="\r")

# Add a FWHM column to the PSF index (for safe(r) keeping) file index.
PSFindex   = Table()
FWHMcolumn = Column(name='FWHM', data=PSFwidths)
PSFindex.add_column(FWHMcolumn)
reducedFileIndex.add_column(FWHMcolumn)

# Save the index to disk.
PSFindex.write(PSFindexFile, format='ascii.csv', overwrite=True)
reducedFileIndex.write(reducedFileIndexFile, format='ascii.csv', overwrite=True)
