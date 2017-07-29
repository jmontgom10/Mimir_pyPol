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

# Set the directory for the pyPol reduced data
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201612\\'

# Set the filename for the reduced data indexFile and read it in
reducedFileIndexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
reducedFileIndex     = Table.read(reducedFileIndexFile)

# Set the filename for the PSF backup index file. This file will be saved
# separately, in addition to the modified reduced file index. This way, if the
# user invokes 01_buildIndex.py again, it will not OVERWRITE the PSF data.
PSFindexFile = os.path.join(pyPol_data, 'PSFindex.csv')

#==============================================================================
# *********************** CUSTOM USER CODE ************************************
# this is where the user specifies where the raw data is stored
# and some of the subdirectory structure to find the actual .FITS images
#==============================================================================
# This is the location of all pyBDP data (index, calibration images, reduced...)
pyBDP_data='C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyBDP_data\\201612'

# This is the location where all pyPol data will be saved
pyPol_data='C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyPol_data\\201612'

# This is the location of the pyBDP processed Data
pyBDP_reducedDir = os.path.join(pyBDP_data, 'pyBDP_reduced_images')

# Finally, loop through EACH image and compute the PSF
PSFwidths = []
sigm2FWHM = 2*np.sqrt(2*np.log(2))
numberOfFiles = len(reducedFileIndex)
for iFile, filename in enumerate(reducedFileIndex['FILENAME'].data):
    if reducedFileIndex['AB'][i] == 'B':
        PSFwidths = [-1]
        continue

    # Read in the image
    thisImg = ai.ReducedScience.read(filename)

    # Estimate the PSF for this image
    PSFstamp, PSFparams = thisImg.get_psf()

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
