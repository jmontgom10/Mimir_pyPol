# Marks specific "bad" groups or files manually identified in the previous step
# as "unusable" in the file index.

import os
import sys
import numpy as np
from astropy.io import ascii
from astropy.table import Table as Table
from astropy.table import Column as Column
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.stats import gaussian_fwhm_to_sigma, sigma_clipped_stats
from photutils import detect_threshold, detect_sources
from scipy.ndimage.filters import median_filter, gaussian_filter
import matplotlib.pyplot as plt

################################################################################
# Specify the bad groups or files
################################################################################
badGroupIDs  = [
    5, # ???
    41, # NGC2023_K4_G41
    36, # Orion_cal_S_161120_G36 (gridding - maybe not too important)
    43, # Orion_cal_H_L_161123_G43
    38, # Orion_cal_K_S_161120_G38
    11, # NGC7023_H1_G11
    56, # M78_K6_G56
    62 # NGC2023_K6_62 (weak HWP features... unclear if this is really a failure...)
]
# List the ski-jump contaminated files
badFilenames = [
    # PPOL identified
    '20161119.038_LDFC.fits', '20161119.097_LDFC.fits',
    '20161119.104_LDFC.fits', '20161119.129_LDFC.fits',
    '20161119.132_LDFC.fits', '20161119.133_LDFC.fits',
    '20161119.136_LDFC.fits', '20161119.203_LDFC.fits',
    '20161119.209_LDFC.fits', '20161119.241_LDFC.fits',
    '20161119.265_LDFC.fits', '20161119.276_LDFC.fits',
    '20161119.279_LDFC.fits', '20161119.288_LDFC.fits',
    '20161119.298_LDFC.fits', '20161119.321_LDFC.fits',
    '20161119.323_LDFC.fits', '20161119.324_LDFC.fits',
    '20161119.327_LDFC.fits', '20161119.328_LDFC.fits',
    '20161119.329_LDFC.fits', '20161119.330_LDFC.fits',
    '20161119.331_LDFC.fits', '20161119.332_LDFC.fits',
    '20161119.333_LDFC.fits', '20161119.334_LDFC.fits',
    '20161119.335_LDFC.fits', '20161119.336_LDFC.fits',
    '20161119.337_LDFC.fits', '20161119.338_LDFC.fits',
    '20161119.339_LDFC.fits', '20161119.340_LDFC.fits',
    '20161119.345_LDFC.fits', '20161119.346_LDFC.fits',
    '20161119.347_LDFC.fits', '20161119.348_LDFC.fits',
    '20161119.349_LDFC.fits', '20161119.350_LDFC.fits',
    '20161119.351_LDFC.fits', '20161119.352_LDFC.fits',
    '20161119.353_LDFC.fits', '20161119.354_LDFC.fits',
    '20161119.355_LDFC.fits', '20161119.356_LDFC.fits',
    '20161119.357_LDFC.fits', '20161119.359_LDFC.fits',
    '20161119.360_LDFC.fits', '20161119.361_LDFC.fits',
    '20161119.362_LDFC.fits', '20161119.363_LDFC.fits',
    '20161119.364_LDFC.fits', '20161119.365_LDFC.fits',
    '20161119.366_LDFC.fits', '20161119.367_LDFC.fits',
    '20161119.368_LDFC.fits', '20161119.369_LDFC.fits',
    '20161119.370_LDFC.fits', '20161119.371_LDFC.fits',
    '20161119.372_LDFC.fits', '20161119.377_LDFC.fits',
    '20161119.378_LDFC.fits', '20161119.379_LDFC.fits',
    '20161119.380_LDFC.fits', '20161119.381_LDFC.fits',
    '20161119.382_LDFC.fits', '20161119.383_LDFC.fits',
    '20161119.384_LDFC.fits', '20161119.404_LDFC.fits',
    '20161119.414_LDFC.fits', '20161119.417_LDFC.fits',
    '20161119.421_LDFC.fits', '20161119.429_LDFC.fits',
    '20161119.449_LDFC.fits', '20161119.508_LDFC.fits',
    '20161119.721_LDFC.fits', '20161119.722_LDFC.fits',
    '20161119.724_LDFC.fits', '20161119.732_LDFC.fits',
    '20161119.737_LDFC.fits', '20161119.743_LDFC.fits',
    '20161119.756_LDFC.fits', '20161119.770_LDFC.fits',
    '20161119.772_LDFC.fits', '20161119.774_LDFC.fits',
    '20161120.3327_LDFC.fits', '20161120.3458_LDFC.fits',
    '20161120.3482_LDFC.fits', '20161120.3488_LDFC.fits',
    '20161120.3620_LDFC.fits', '20161120.3687_LDFC.fits',
    '20161120.3715_LDFC.fits', '20161120.3726_LDFC.fits',
    '20161120.3734_LDFC.fits', '20161123.1051_LDFC.fits',
    '20161123.1056_LDFC.fits', '20161123.1062_LDFC.fits',
    '20161123.1107_LDFC.fits', '20161123.1118_LDFC.fits',
    '20161123.1177_LDFC.fits', '20161123.1230_LDFC.fits',
    '20161123.1281_LDFC.fits', '20161123.1412_LDFC.fits',
    '20161123.1417_LDFC.fits', '20161123.1535_LDFC.fits',
    '20161123.159_LDFC.fits', '20161123.1600_LDFC.fits',
    '20161123.169_LDFC.fits', '20161123.1707_LDFC.fits',
    '20161123.1834_LDFC.fits', '20161123.185_LDFC.fits',
    '20161123.324_LDFC.fits', '20161123.344_LDFC.fits',
    '20161123.353_LDFC.fits', '20161123.354_LDFC.fits',
    '20161123.413_LDFC.fits', '20161123.469_LDFC.fits',
    '20161123.477_LDFC.fits', '20161123.478_LDFC.fits',
    '20161123.625_LDFC.fits', '20161123.640_LDFC.fits',
    '20161123.650_LDFC.fits', '20161123.675_LDFC.fits',
    '20161123.694_LDFC.fits', '20161123.707_LDFC.fits',
    '20161123.737_LDFC.fits', '20161123.742_LDFC.fits',
    '20161123.753_LDFC.fits', '20161123.769_LDFC.fits',
    '20161123.801_LDFC.fits', '20161123.816_LDFC.fits',
    '20161123.817_LDFC.fits', '20161123.905_LDFC.fits',
    '20161123.908_LDFC.fits', '20161123.964_LDFC.fits',
    '20161123.967_LDFC.fits', '20161123.984_LDFC.fits',
    '20161124.1078_LDFC.fits', '20161124.1093_LDFC.fits',
    '20161124.1106_LDFC.fits', '20161124.1123_LDFC.fits',
    '20161124.1175_LDFC.fits', '20161124.1182_LDFC.fits',
    '20161124.1197_LDFC.fits', '20161124.1203_LDFC.fits',
    '20161124.1207_LDFC.fits', '20161124.1208_LDFC.fits',
    '20161124.1211_LDFC.fits', '20161124.1212_LDFC.fits',
    '20161124.1216_LDFC.fits', '20161124.1246_LDFC.fits',
    '20161124.1265_LDFC.fits', '20161124.1274_LDFC.fits',
    '20161124.1289_LDFC.fits', '20161124.1360_LDFC.fits',
    '20161124.1364_LDFC.fits', '20161124.1375_LDFC.fits',
    '20161124.1398_LDFC.fits', '20161124.1409_LDFC.fits',
    '20161124.1410_LDFC.fits', '20161124.1411_LDFC.fits',
    '20161124.1412_LDFC.fits', '20161124.1413_LDFC.fits',
    '20161124.1423_LDFC.fits', '20161124.1424_LDFC.fits',
    '20161124.1426_LDFC.fits', '20161124.1427_LDFC.fits',
    '20161124.1428_LDFC.fits', '20161124.1429_LDFC.fits',
    '20161124.1438_LDFC.fits', '20161124.1439_LDFC.fits',
    '20161124.1440_LDFC.fits', '20161124.1442_LDFC.fits',
    '20161124.1443_LDFC.fits', '20161124.1444_LDFC.fits',
    '20161124.1445_LDFC.fits', '20161124.1454_LDFC.fits',
    '20161124.1455_LDFC.fits', '20161124.1456_LDFC.fits',
    '20161124.1457_LDFC.fits', '20161124.193_LDFC.fits',
    '20161124.290_LDFC.fits', '20161124.307_LDFC.fits',
    '20161124.331_LDFC.fits', '20161124.341_LDFC.fits',
    '20161124.354_LDFC.fits', '20161124.383_LDFC.fits',
    '20161124.421_LDFC.fits', '20161124.448_LDFC.fits',
    '20161124.464_LDFC.fits', '20161124.698_LDFC.fits',
    '20161124.858_LDFC.fits', '20161124.865_LDFC.fits',
    '20161124.911_LDFC.fits', '20161124.946_LDFC.fits',
    '20161124.962_LDFC.fits', '20161124.971_LDFC.fits',
    '20161126.1063_LDFC.fits', '20161126.1160_LDFC.fits',
    '20161126.1186_LDFC.fits', '20161126.1204_LDFC.fits',
    '20161126.1265_LDFC.fits', '20161126.1342_LDFC.fits',
    '20161126.1354_LDFC.fits', '20161126.1362_LDFC.fits',
    '20161126.1384_LDFC.fits', '20161126.257_LDFC.fits',
    '20161126.285_LDFC.fits', '20161126.307_LDFC.fits',
    '20161126.342_LDFC.fits', '20161126.349_LDFC.fits',
    '20161126.434_LDFC.fits', '20161126.435_LDFC.fits',
    '20161126.440_LDFC.fits', '20161126.446_LDFC.fits',
    '20161126.705_LDFC.fits', '20161126.808_LDFC.fits',
    '20161126.833_LDFC.fits', '20161126.839_LDFC.fits',
    '20161126.863_LDFC.fits', '20161126.877_LDFC.fits',
    '20161126.878_LDFC.fits', '20161126.890_LDFC.fits',
    '20161126.891_LDFC.fits', '20161126.893_LDFC.fits',
    '20161126.897_LDFC.fits', '20161126.917_LDFC.fits',
    '20161126.968_LDFC.fits', '20161126.995_LDFC.fits',

    # Manually identified
    '20161126.1160_LDFC.fits',
    '20161126.1204_LDFC.fits',
    '20161126.1265_LDFC.fits',
    '20161126.1384_LDFC.fits',
    '20161126.1412_LDFC.fits',
    '20161123.1417_LDFC.fits',
    '20161123.1160_LDFC.fits',
    '20161123.1535_LDFC.fits',
    '20161123.640_LDFC.fits',
    '20161123.707_LDFC.fits',
    '20161123.742_LDFC.fits',
    '20161124.865_LDFC.fits',
    '20161124.911_LDFC.fits',
    '20161126.1160_LDFC.fits',
    '20161126.1186_LDFC.fits',
    '20161126.1204_LDFC.fits',
    '20161126.1265_LDFC.fits',
    '20161126.1342_LDFC.fits',
    '20161126.1354_LDFC.fits',
    '20161126.1362_LDFC.fits',
    '20161126.1384_LDFC.fits',
    '20161126.307_LDFC.fits',
    '20161126.342_LDFC.fits',
    '20161126.349_LDFC.fits',
]

# This is the location where all pyPol data will be saved
pyPol_data = 'C:\\Users\\Jordan\\FITS_data\\Mimir_data\\pyPol_Reduced\\201611\\'

# Build the path to the supersky directory
bkgImagesDir = os.path.join(pyPol_data, 'bkgImages')

# Read in the indexFile data and select the filenames
indexFile = os.path.join(pyPol_data, 'reducedFileIndex.csv')
fileIndex = Table.read(indexFile, format='csv')
