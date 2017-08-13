import os
import sys

# Add the AstroImage class
import astroimage as ai

# This script will allow you to simply select a range of images and change the
# "OBJECT" keyword in their header to match the appropriate group name. Simply
# change the start and end file numbers for each grup in the dictionary.

# This is the location of all pyBDP data (index, calibration images, reduced...)
pyBDP_data    = 'C:\\Users\\Jordan\\FITS_data\\PRISM_data\\pyBDP_data\\201612'

# This dictionary defines the night and first and last image for each group
# The following group dictionary is viable for the PRISM run in January 2015
groupDict = {
    'NGC4565_V12b': {'night': 20161209, 'numbers': (241, 264)},
}

pyBDP_reduced = os.path.join(pyBDP_data, 'pyBDP_reduced_images')

for group in groupDict.keys():
    print('Processing group {0}'.format(group))
    night    = '{0:d}'.format(groupDict[group]['night'])
    startNum, stopNum = groupDict[group]['numbers']
    # Loop through all the file numbers (be sure to include the LAST file!)
    for fileNum in range(startNum, stopNum + 1):
        # Construct the filename for this file
        strNum   = '{0:03d}'.format(fileNum)
        fileName = '.'.join([str(night), strNum, 'fits'])
        thisFile = os.path.join(pyBDP_reduced, fileName)

        if os.path.isfile(thisFile):
            # Read in the file
            thisImg = ai.ReducedScience.read(thisFile)

            # Modify the header to match the group name and re-write
            thisImg.header['OBJECT'] = group
            thisImg.write(clobber=True)

print('Done!')
