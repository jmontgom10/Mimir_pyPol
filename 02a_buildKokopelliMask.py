"""
Build a simple mask to cover up the 'Kokopelli' features in the Mimir detector.
This mask is useful at several stages of te data reduction, so it should be
generated once and used whenever necessary.
"""
# This script will generate a mask to cover Mimir's kokopelli crack.
import numpy as np
from astropy.io import fits
import pdb

# Label the end points of the kokopelli crack lines
x1Tup = (258, 458, 506, 586)
x2Tup = (588, 537, 588, 690)

y1Tup = (0, 634, 720, 581)
y2Tup = (581, 772, 582, 581)

# Compute the x and y positions of each pixel in the map
yy, xx = np.mgrid[0:1026, 0:1024]

# Initalize an empty array
kokopelliMask = np.zeros_like(xx, dtype = bool)

# Loop through each line
for xy1, xy2 in zip(zip(x1Tup, y1Tup), zip(x2Tup, y2Tup)):
    # Unpack the endpoints
    x1, y1 = xy1
    x2, y2 = xy2

    # Compute the slope of this line
    thisSlope = (y2-y1)/(x2-x1)

    # Compute 1 pixel perpendicular to the slope
    if np.abs(thisSlope) > 1e-2:
        dx = 1.0
        dy = (-1.0/thisSlope)*dx
        mag = np.sqrt(dx**2 + dy**2)
        dx /= mag
        dy /= mag
    else:
        dx = 0.0
        dy = 1.0

    # Compute the sign on dy
    signY = int(round(np.abs(dy)/dy))

    # Compute the y-intercept for this line
    if signY > 0:
        thisInterceptTop = (y1 + dy) - (x1 + dx)*thisSlope
        thisInterceptBot = (y1 - dy) - (x1 - dx)*thisSlope
    elif signY < 0:
        thisInterceptTop = (y1 - dy) - (x1 - dx)*thisSlope
        thisInterceptBot = (y1 + dy) - (x1 + dx)*thisSlope
    else:
        pdb.set_trace()

    # Compute the thise line y-value at each point
    thisLineYtop = thisSlope*xx + thisInterceptTop
    thisLineYbot = thisSlope*xx + thisInterceptBot

    # Compute the kokopelli region
    thisKokopelli = np.logical_and(
        np.logical_and(
        np.logical_and(
        (yy <= thisLineYtop),
        (yy >= thisLineYbot)),
        (xx > np.min([x1, x2]))),
        (xx < np.max([x1, x2])))


    # Combine this component of the map with the final map
    kokopelliMask = np.logical_or(kokopelliMask, thisKokopelli)

 # Add the center row to the mask
centerRowMask = np.logical_and(yy >= 512, yy <= 514)
kokopelliMask = np.logical_or(kokopelliMask, centerRowMask)

hdu = fits.PrimaryHDU(kokopelliMask.astype(int))
hdu.writeto('kokopelliMask.fits')

print('Done!')
