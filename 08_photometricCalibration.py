# -*- coding: utf-8 -*-
"""
Created on Sat Nov 28 17:08:13 2015

@author: jordan
"""
import os
import numpy as np
from astropy.table import Table, hstack
from astropy.stats import sigma_clipped_stats
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_area
from astropy import units as u
from astropy.coordinates import SkyCoord
from matplotlib import pyplot as plt
from photutils import daofind, aperture_photometry, CircularAperture, CircularAnnulus
import pdb
from pyPol import Image

#Setup the path delimeter for this operating system
delim = os.path.sep

# Grab all the *.fits files in the reduced science data directory
reducedDir = '/home/jordan/ThesisData/PRISM_Data/Reduced_data'

# Setup new directory for polarimetry data
polarimetryDir = reducedDir + delim + 'Polarimetry'
if (not os.path.isdir(polarimetryDir)):
    os.mkdir(polarimetryDir, 0o755)

polAngDir = polarimetryDir + delim + 'polAngImgs'
if (not os.path.isdir(polAngDir)):
    os.mkdir(polAngDir, 0o755)

stokesDir = polarimetryDir + delim + 'stokesImgs'
if (not os.path.isdir(stokesDir)):
    os.mkdir(stokesDir, 0o755)

# Read in the indexFile data and select the filenames
print('\nReading file index from disk')
indexFile = 'fileIndex.csv'
fileIndex = Table.read(indexFile, format='csv')
#fileIndex = ascii.read(indexFile, guess=False, delimiter=',')
fileList  = fileIndex['Filename']

# Determine which parts of the fileIndex pertain to science images
useFiles = np.logical_and((fileIndex['Use'] == 1), (fileIndex['Dither'] == 'ABBA'))

# Cull the file index to only include files selected for use
fileIndex = fileIndex[np.where(useFiles)]

# Group the fileIndex by...
# 1. Target
# 2. Waveband
# 3. Dither (pattern)
fileIndexByTarget = fileIndex.group_by(['Target', 'Waveband', 'Dither'])

# Define any required conversion constants
rad2deg = (180.0/np.pi)
deg2rad = (np.pi/180.0)

# Use the following data for final calibration
# ...
# Magnitude zero point from Bessel 1990, PASP 102, 1181 
# Extinction from Cardelli Clayton & Mathis 1989, ApJ 345, 245 or Rieke and Lebosky 1985, ApJ
# ...
#                   Bands and zero point flux [in Jy = 10^(-26) W /(m^2 Hz)]
zeroFlux = dict(zip(['U',  'B',  'V',  'R'],
                    [1884, 4646, 3953, 2875]))

# Loop through each group
groupKeys = fileIndexByTarget.groups.keys
for group in fileIndexByTarget.groups:
    # Grab the current target information
    thisTarget   = str(np.unique(group['Target'].data)[0])
    thisWaveband = str(np.unique(group['Waveband'].data)[0])
    numImgs      = len(group)
    print('\nProcessing images for'.format(numImgs))
    print('\tTarget   : {0}'.format(thisTarget))
    print('\tWaveband : {0}'.format(thisWaveband))
    
    # Calibrate stokes intensity from USNOB stars...
    catalogFile = stokesDir + delim + thisTarget + '_stars.xml'
    starCatalog = Table.read(catalogFile)
    thisMag     = thisWaveband + 'mag'
    theseStars  = np.logical_not(starCatalog[thisMag].mask)

    # Cull the catalog to those entries with relevant photometric data
    starCatalog = starCatalog[np.where(theseStars)]

    # Form a "catalog" of position entries for matching
    ra1      = starCatalog['RAJ2000']
    dec1     = starCatalog['DEJ2000']
    catalog1 = SkyCoord(ra = ra1, dec = dec1, frame = 'fk5')
    
    # Read in the image and find tars in the image
    Ifile = (stokesDir + delim +
             '_'.join([thisTarget, thisWaveband, 'I']) + '.fits')
    stokesI   = Image(Ifile)
    mean, median, std = sigma_clipped_stats(stokesI.arr, sigma=3.0, iters=5)
    threshold = median + 3.0*std
    fwhm    = 3.0
    sources = daofind(stokesI.arr, threshold, fwhm, ratio=1.0, theta=0.0,
                      sigma_radius=1.5, sharplo=0.2, sharphi=1.0,
                      roundlo=-1.0, roundhi=1.0, sky=0.0,
                      exclude_border=True)
    
    # Convert source positions to RA and Dec
    wcs      = WCS(stokesI.header)
    ADstars  = wcs.all_pix2world(sources['xcentroid'], sources['ycentroid'], 0)
    catalog2 = SkyCoord(ra = ADstars[0]*u.deg, dec = ADstars[1]*u.deg, frame = 'fk5')
    
    
    ###
    ### This slow, meat-axe method was useful for verification.
    ### It produces the same results as the method below.
    ###
#    # Loop through each of the detected sources, and check for possible confusion
#    keepStars    = []
#    numCat1Match = []
#    numCat2Match = []
#    for i in range(len(catalog2)):
#        # Establish the coordinates of the current star
#        thisCoord = SkyCoord(ra = catalog2[i].ra, dec = catalog2[i].dec)
#        
#        # Compute the distances from this star to other stars in both catalogs
#        d2d_cat1 = thisCoord.separation(catalog1)
#        d2d_cat2 = thisCoord.separation(catalog2)
#        
#        # Check if there is more than one star nearby in EITHER catalog
#        numCat1Match.append(np.sum(d2d_cat1 < 20*u.arcsec))
#        numCat2Match.append(np.sum(d2d_cat2 < 20*u.arcsec))
#        keepStars.append(numCat1Match[i] == 1 and numCat2Match[i] == 1)

     #Search for all possible matches within 10 arcsec of a given detected source
    idxc1, idxc2, d2d, d3d = catalog2.search_around_sky(catalog1, 20*u.arcsec)
    
    # Detect which matches from catalog2 to catalog1 are unique
    bins, freq  = np.unique(idxc2, return_inverse=True)
    cat2Keep    = np.bincount(freq) == 1
    isoCatalog2 = bins[np.where(cat2Keep)]

    # Cull the decteced sources catalog to only include good, isolated stars
    xStars   = sources['xcentroid'][isoCatalog2]
    yStars   = sources['ycentroid'][isoCatalog2]
    
    # Now check for saturation near this star
    saturatedStars = []
    for xStar, yStar in zip(xStars, yStars):
        lf, rt = xStar - 10, xStar + 10
        bt, tp = yStar - 10, yStar + 10
        patch = stokesI.arr[lf:rt, bt:tp]
        saturatedStars.append(np.sum(np.logical_or(patch > 9e3, patch < -100)) > 1)
    
    # Check for stars near the edge of the image
    ny, nx    = stokesI.arr.shape
    edgeStars = np.logical_or(
                np.logical_or(xStars < 40, xStars > (nx - 40)),
                np.logical_or(yStars < 40, yStars > (ny - 40)))
    
    # Now let's do aperture photometry on the remaining sources in the image
    # 1. Setup the apertures
    sourcePos = [xStars, yStars]
    apertures = CircularAperture(sourcePos, r = 6.0)
    annulus_apertures = CircularAnnulus(sourcePos, r_in=12., r_out=14.)
    # 2. Perform the basic photometry
    rawflux_table = aperture_photometry(stokesI.arr, apertures)
    bkgflux_table = aperture_photometry(stokesI.arr, annulus_apertures)
    phot_table = hstack([rawflux_table, bkgflux_table], table_names=['raw', 'bkg'])
    
    # 3. Compute background contribution and subtract from raw photometry
    bkg_mean = phot_table['aperture_sum_bkg'] / annulus_apertures.area()
    bkg_sum = bkg_mean * apertures.area()
    final_sum = phot_table['aperture_sum_raw'] - bkg_sum
    phot_table['residual_aperture_sum'] = final_sum

    # Compute the signal-to-noise ratio and find the stars with SNR < 3.0
    SNR          = final_sum/bkg_sum
    bkgDominated = SNR < 1.0
    
#    ###########################################################################
#    # PRINT OUT THE PHOTOMETRY TO CHECK FOR CONSISTENCY
#    ###########################################################################
#    xFmtStr    = '{x[0]:>6}.{x[1]:<3}'
#    yFmtStr    = '{y[0]:>6}.{y[1]:<3}'
#    starFmtStr = '{star[0]:>9}.{star[1]:<3}'
#    bkgFmtStr  = '{bkg[0]:>9}.{bkg[1]:<3}'
#    snrFmtStr  = '{snr[0]:>9}.{snr[1]:<3}'
#    print('final photometry is...')
#    print('      x         y        Star Flux     Bkg Flux       SNR')
#    print('===========================================================')
#    printStr = xFmtStr + yFmtStr + starFmtStr + bkgFmtStr + snrFmtStr
#    for i in range(len(SNR)):
#        xVal = str(xStars[i]).split('.')
#        xVal[1] = (xVal[1])[0:3]
#        yVal = str(yStars[i]).split('.')
#        yVal[1] = (yVal[1])[0:3]
#        starVal = str(final_sum[i]).split('.')
#        starVal[1] = (starVal[1])[0:3]
#        bkgVal = str(bkg_sum[i]).split('.')
#        bkgVal[1] = (bkgVal[1])[0:3]
#        snrVal = str(SNR[i]).split('.')
#        snrVal[1] = (snrVal[1])[0:3]
#        print(printStr.
#              format(x = xVal, y = yVal, star = starVal, bkg = bkgVal, snr = snrVal))
    
    # Cull the lists to juts keep
    # non-saturated,
    # non-edge,
    # non-background-dominated stars
    keepStars  = np.logical_not(
                 np.logical_or(
                 np.logical_or(saturatedStars, edgeStars), bkgDominated))
    keepInds   = np.where(keepStars)
    xStars     = xStars[keepInds]
    yStars     = yStars[keepInds]
    phot_table = phot_table[keepInds]
    ADstars    = wcs.all_pix2world(xStars, yStars, 0)
    catalog2   = SkyCoord(ra = ADstars[0]*u.deg, dec = ADstars[1]*u.deg, frame='fk5')
    
    # Now that we have a final star list,
    # let's match this catalog to the NOMAD "catalog1"
    # (After this: starCatalog1[i] <==> catalog2[i] for all i)
    idx, sep2d, dist3d = catalog2.match_to_catalog_sky(catalog1)
    starCatalog1 = starCatalog[idx]
    
    ###########################################################################
    # Check with the user if the selected stars are photometrically viable
    ###########################################################################
    fig, ax, axIm = stokesI.show(cmap='cubehelix', vmin=0, vmax=300)
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    tr_fk5 = ax.get_transform("fk5")
    
#    plotAps = CircularAnnulus(catalog2[0], catalog2[1], rin = 12.0, rout = 14.0)
#    plotAps.positions = wcs.all_pix2world()
#    annulus_apertures.plot(ax=ax, transform=tr_fk5, color='red')
    line1, = ax.plot(ADstars[0], ADstars[1],
                     transform=tr_fk5, linestyle='None', 
                     marker='o', markersize=10.0, markeredgewidth=2.0,
                     markeredgecolor='red', fillstyle='none')
    axTexts = []
    for i in range(len(ADstars[0])):
        axTexts.append(ax.text(ADstars[0][i], ADstars[1][i] + 10.0/3600.0,
                               str(i), horizontalalignment='center',
                               color='green', transform=tr_fk5))
    line2, = ax.plot(starCatalog1['RAJ2000'], starCatalog1['DEJ2000'],
                     transform=tr_fk5, linestyle='None',
                     marker='x', markeredgewidth=2.0,
                     markeredgecolor='blue', fillstyle='none')
    ax.set_xlim(*xlim), ax.set_ylim(*ylim)
    
    # Print a prompt to the user
    print('\nType the index of stars which should be removed from this matching.')
    print('Type "done" to exit deletion loop.')
    inputText = ''
    starIndices = []
    while inputText.upper() != 'DONE':
        inputText = input("Enter a star index: ")
        if inputText.isdigit():
            # Remove stars that were selected
            starIndex = int(inputText)
            starIndices.append(starIndex)
            axTexts[starIndex].set_color('red')
            fig.canvas.draw()
    
    plt.close(fig)
    
    # Remove the specified stars from the catalogs and multFactors
    if len(starIndices) > 0:
        keepStars = np.array([True]*len(ADstars[0]), dtype=bool)
        keepStars[np.array(starIndices)] = False
        ADstars      = (ADstars[0][keepStars], ADstars[1][keepStars])
        starCatalog1 = starCatalog1[keepStars]
        catalog2     = catalog2[keepStars]
        phot_table   = phot_table[keepStars]
    
    # Compute the expected instrumental fluxes for these stars
    instFlux = np.array(10.0**(-0.4*starCatalog1[thisMag]))
    
    # Divide and compute a comparative multiplicative factor
    multFacts = np.array(phot_table['residual_aperture_sum']/instFlux)
    multFact, med, std = sigma_clipped_stats(multFacts, sigma=2.5, iters=5)
    multFact1 = 1.0/multFact
    
    # Plot the deltaMag distribution to check if it is centered about zero
    num1, bins1, patches1 = plt.hist(multFacts)
    non_clipped = np.abs(multFacts - med) < 1.5*std
    num2, bins2, patches2 = plt.hist(multFacts[np.where(non_clipped)])
    plt.plot([multFact, multFact], [0, np.max(np.array([num1, num2]))])
    plt.ion()
    plt.show()
    plt.ioff()
    input("Press ENTER to continue")
    plt.close()
#    pdb.set_trace()

    
#    ###########################################################################
#    # REDO PHOTOMETRY TABLE TO CHECK IF IT GIVES THE EXPECTED ANSWERS
#    ###########################################################################
#    fluxImg = multFact1*stokesI.arr
#    # 1. Setup the apertures
#    sourcePos = [xStars, yStars]
#    apertures = CircularAperture(sourcePos, r = 6.0)
#    annulus_apertures = CircularAnnulus(sourcePos, r_in=12., r_out=14.)
#    # 2. Perform the basic photometry
#    rawflux_table = aperture_photometry(fluxImg, apertures)
#    bkgflux_table = aperture_photometry(fluxImg, annulus_apertures)
#    phot_table = hstack([rawflux_table, bkgflux_table], table_names=['raw', 'bkg'])
#    
#    # 3. Compute background contribution and subtract from raw photometry
#    bkg_mean = phot_table['aperture_sum_bkg'] / annulus_apertures.area()
#    bkg_sum = bkg_mean * apertures.area()
#    final_sum = phot_table['aperture_sum_raw'] - bkg_sum
#    phot_table['residual_aperture_sum'] = final_sum
#    
#    # Compute magnitudes and magnitude differences as a double check
#    # (these should be centered around zero)
#    instMag   = -2.5*np.log10(phot_table['residual_aperture_sum'])
#    deltaMags = np.array(instMag - starCatalog1[thisMag])
#    deltaMag, med, std = sigma_clipped_stats(deltaMags, sigma=3.0, iters=5)
#
#    # Plot the deltaMag distribution to check if it is centered about zero
#    num, bins, patches = plt.hist(deltaMags)
#    plt.plot([deltaMag, deltaMag], [0, np.max(num)])
#    plt.show()
    
    # Perform the final conversion to Jy/arcsec^2
    wcs = WCS(stokesI.header)
    pixel_area = proj_plane_pixel_area(wcs)*(3600**2)
    BUNIT = 'uJy/sqarcs'
    BSCALE = zeroFlux[thisWaveband]*(1e6)*(multFact1/pixel_area)
    BZERO  = 0.0
    
    stokesI.header.set('BUNIT', value = BUNIT,
               comment='Physical units for the image')
    stokesI.header.set('BSCALE', value = BSCALE, after='BUNIT',
               comment='Conversion factor for physical units', )
    stokesI.header.set('BZERO', value = BZERO, after='BSCALE',
               comment='Zero level for the physical units.')
    
    # Finally write the updated header to disk
    stokesI.write()