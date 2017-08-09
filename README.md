# Mimir_pyPol

This set of scripts uses the AstroImage package (at the jmontgom10 github site) to process Mimir data for an off-on-on-off (BAAB) polarimetry dither.

## The Observing Procedure

The observing procedure assumed for this set of scripts has the following pattern:

| Img. num. | On/Off target | HWP position |
|-----------|---------------|--------------|
| 1         | Off-target    | HWP pos \#1  |
| 2         | On-target     | HWP pos \#1  |
| 3         | On-target     | HWP pos \#1  |
| 4         | Off-target    | HWP pos \#1  |
| 5         | Off-target    | HWP pos \#2  |
| 6         | On-target     | HWP pos \#2  |

... etc ...

If you used a different observing procedure, you'll need to modify the Mimir_pyPol scripts to match your observing procedure. This is a non-trivial task.

## Cooperation with MSP-PPOL

The Mimir_pyPol package works in tandem with the Mimir Software Package (MSP) Photo-polarimetry tool (PPOL). Thus, all Mimir images must first be processed up through PPOL astrometric solutions (step 3a) before doing anything with Mimir_pyPol. Once those steps are complete, you can begin running the Mimir_pyPol scripts in the order described below.

The observing procedure uses the off-target frames to subtract thermal emission from the on-target frames. Thus, *none* of the frames should be flat-fielded or dark-subtracted before being processed up to this point.

# Script Descriptions and Procedure

## 00_renameGroups.pro

Late night astronomy can yield some misnamed targets. This allows the user to specify a sequence of files for which to alter the "OBJECT" header keyword. This is useful to do *before* initializing a working PPOL directory because then PPOL will correctly infer the name of each group using the updated header keywords.

## Initialize PPOL project

At this point a PPOL project, including all the BAAB observations to be processed, must be created. Simply read in the relevant files in the PPOL Step 1 tab.

## 00b_separate_AB_groups.pro

This script generate two separate group summary files per BAAB observation. Since the BAAB observing procedure includes frames which have absolutely no overlap on the sky, it is impossible for PPOL to match stars in PPOL step 5, and hence it is impossible to compute sky-transmission correction factors in PPOL step 6. Thus, each BAAB observation must separated into B-frames and A-frames. This step identifies which files belong to each of those two groups.

## 00c_fixAstrometry.pro

Some fields do not have enough stars to perform accurate automated astrometry. This script launches a simple GUI-style user-assisted astrometric solution program. You will simply need to click on a few highlighted stars in the image to help the computer locate those positions and compute a final astrometric solution.

## 00d_removeSkiJumpFiles.pro

In the case of resolved, extended emission polarimetry, the ski-jump features seem to  really wreak havoc on the results. This script simply loops through *all* images in the 'S3_Astrometry' directory and allows the user to identify (and backup/remove) ski-jump contaminated images. By removing these files, you guarantee that their ski-jumps will not end up in the final polarimetry.

I probably need to have the user do this by hand since some ski-jumps were not properly identified.

## 00e_fixBadPix.pro

Once the bad, ski-jump images have removed from the "S3_Astrometry" directory and the astrometry has been solved for all the remaining images, you must run this script to apply a uniform bad-pixel modeling procedure to *all* the images. This guarantees that there will be no errors due to non-uniform handling of bad pixels from image to image. This script reads in the Science quality images from the Basic Data Processor (BDP), applies the bad-pixel fixing procedure, and re-saves the image with the previously determined astrometric solution.

## Run PPOL through step 3

## 01_buildIndex.py

This Python script loops through all the files in the "S3_Astrometry" directory and compiles an "index" including the file name, date and time of observation, dither position (A or B) for each image.

## 02a_buildKokopelliMask.py

The "Kokopelli" feature in Mimir causes all kinds of problems. This script quickly generates a mask to use when coadding images.

## 02b_buildStarAndNebulaMasks.py

To estimate the background level in each image, an identical set of pixels need to be used for all images to be compared. Thus, the pixels located in-or-near star PSFs or nebular emission must be identified. This script identifies all the faint stars in the off-target emission.

The star or nebular emission pixels in 2MASS tiles are identified at saved to disk. These reference masks can be read in and *quickly* used to identify the corresponding masked pixels within a Mimir frame.

## 03a_computeBackgroundFreeImages.py

This script uses the bracketing off-target images to estimate the background level at the time of the on-target observations. This also subtracts an interpolated off-target image in order to account for the thermal emission component, which is significant for some NIR bands (and some temperatures). In the case where one of the bracketing off-target was bad (e.g., it contained a "ski-jump"), the single surviving off-target image was used to subtract the background level.

After subtraction, the on-target image is divided by the flat-field for that HWP. This accounts for the HWP specific shape of the flat-field and should remove the vast majority of instrumental polarization effects.

After subtraction *and* division, a fourth-order polynomial is fit to the non-nebular locations in residual image (as determined from the 2MASS based masks produced in 02b_buildStarAndNebulaMasks.py) to take care of any remaining irregularities.

The output of this procedure is a very flat image without any thermal emission or flat-field effects. These can be used to perform aperture photometry and estimate transmission coefficients, as done in the PPOL steps below.

## 03b_examineImagesAndMaskArtifacts.py

This Python script loops through each image of a given HWP for a given target and asks the user to identify regions of the images which need to be masked. This is especially useful for getting rid of artifacts from bright stars which might affect the average HWP images.

The user can also press the "up" or "down" arrow keys to change the Mimir_pyPol usage flag for each displayed image. Pressing "up" will force the usage flag to 1 (use this image) and pressing "down" will force the usage flag to 0 (don't use this image). This should be reflected by the color of the textbox in the upper left corner of the image. This will not affect the usage flags in PPOL, but it is useful for getting rid of questionable images before computing a final HWP average.

## 03c_createPlacehoderHWP0images.py

Unfortunately, some groups do not have *any* viable off-target frames available to subtract the thermal and airglow background from the on-target frames. When this happens in the HWP home (HWP = 1) position, the PPOL routines are not able to estimate transmission coefficients. This script simply loops through each group, looks for a viable on-target HWP = 1 image, and if none is found, then it creates a substitute image using the data from the next available HWP = 5, 9, or 13 image (these have the same IPPA value as the HWp = 1 images). This simply allows PPOL to proceed with Step 6 transmission coefficients, which will be necessary in order to produce accurate HWP images of each target.

## PPOL steps 3b - 6

At this point, the files in the 'bkgFreeHWPimages' directory must be copied into the PPOL 'S3_Astrometry' directory. Of course, it is a very good idea to store backup copies of the original PPOL S3 data products in case you need them later.

### PPOL Step 3b

<!-- Once this step has been completed, you need to set *all* the PPOL image usage flags for all the images to 1. This will tell PPOL to not assume that the images it was unable to solve astrometry for are simply missing.

After setting the usage flags to 1, run the PPOL step 3 "check astrometry" procedure. This will enable PPOL to check all of the proposed astrometric solutions as well as identify which files *are* in fact missing (e.g., the ski-jump files *should* be missing from the "S3_Astrometry" directory, and should have their usage flags reset to 0). -->

Now, launch PPOL, open the "Step 1" tab, select *all* the "A" (on-target) groups and click the "Reset Image Use Flags" button. Next, open the "Step 3" tab, add all the "A" groups to the processing queue, then click the "Test Astrometry Quality Button". This will read through *all* the images in all the groups and mark any missing images with a "use = 0" flag. This is the fastest way to get your image use flags to match the results of the Mimir_pyPol processing scripts.

### PPOL Step 4

Run the step 4 photometry as usual for al the "A" groups. This will measure the aperture photometry for all the stars in on-target field.

### PPOL Step 5

Run the step 5 star matching algorithm as usual for all the "A" groups.

### PPOL Step 6

Run the step 6 scaling factor computation as usual for all the "A" groups. This will generate lists of "correction factors" to use when computing final, average HWP images.

#### Metagrouping at Step 6

Now that the scaling factors for each individual group have been computed, all the groups for a given target and filter can be combined into a single "metagroup" at step 6. This compute the relative value of the correction factors between different observed groups (even across multiple nights). *These* are the correction factors which will ultimately be used to combine *all* the observations of a given target-filter-HWP angle combination into a single average image.

## 04_computeAverageHWPimages.py

This Python script computes one average image per target per filter per HWP rotation angle using the metagroup scaling factors computed in PPOL step 6. The results should be examined for sanity! Any oddities in these images are guaranteed to show up in the final polarimetry, so if something is off in the polarimetry, there's a good chance it happens in this script.

## 05_finalPolarimetry.py

Once the observations have been reduced to 16 master HWP rotation angle images per target, these can be further reduced to yield Stokes I, Q, and U images. That processing is taken care of in this script. First, the alignment is computed to sub-pixel accuracy, which is absolutely critical.

After the HWP images have been aligned (across both H- and Ks-band), the average Stokes images are computed. The uncertainty in the Q and U are forced to be equal to the average of the two (sQ' = sU' = 0.5*(sQ + sU)), and then a rotated Q and U image can be computed with with Gaussian error-propagation being correctly handled by the AstroImage "ReducedScience" class. These rotated and polarimetric efficiency corrected images are saved to disk.

## 06_photometricCalibration.py

The Stokes I image can be calibrated to the Johnson-Cousins photometric system by matching aperture photometry from the images to the entries from the 2MASS catalog. The linear regression of the following three equation is performed.

    m_2MASS_H - m_Mimir_H = zp_H + c*(m_Mimir_H - m_Mimir_K)

    m_2MASS_K - m_Mimir_K = zp_H + c*(m_Mimir_H - m_Mimir_K)

    m_2MASS_H - m_2MASS_K = c0   + c*(m_Mimir_H - m_Mimir_K)

The results of those regressions are used to construct calibrated Mimir H- and K-band images and a calibrated H-K color map.
