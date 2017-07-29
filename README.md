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

The Mimir_pyPol package works in tandem with the Mimir Software Package (MSP) Photo-polarimetry tool (PPOL). Thus, all Mimir images must first be processed up through PPOL astrometric solutions (step 3) before doing anything with Mimir_pyPol. Once those steps are complete, you can begin running the Mimir_pyPol scripts in the order described below.

# Script Descriptions

## 00a_parse_BAAB.pro

This script generate two separate group summary files per BAAB observation. Since the BAAB observing procedure includes frames which have absolutely no overlap on the sky, it is impossible for PPOL to match stars in PPOL step 5, and hence it is impossible to compute sky-transmission correction factors in PPOL step 6. Thus, each BAAB observation must separated into B-frames and A-frames. This step identifies which files belong to each of those two groups.

## 00b_testForSkiJump.pro

In the case of resolved, extended emission polarimetry, the ski-jump features seem to  really wreak havoc on the results. This script simply identifies which images contain ski-jumps and deletes those files from the PPOL "S3_Astrometry" directory. By deleting these files, you guarantee that their ski-jumps will not end up in the final polarimetry.

## 00c_fixAstrometry.pro

Some fields do not have enough stars to perform accurate automated astrometry. This script launches a simple GUI-style user-assisted astrometric solution program. You will simply need to click on a few highlighted stars in the image to help the computer locate those positions and compute a final astrometric solution.

## 00d_fixBadPix.pro

Once the bad, ski-jump images have removed from the "S3_Astrometry" directory and the astrometry has been solved for all the remaining images, you must run this script to apply a uniform bad-pixel modeling procedure to *all* the images. This guarantees that there will be no errors due to non-uniform handling of bad pixels from image to image. This script reads in the Science quality images from the Basic Data Processor (BDP), applies the bad-pixel fixing procedure, and re-saves the image with the previously determined astrometric solution.

Once this step has been completed, you need to set *all* the PPOL image usage flags for all the images to 1. This will tell PPOL to not assume that the images it was unable to solve astrometry for are simply missing.

After setting the usage flags to 1, run the PPOL step 3 "check astrometry" procedure. This will enable PPOL to check all of the proposed astrometric solutions as well as identify which files *are* in fact missing (e.g., the ski-jump files *should* be missing from the "S3_Astrometry" directory, and should have their usage flags reset to 0).

## 01_buildIndex.py

This Python script loops through all the files in the "S3_Astrometry" directory and compiles an "index" including the file name, date and time of observation, dither position (A or B) for each image.

## 02_measurePSFs.py

This Python script loops back through all of those images and estimates the average PSF for each image. The file index is then appended with a column containing the best estimate of the PSF for each image. This can be used to identify observations with poor seeing and excluding them from the analysis.

## 03_buildMasks.py

This Python script loops through each image of a given HWP for a given target and asks the user to identify regions of the images which need to be masked. This is especially useful for getting rid of artifacts from bright stars which might affect either (1) supersky images or (2) median-filtered-mean images for each HWP.

## 04_avgBAABditherPolAngImages.py

This Python script computes one average image per target per HWP rotation. The results should be examined for sanity! Any oddities in these images are guaranteed to show up in the final polarimetry, so if something is off in the polarimetry, there's a good chance it happens in this script.

## 05_finalPolarimetry.py

Once the observations have been reduced to 16 master HWP rotation angle images per target, these can be further reduced to yield Stokes I, Q, and U images. That processing is taken care of in this script. First, the alignment is computed to sub-pixel accuracy, which is absolutely critical.

After the HWP images have been aligned (across both H- and Ks-band), the average Stokes images are computed. The uncertainty in the Q and U are forced to be equal to the average of the two (sQ' = sU' = 0.5*(sQ + sU)), and then a rotated Q and U image can be computed with with Gaussian error-propagation being correctly handled by the AstroImage "ReducedScience" class. These rotated and polarimetric efficiency corrected images are saved to disk.

## 06b_photometricCalibration.py

The Stokes I image can be calibrated to the Johnson-Cousins photometric system by matching aperture photometry from the images to the entries from the 2MASS catalog. The linear regression of the following three equation is performed.

    m_2MASS_H - m_Mimir_H = zp_H + c*(m_Mimir_H - m_Mimir_K)

    m_2MASS_K - m_Mimir_K = zp_H + c*(m_Mimir_H - m_Mimir_K)

    m_2MASS_H - m_2MASS_K = c0   + c*(m_Mimir_H - m_Mimir_K)

The results of those regressions are used to construct calibrated Mimir H- and K-band images and a calibrated H-K color map.
