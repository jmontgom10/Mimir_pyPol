; This script will loop through all the files in the S3 directory
; and re-process the arrays so that all data have passed through
; the 'model_bad_pixels' function
;
; This script will loop through each image of the S3 directory and...
; 1) Determine the original BDP or S2 data based on file name.
; 2) Read in the S3 header and the BDP or S2 image
; 3) Process the BDP or S2 image through the 'model_bad_pixels' function
; 4) Re-save the processed image with the S3 header.
;
; Check below the PYPOL_REPAIR_ASTROMETRY procedure to set the PPOL directory
; for this PPOL project.

; Setup the directories
PPOL_dir = 'C:\Users\Jordan\FITS_data\Mimir_data\PPOL_reduced' 
BDP_dir  = 'C:\Users\Jordan\FITS_data\Mimir_data\BDP_data'
S2_dir   = PPOL_dir + PATH_SEP() + 'S2_Ski_Jump_Fixes'
S3_dir   = PPOL_dir + PATH_SEP() + 'S3_Astrometry'

; Search for all the S3 files
S3_files = FILE_SEARCH(S3_dir, '*.fits')
numFiles = N_ELEMENTS(S3_files)

; Loop through each S3 files and apply the fixes
FOR i = 0, numFiles - 1 DO BEGIN
  ; Grab the file name properties
  thisFile     = S3_files[i]
  thisBasename = FILE_BASENAME(thisFile)
  
  ; Read in the header from the S3 file
  header = HEADFITS(thisFile)
  
  ; Construct the BDP file and read in the BDP file array
  yyyymmdd = STRMID(thisBasename, 0, 8)
  BDPfile  = BDP_dir + PATH_SEP() + yyyymmdd + PATH_SEP() + thisBasename
  arr      = READFITS(BDPfile)
  
  ; Compute image statistics
  SKY, arr, skyMode, skySig, /SILENT
  
  ; Identify anomalous pixels
  medImg = MEDIAN(arr, 5)
  badPix = ABS(arr - medImg)/skySig GT 4.0
  
  ; Count up number of neighboring bad pixels
  numNeighbors = FIX(0*arr)
  FOR dx = -1, 1 DO BEGIN
    FOR dy = -1, 1 DO BEGIN
      numNeighbors += SHIFT(badPix, dx, dy)
    ENDFOR
  ENDFOR
  
  badPix = badPix AND (numNeighbors LT 5) OR (arr LT -1E4)
  badInd = WHERE(badPix, numBad)
  IF numBad GT 0 THEN arr[badInd] = -1E6
  
  ; Now that the array and header have been read in,
  ; process the array and save the results
  outArr = MODEL_BAD_PIXELS(arr)
  WRITEFITS, thisFile, outArr, header
ENDFOR


PRINT, 'Done!'

END