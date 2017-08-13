; This script will quickly apply the "model_bad_pix" function to the
; "unflattened" BDP_step_12 images and 

; Set the relevant directories
BDP_dir = 'C:\Users\Jordan\FITS_data\Mimir_data\BDP_Data\201611\12_Science_Images'

PPOL_flattened_dir = 'C:\Users\Jordan\FITS_data\Mimir_data\PPOL_Reduced\201611\preFlattenedBackup\S3_Astrometry'
PPOL_not_flat_dir  = 'C:\Users\Jordan\FITS_data\Mimir_data\PPOL_Reduced\201611\notPreFlattened\S3_Astrometry'

; Search all the files in the original PPOL directory
S3files = FILE_SEARCH(PPOL_flattened_dir + PATH_SEP() + '*.fits')

; Loop through these files
FOREACH file, S3files DO BEGIN
  ; Grab the file basename
  thisBasename = FILE_BASENAME(file)

  ; Construct the output file name  
  outFile = PPOL_not_flat_dir + PATH_SEP() + thisBasename

  ; Skip files which have already been done
  IF FILE_TEST(outFile) THEN CONTINUE
  
  ; Extract the night from the basename
  night = (STRSPLIT(thisBasename, '.', /EXTRACT))[0]
  
  ; Construct the corresponding BDP filename
  BDPfile = BDP_dir + PATH_SEP() + night + PATH_SEP() + thisBasename
  ; Read in the header from the S3 file
  header = HEADFITS(file)
  
  ; Construct the BDP file and read in the BDP file array
  yyyymmdd = STRMID(thisBasename, 0, 8)
  BDPfile  = BDP_dir + PATH_SEP() + yyyymmdd + PATH_SEP() + 'polarimetry' + PATH_SEP() + thisBasename
  
  ; If the BDP file does not exist, then skip it!
  IF ~FILE_TEST(BDPfile) THEN CONTINUE
  
  ; Read in the BDP array
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
  
  ; Write to disk
  WRITEFITS, outFile, outArr, header

ENDFOREACH

PRINT, 'Done!'

END