; This script will loop through all the images in the PPOL groups and let the
; user decide if a ski jump is present. Any images with ski-jumps will be
; deleted from the S3 directory (and thus excluded from the pyPol index file).
; 

;******************************************************************************
; THIS IS WHERE THE USER CAN DEFINE THE SPECIFICS OF THIS PPOL_PROJECT
;******************************************************************************
; Define the PPOL directory
PPOL_dir  = 'C:\Users\Jordan\FITS_data\Mimir_data\PPOL_Reduced\201611'
backup_dir = PPOL_dir + PATH_SEP() + 'S3_Backups'
IF ~FILE_TEST(backup_dir, /DIRECTORY) THEN FILE_MKDIR, backup_dir


; This will define the relative path to the group summary file
summaryFile = PPOL_dir + PATH_SEP() + 'S1_Image_Groups_and_Meta_Groups' + PATH_SEP() + 'Group_Summary.sav'

; Test if this file exists
IF NOT FILE_TEST(summaryFile) THEN BEGIN
  PRINT, 'Could not find summary file.'
  STOP
ENDIF

; Read in the summary file
RESTORE, summaryFile

; Reset to gray colortable
LOADCT, 0
WINDOW, 0, XS = 800, YS = 800

; Loop through each group
FOR iGroup = 0, ((*G_PTR).N_GROUPS - 1) DO BEGIN
  thisGroupCount = (*G_PTR).GROUP_NUMBERS[iGroup]  ; Grab the number of files in this group
  thisGroupName  = (*G_PTR).GROUP_NAMES[iGroup]
  PRINT, 'Starting group ' + thisGroupName + ' with ', thisgroupCount, ' images.'
;  if thisGRoupName eq 'NGC2023_K1_B' then stop
  ; Loop through each file in each group and display the image to the user
  FOR iFile = 0, (thisGroupCount - 1) DO BEGIN        ; Loop through all of the files in this group
    ; Grab the filename
    thisFile    = (*G_PTR).GROUP_IMAGES[iGroup, iFile]; Grab the BDP filename for this file
    
    ; If S2 has already been run, then you can use this code segment to skip over any files
    ; that were not idenified as "ski-jumps" in S2.
    fileBase    = FILE_BASENAME(thisFile)
    thisS2file  = PPOL_dir + PATH_SEP() + $           ; Grab the S3 filename for this file
      'S2_Ski_Jump_Fixes' + PATH_SEP() + fileBase

    IF ~FILE_TEST(thisS2file) THEN CONTINUE

    ; Construct the S3 file path
    thisS3file  = PPOL_dir + PATH_SEP() + $           ; Grab the S3 filename for this file
      'S3_Astrometry' + PATH_SEP() + fileBase

    ; If the S3 file does not exist, then don't bother
    IF ~FILE_TEST(thisS3file) THEN CONTINUE

    ; Read in the array and display to user
    arr = READFITS(thisFile, /SILENT)
    SKY, arr, skymode, skysig, /silent
    TVIM, arr, RANGE = skymode + [-2, +10]*skysig, TITLE = FILE_BASENAME(thisFile)
    
    ; Query if the image shows a ski-jump
    skiJump = ''
    READ, skiJump, PROMPT = 'Enter a non-null string to mark as a ski-jump image: '

    IF skiJump NE '' THEN BEGIN
      ; If the S3 file exists, then move it into the backup directory
      fileBase    = FILE_BASENAME(thisFile)
      thisS3file  = PPOL_dir + PATH_SEP() + $           ; Grab the S3 filename for this file
        'S3_Astrometry' + PATH_SEP() + fileBase
      IF FILE_TEST(thisS3file) THEN BEGIN
        PRINT, 'Backing up S3 file'
        moveFile = backup_dir + PATH_SEP() + fileBase
        FILE_MOVE, thisS3file, moveFile
      ENDIF ELSE BEGIN
        PRINT, 'No S3  file found. Doing nothing.'
      ENDELSE
    ENDIF
  ENDFOR
ENDFOR

; Free up the heap space from the pointers
PTR_FREE, G_PTR

PRINT, 'Done!'

END