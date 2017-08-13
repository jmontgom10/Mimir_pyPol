; Define a procedure to perform the updating
PRO UPDATE_HEADER_KEYWORD, file, key, value
  ; Read in the file
  tmpArr = READFITS(file, tmpHeader)
  
  ; Modify the header to hold the expected values
  SXADDPAR, tmpHeader, key, value
  
  ; Resave the file with the updated header
  WRITEFITS, file, tmpArr, tmpHeader
END

; Specify the location of the BDP data (for whole observing run)
BDP_dir  = 'C:\Users\Jordan\FITS_data\Mimir_data\BDP_Data\201611'
PPOL_dir = 'C:\Users\Jordan\FITS_data\Mimir_data\PPOL_Reduced\201611'
S2_dir   = PPOL_dir + PATH_SEP() + 'S2_Ski_Jump_Fixes'
S3_dir   = PPOL_dir + PATH_SEP() + 'S3_Astrometry'

; Specify a list of nights on which the data need to be modified
nights = [20161123, 20161124]

; Specify the first and last file number to be modified in the group
startFile = [548, 482]
endFile   = [608, 545]

; Specify the OBJECT value which SHOULD be in the header
objectName = ['NGC7023_H4']

; Loop through each of the groups to modify
FOR iGroup = 0, N_ELEMENTS(nights) -1 DO BEGIN
  ; Now loop through the files in this group, read them in and modify
  FOR iFile = startFile[iGroup] - 2, endFile[iGroup] + 2 DO BEGIN
    ; Construct the expected basename of the file
    filebasename = STRTRIM(nights[iGroup], 2) + '.' + STRTRIM(iFile, 2) + '_LDFC.fits'

    ;***********************
    ;* HANDLE THE BDP FILE *
    ;***********************
    ; Construct the BDP filename
    BDP_dir  = BDP_dir + PATH_SEP() + STRTRIM(nights[iGroup], 2) + PATH_SEP() + 'polarimetry'
    BDP_file = BDP_dir + PATH_SEP() + filebasename

    ; Construct the PPOL filenames
    S2_file = S2_dir + PATH_SEP() + filebasename
    S3_file = S3_dir + PATH_SEP() + filebasename

    ; Test if the file exists and skip those that don't
    IF FILE_TEST(BDP_file) THEN BEGIN
      UPDATE_HEADER_KEYWORD, BDP_file, 'OBJECT', objectName[iGroup]
    ENDIF ELSE IF FILE_TEST(S2_file) THEN BEGIN
      UPDATE_HEADER_KEYWORD, S2_file, 'OBJECT', objectName[iGroup]
    ENDIF ELSE IF FILE_TEST(S3_file) THEN BEGIN
      UPDATE_HEADER_KEYWORD, S3_file, 'OBJECT', objectName[iGroup]
    ENDIF
  ENDFOR
ENDFOR

PRINT, 'Done!'
END