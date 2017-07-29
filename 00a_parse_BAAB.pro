; This script parses the information in the PPOL save file. It's principle
; concern is to divide the ABBA dithering into the constituent file lists and
; produce a new save file with each group divided into two separate groups
; consisting of only A or B images.
; 

; Define the PPOL directory
PPOL_dir  = 'C:\Users\Jordan\FITS_data\Mimir_data\PPOL_Reduced\201611'

; This will define the relative path to the group summary file
summaryFile = PPOL_dir + PATH_SEP() + 'S1_Image_Groups_and_Meta_Groups' + PATH_SEP() + 'Group_Summary.sav'

; Test if this file exists
IF FILE_TEST(summaryFile) THEN BEGIN
  ; Generate a time stamp to append to the summary file for a backup naming.
  time_str = TIMESTAMP(OFFSET = -5)
  time1Arr = STRSPLIT(time_str, '-', /EXTRACT)
  time2Arr = STRSPLIT(time1Arr[2], ':', /EXTRACT)
  time3Arr = STRSPLIT(time2Arr[2], '.', /EXTRACT)
  time_str = STRJOIN([time1Arr[0:1], time2Arr[0:1], time3Arr[0]], '')

  ; Generate a backup summary file name
  backupFile = PPOL_dir + PATH_SEP() + 'S1_Image_Groups_and_Meta_Groups' + $
    PATH_SEP() + 'Group_Summary_' + time_str + '.sav

  ; Copy the original file as a backup save.
  FILE_COPY, summaryFile, backupFile


  ; Restoring the summaryFile will created a pointer named "G_PTR" which points
  ; to the group summary structure.
  RESTORE, summaryFile

  ;If the PPOL summary was successfully read, then update it and save to disk
  IF N_ELEMENTS(G_PTR) GT 0 THEN BEGIN
    ; Begin by making a copy of the group summary
    G_PTR_out = PTR_NEW(*G_PTR)                           ; Make a new pointer for storing the new group summary
    nxny      = SIZE((*G_PTR).GROUP_IMAGE_FLAGS, /DIMENSIONS); Grab the shape of the group summary array
    HWPlist   = [34,    4556,  2261,  6784,  9011,  13534, 11306, 15761, $
                 18056, 22511, 20284, 24806, 27034, 31556, 29261, 33784]

    ; Loop through the group summary and build arrays for parsing ABBA groups.
    FOR iGroup = 0, ((*G_PTR).N_GROUPS - 1) DO BEGIN
      PRINT, 'Parsing group ', (*G_PTR).GROUP_NAMES[iGroup]
      ; For each of each group file, we will need two pieces of information
      ; 1) The FILENUMBER (essentially the date plus the nightly file number)
      ; 2) The HWP_ANGLE (the rotation of the HWP)
      ; Using this information, we can parse which files are A vs. B
      thisGroupCount    = (*G_PTR).GROUP_NUMBERS[iGroup]  ; Grab the number of files in this group
      thisGroupFileNums = LON64ARR(thisGroupCount)        ; Initalize an array to store file numbers for this group
      thisGroupHWPs     = LONARR(thisGroupCount)          ; Initalize an array to store HWP numbers
      thisGroupRAs      = DBLARR(thisGroupCount)          ; Initalize an array to store RAs
      thisGroupDecs     = DBLARR(thisGroupCount)          ; Initalize an array to store Decs
      thisGroupABBAs    = STRARR(thisGroupCount)          ; Initalize an array to store The ABBA values
      FOR iFile = 0, (thisGroupCount - 1) DO BEGIN        ; Loop through all of the files in this group
        thisFile    = (*G_PTR).GROUP_IMAGES[iGroup, iFile]; Grab the filename for this file
        
        ; Parse the file number for this file
        thisFileNum = FILE_BASENAME(thisFile)             ; Isolate the file number alone.
        thisFileNum = (STRSPLIT(thisFileNum, '_', /EXTRACT))[0]
        thisFileNum = STRJOIN(STRSPLIT(thisFileNum, '.', /EXTRACT), '')
        thisGroupFileNums[iFile] = LONG64(thisFileNum)    ; Save that file number in the array

        ; Parse the HWP for this file
        thisHead = HEADFITS(thisFile)                     ; Read in the header for this image
        thisHWP  = ROUND(100*SXPAR(thisHead, 'HWP_ANG'))  ; Get the HWP number
        thisGroupHWPs[iFile] = thisHWP                    ; Store the parsed HWP angle
        
        ; Parse the pointing for this file
        telRA    = STRSPLIT(SXPAR(thisHead, 'TELRA'), ':', /EXTRACT)      ;Store the telescope RA pointing
        telRA    = 15D*TEN(telRA[0], telRA[1], telRA[2])                  ;Convert pointing to float (deg)
        telDec   = STRSPLIT(SXPAR(thisHead, 'TELDEC'), ':', /EXTRACT)     ;Store the telescope Dec pointing
        telDec   = TEN(telDec[0], telDec[1], telDec[2])                   ;Convert pointing to float (deg)
        thisGroupRAs[iFile]  = telRA                                      ; Store the parsed RA
        thisGroupDecs[iFile] = telDec                                     ; Store the parsed Dec
        
        comments = SXPAR(thisHead, 'COMMENT')
        FOREACH comment, comments, iCom DO BEGIN
          IF STRMID(comment, 0, 3) EQ 'HWP' THEN BEGIN
            startInd = STREGEX(comment, '- *')
            endInd   = STREGEX(comment, ' * posn')
            thisGroupABBAs[iFile] = STRMID(comment, startInd + 2, endInd - startInd - 2)
          ENDIF
        ENDFOREACH
      ENDFOR
      
      ; Compute the incremental step for each image in the sequence
      numIncr = thisGroupFileNums - shift(thisGroupFileNums, 1)
      numIncr[0] = 1
      
      ; Check if the mean increment is about 2, and skip if it is...
      meanIncr = (10.0*MEAN(numIncr))/10.0
      skipGroup = 0
      IF (meanIncr GE 1.85) AND (meanIncr LE 2.15) THEN BEGIN
        PRINT, (*G_PTR).group_names[iGroup] + ' appears to already have been parsed'
        skipGroup = 1
      ENDIF
      
      IF skipGroup EQ 1 THEN CONTINUE

      ; Find where the HWP changes
      HWPshifts = thisGroupHWPs NE SHIFT(thisGroupHWPs, 1)
      HWPshifts[0] = 0B
      
      IF TOTAL(HWPshifts) LT 0.5*thisGroupCount THEN BEGIN
        ;****************************************************
        ; This group has  the 16(BAAB) dither type.
        ;****************************************************
        PRINT, 'Dither type 16(BAAB)'
        ; Find places where the HWP change corresponds to a numIncr of 1
        ABBArestart = HWPshifts AND (numIncr EQ 1)
        
        ; Check for the index of these ABBA restart points
        ABBArestartInd = WHERE(ABBArestart, restartCount)
        IF restartCount GT 0 THEN BEGIN
          ; Find the index of the first restart
          firstRestartInd = MIN(ABBArestartInd)
          ; Find the amount of shift needed to coincide ABBAinds with ABBArestart
          numSkips  = ROUND(TOTAL(numIncr[0:firstRestartInd]) - 1) - firstRestartInd
          ABBAshift = (64 - (firstRestartInd + numSkips)) MOD 4
          ABBAinds  = (thisGroupFileNums - thisGroupFileNums[0] + ABBAshift) MOD 4
        ENDIF ELSE BEGIN
          PRINT, 'The HWP shifts are not well mapped, so a solution is not possible.'
          STOP
        ENDELSE
        ; Setup the dither pattern array
        ; ABBAarr   = ['A','B','B','A']
        ABBAarr   = ['B','A','A','B'] 
               
        ; Grab the ABBA values for each file
        thisGroupABBAs2 = ABBAarr[ABBAinds]

        ; Test if the BAAB arrangement from the comments and from HWP rotation rate agree
        IF total(thisGroupABBAs2 NE thisGroupABBAs) GT 0 THEN STOP
        
        ; Parse the indices for A images and B images
        Ainds = WHERE(thisGroupABBAs EQ 'A', AimgCount)
        Binds = WHERE(thisGroupABBAs EQ 'B', BimgCount)
      ENDIF ELSE BEGIN
        ;****************************************************
        ; This group has  the (16A, 16B, 16B, 16A) dither type.
        ;****************************************************
        ; This kind of dither type should not be allowed (for now)
        CONTINUE

        PRINT, 'Dither type (16A, 16B, 16B, 16A)'
        
        ; Setup the group dither pattern array (16*A, 16*B, 16*B, 16*A)
        As = replicate('A', 16)
        Bs = replicate('B', 16)
        ABBAarr  = [Bs, As, As, Bs]
        
        ; Figure out if any of the first images were dropped
        HWPdiff     = ABS(HWPlist - thisGroupHWPs[0])
        firstHWPind = WHERE(HWPdiff EQ MIN(HWPdiff))
        ABBAinds    = (thisGroupFileNums - thisGroupFileNums[0] + firstHWPInd[0])
        
        ; Grab the ABBA values for each file
        thisGroupABBAs = ABBAarr[ABBAinds]
        
        ; Parse the indices for A images and B images
        Ainds = WHERE(thisGroupABBAs EQ 'A', AimgCount)
        Binds = WHERE(thisGroupABBAs EQ 'B', BimgCount)
      ENDELSE
      
      ; Testing that the comments and HWP rotation rate agree is a better double check than RA/Dec.
;      ; Double check that the pointing for each group is correct.
;      outliersPresent = 1B
;      WHILE outLiersPresent DO BEGIN
;        ; Compute the median pointings for A and B dithers
;        A_medRA  = MEDIAN(thisGroupRAs[Ainds])
;        A_medDec = MEDIAN(thisGroupDecs[Ainds])
;        B_medRA  = MEDIAN(thisGroupRAs[Binds])
;        B_medDec = MEDIAN(thisGroupDecs[Binds])
;        
;        ; Compute the (RA, Dec) offsets from the median pointings
;        A_delRA  = thisGroupRAs[Ainds] - A_medRA
;        A_delDec = thisGroupDecs[Ainds] - A_medDec
;        B_delRA  = thisGroupRAs[Binds] - B_medRA
;        B_delDec = thisGroupDecs[Binds] - B_medDec
;        
;        ; Search for outliers in either RA **OR** Dec
;        ; (more than 1 arcmin off median pointing).
;        A_RA_out  = ABS(A_delRA) GT 1E/60E
;        A_Dec_out = ABS(A_delDec) GT 1E/60E
;        B_RA_out  = ABS(B_delRA) GT 1E/60E
;        B_Dec_out = ABS(B_delDec) GT 1E/60E
;        
;        ; Set a flag to determine if there are still any outliers
;        outliersPresent = ((TOTAL(A_RA_out OR A_Dec_out) + TOTAL(B_RA_out OR B_Dec_out)) GT 0)
;               
;        ; If there **DO** still seem to be outliers present,
;        ; then swap offending images between groups.
;        IF outliersPresent THEN BEGIN
;          PRINT, 'Repairing pointing outliers'
;
;          ; First identify offending images from each group
;          A_out = (A_RA_out OR A_Dec_out)
;          B_out = (B_RA_out OR B_Dec_out)
;          
;          ; Now identify which of the Aind and Binds need to be swapped
;          IF TOTAL(A_out) GT 0 THEN BEGIN
;            AswapInds = Ainds[WHERE(A_out)]
;            AkeepInds = Ainds[WHERE(~A_out)]
;          ENDIF
;          IF TOTAL(B_out) GT 0 THEN BEGIN
;            BswapInds = Binds[WHERE(B_out)]
;            BkeepInds = Binds[WHERE(~B_out)]
;          ENDIF
;          
;          ; Reconstruct the Ainds and Binds arrays
;          Ainds = [AkeepInds, BswapInds]
;          Binds = [BkeepInds, AswapInds]
;          
;          ; Sort the newly constructed Ainds and Binds arrays
;          AsortArr = SORT(Ainds)
;          Ainds    = Ainds[AsortArr]
;          BsortArr = SORT(Binds)
;          Binds    = Binds[BsortArr]
;          
;          ; Count the number of images in each group
;          AimgCount = N_ELEMENTS(Ainds)
;          BimgCount = N_ELEMENTS(Binds)
;        ENDIF
;      ENDWHILE
      
      IF (AimgCount GT 0) AND (BimgCount GT 0) THEN BEGIN
        ; If there are A images and B images to parse up,
        ; then re-organize the (*G_PTR) structure.
        ; .....
        ; Shift the Group names,files, and flags over by one to make room for the new group parsing
        iiGroup = 2*iGroup
        (*G_PTR_out).GROUP_NAMES[iiGroup+2:*] = (*G_PTR_out).GROUP_NAMES[iiGroup+1:998]
        (*G_PTR_out).GROUP_IMAGES[iiGroup+2:*,*] = (*G_PTR_out).GROUP_IMAGES[iiGroup+1:998,*]
        (*G_PTR_out).GROUP_IMAGE_FLAGS[iiGroup+2:*,*] = (*G_PTR_out).GROUP_IMAGE_FLAGS[iiGroup+1:998,*]
        (*G_PTR_out).GROUP_LLQ_FLAGS[iiGroup+2:*,*] = (*G_PTR_out).GROUP_LLQ_FLAGS[iiGroup+1:998,*]
        (*G_PTR_out).GROUP_NUMBERS[iiGroup+2:*] = (*G_PTR_out).GROUP_NUMBERS[iiGroup+1:998]
        (*G_PTR_out).GROUP_STEP[iiGroup+2:*] = (*G_PTR_out).GROUP_STEP[iiGroup+1:998]
        
        ; Now break up the original gruop into two groups
        ; Start by making two copies of the "group-name" with an A and B appended
        (*G_PTR_out).GROUP_NAMES[iiGroup+1] = (*G_PTR).GROUP_NAMES[iGroup] + '_B'
        (*G_PTR_out).GROUP_NAMES[iiGroup] = (*G_PTR).GROUP_NAMES[iGroup] + '_A'
        
        ; Next Divy up the group images (GROUP_IMAGES - STRING)
        tmpArr = STRARR(1,512)
        tmpArr[0, 0:BimgCount-1] = (*G_PTR).GROUP_IMAGES[iGroup, Binds]
        (*G_PTR_out).GROUP_IMAGES[iiGroup+1,*] = TEMPORARY(tmpArr)
        tmpArr = STRARR(1,512)
        tmpArr[0, 0:AimgCount-1] = (*G_PTR).GROUP_IMAGES[iGroup, Ainds]
        (*G_PTR_out).GROUP_IMAGES[iiGroup,*] = TEMPORARY(tmpArr)
        
        ; GROUP_IMAGE_FLAGS - INT
        tmpArr = INTARR(1,512)
        tmpArr[0, 0:BimgCount-1] = (*G_PTR).GROUP_IMAGE_FLAGS[iGroup, Binds]
        (*G_PTR_out).GROUP_IMAGE_FLAGS[iiGroup+1,*] = TEMPORARY(tmpArr)
        tmpArr = INTARR(1,512)
        tmpArr[0, 0:AimgCount-1] = (*G_PTR).GROUP_IMAGE_FLAGS[iGroup, Ainds]
        (*G_PTR_out).GROUP_IMAGE_FLAGS[iiGroup,*] = TEMPORARY(tmpArr)
        
        ; GROUP_LLQ_FLAGS - BYTE
        tmpArr = INTARR(1,512)
        tmpArr[0, 0:BimgCount-1] = (*G_PTR).GROUP_LLQ_FLAGS[iGroup, Binds]
        (*G_PTR_out).GROUP_LLQ_FLAGS[iiGroup+1,*] = TEMPORARY(tmpArr)
        tmpArr = INTARR(1,512)
        tmpArr[0, 0:AimgCount-1] = (*G_PTR).GROUP_LLQ_FLAGS[iGroup, Ainds]
        (*G_PTR_out).GROUP_LLQ_FLAGS[iiGroup,*] = TEMPORARY(tmpArr)
        
        ; GROUP_NUMBERS - INT
        (*G_PTR_out).GROUP_NUMBERS[iiGroup+1] = BimgCount
        (*G_PTR_out).GROUP_NUMBERS[iiGroup] = AimgCount
        
        ; GROUP_STEP - INT
        (*G_PTR_out).GROUP_STEP[iiGroup+1] = (*G_PTR).GROUP_STEP[iGroup]
        (*G_PTR_out).GROUP_STEP[iiGroup] = (*G_PTR).GROUP_STEP[iGroup]
        
        ; Finally increase the number of groups to account for the newly generated group.
        (*G_PTR_out).N_GROUPS = (*G_PTR_out).N_GROUPS + 1     ; Increase the number of groups
      ENDIF ELSE BEGIN
        PRINT, 'That is weird. The are only A images or B images... you sure this is right?'
        STOP
      ENDELSE
    ENDFOR

    ; Rename the variable, and then save to disk
    PTR_FREE, G_PTR
    G_PTR = G_PTR_OUT
    SAVE, G_PTR, DESCRIPTION="PPOL Group Summary File ", FILENAME = summaryFile
    
    ; Free up the heap space from the pointers
    PTR_FREE, G_PTR_OUT
    
  ENDIF ELSE BEGIN
    PRINT, 'Group summary file does not contain "G_PTR" variable'
  ENDELSE
ENDIF ELSE BEGIN
  PRINT, 'Group summary file not found'
ENDELSE

PRINT, 'Done!'

END