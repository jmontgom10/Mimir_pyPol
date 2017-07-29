; This script will make sure that every file in every group
; has a processed, astrometrically registered file.
;
; The astrometry from PPOL is sometimes no good.
; This script will loop through EVERY image of EVERY group, and do the following:
; 1) Test if the file has a step 3 processed FITS save
; 2) Test if such a file has astrometry
; 3) Show the astrometry to the user for evaluation
; 4) Re-solve the astrometry if it needs it
;
; Check below the PYPOL_REPAIR_ASTROMETRY procedure to set the PPOL directory
; for this PPOL project.

;******************************************************************************
; THE FOLLOWING CODE DEFINES FUNCTIONS TO BE USED IN THIS SCRIPT
;******************************************************************************
FUNCTION PYPOL_FIND_CENTROID, img, IS_SUBIMG = is_subimg, RANGE = range, WINDOW_ID = window_id, $
  CUTOUT_SIZE = cutout_size
  
  ;Check keywords, image size, and other initial requirements
  IF N_ELEMENTS(cutout_size) EQ 0 THEN cutout_size = 20
  ;  IF N_ELEMENTS(range) EQ 0 THEN STOP
  sz = SIZE(img, /DIMENSIONS)
  
  SKY, img, skyMode, skyNoise, /SILENT                         ;Estimate sky brightness and noise
  IF ~KEYWORD_SET(is_subimg) THEN BEGIN
    ;Begin by asking the user to estimate the bump (star or bulge) locaiton
    CURSOR, xGuess, yGuess, /DATA, /DOWN                         ;Click on the approximate location
    
    ;Cut out a subarray for a more precise positioning
    xOff   = (xGuess - (cutout_size - 1)) > 0
    xRt    = (xOff  + 2*cutout_size) < (sz[0] - 1)
    yOff   = (yGuess - (cutout_size - 1)) > 0
    yTop   = (yOff + 2*cutout_size)  < (sz[1] - 1)
    subImg = img[xOff:xRt, yOff:yTop]
  ENDIF ELSE BEGIN
    subImg = img
    xOff   = 0
    yOff   = 0
  ENDELSE
  
  retry  = 0
  done   = 0
  manual = 0
  WHILE ~done DO BEGIN
    TVIM, subImg, RANGE = [skyMode-skyNoise, MAX(subImg)]       ;Show the user a zoom-in of the point of interest
    
    ;Draw a pseudo button to toggle manual mode
    buttonXY = CONVERT_COORD([0,0,0.2,0.2,0], [0,0.08,0.08,0,0], /NORMAL, /TO_DATA)
    PLOTS, REFORM(buttonXY[0,*]), REFORM(buttonXY[1,*])
    
    ;    POLYFILL, [0, 0, 0.199, 0.199, 0], [0, 0.0799, 0.0799, 0,0], /NORMAL, COLOR=!P.BACKGROUND
    IF ~manual THEN BEGIN
      XYOUTS, 0.1, 0.03, 'Manual', /NORMAL, ALIGNMENT = 0.5
    ENDIF ELSE BEGIN
      XYOUTS, 0.1, 0.03, 'Centroid', /NORMAL, ALIGNMENT = 0.5
    ENDELSE
    
    IF retry EQ 0 THEN BEGIN
      XYOUTS, 0.5, 0.06, 'Click on the point of interest', /NORMAL, ALIGNMENT = 0.5
    ENDIF
    
    ;Click on the point of interest in the zoom-in
    IF (retry EQ 1) THEN BEGIN
      XYquery = CONVERT_COORD([Xquery], [Yquery], /NORMAL, /TO_DATA)
      xCen1   = XYquery[0]
      yCen1   = XYquery[1]
    ENDIF ELSE BEGIN
      ;If the user is not retrying, then use the cursor to get the guessed position
      CURSOR, xCen1, yCen1, /DATA, /DOWN
    ENDELSE
    
    IF ~manual THEN BEGIN
      ;Centroid the star using GCTRD
      GCNTRD, subImg, xCen1, yCen1, xCen, yCen, 3.0 ;, /SILENT      ;Compute the centroid about the clicked position
    ENDIF ELSE BEGIN
      xCen = xCen1
      yCen = yCen1
    ENDELSE
    
    OPLOT, [xcen,xcen], [0,40], LINESTYLE = 2, THICK = 2, COLOR = '0000FF'x ;Draw cross-hairs on the centroid position
    OPLOT, [0,40], [ycen,ycen], LINESTYLE = 2, THICK = 2, COLOR = '0000FF'x
    
    POLYFILL, [0.201,0.201,1,1,0.0201], [0,0.09,0.09,0,0], /NORMAL, COLOR=!P.BACKGROUND
    XYOUTS, 0.5, 0.06, 'Left click to recentroid', /NORMAL, ALIGNMENT = 0.5
    XYOUTS, 0.5, 0.03, 'Right click to accept centroid poisition', /NORMAL, ALIGNMENT = 0.5
    
    ;Ask the user what they want to do
    CURSOR, Xquery, Yquery, /NORMAL, /DOWN
    
    ;If they clicked in the "manual" button" then toogle the "manual" boolean variable
    IF (Xquery LT 0.2) AND (Yquery LT 0.12) THEN manual = ~manual
    
    ;If they left clicked, then just retry the centroid (or manual positioning)
    IF (!MOUSE.BUTTON EQ 1) THEN BEGIN
      retry = 1
    ENDIF
    
    ;If they right clicked, then proceed to return these values
    IF (!MOUSE.BUTTON EQ 4) THEN BEGIN
      retry = 0
      done  = 1
    ENDIF
    
  ENDWHILE
  
  xPoint = Xcen + xOff                                               ;Recompute the x-position
  yPoint = Ycen + yOff                                               ;Recompute the y-position
  
  RETURN, [xPoint, yPoint]
  
END


PRO PYPOL_REPAIR_ASTROMETRY, thisFile, thisImg, thisHead, ASTROSTARS = astroStars, PPOL_DIR = PPOL_dir
  ; This procedure will perform the necessary astrometry repair operations
  ; to fill in the blanks for PPOL STEP 3
  PRINT, 'Repairing astrometry for file ', thisFile
  
  ; Check if a STEP 2 file exists
  thisBadFile = PPOL_dir + PATH_SEP() + $           ; Grab the S2 filename for this file
    'S2_Ski_Jump_Fixes' + PATH_SEP() + FILE_BASENAME(thisFile)
  IF ~FILE_TEST(thisBadFile) THEN BEGIN
    ; If no STEP 2 file exists, then simply use the BDP file
    thisBadFile = thisFile
  ENDIF

  thisImg = READFITS(thisBadFile, thisHead)
  sz = SIZE(thisImg, /DIMENSIONS)
  
  ; Apply a quick modeling procedure
  fixedImg = MODEL_BAD_PIXELS(thisImg)                              ;Fix bad pixels
  hist     = SXPAR(thisHead, "HISTORY")                             ;Get the history info
  SXDELPAR, thisHead,'HISTORY'                                      ;delete any previous history entries
  telRA    = STRSPLIT(SXPAR(thisHead, 'TELRA'), ':', /EXTRACT)      ;Store the telescope RA pointing
  telRA    = 15D*TEN(telRA[0], telRA[1], telRA[2])                  ;Convert pointing to float (deg)
  telDec   = STRSPLIT(SXPAR(thisHead, 'TELDEC'), ':', /EXTRACT)     ;Store the telescope Dec pointing
  telDec   = TEN(telDec[0], telDec[1], telDec[2])                   ;Convert pointing to float (deg)
  
  ; Estimate the astrometry
  sz      = SIZE(thisImg, /DIMENSIONS)
  cdGuess = [[-0.000160833, 0],$                                    ;Store astrometry guess (0.579 arcsec/pix)
    [0,  0.000160833]]
  MAKE_ASTR, astrGuess, CRPIX = 0.5*sz, CRVAL = [telRA, telDec], $  ;Initalize astrometry structure
    CD = cdGuess
    
  AD2XY, astroStars.RAJ2000, astroStars.DEJ2000, $                  ;Position astrometry stars using astrometry guess
    astrGuess, xGuess, yGuess
    
  ;Subtrcat 1 to account for indexing differences
  xGuess -= 1
  yGuess -= 1
  
  useStar = (xGuess GT 80) AND (xGuess LT (sz[0] - 81)) $           ;Only use stars more than 80 pixels from image edge
    AND (yGuess GT 80) AND (yGuess LT (sz[1] - 81))
  useInds = WHERE(useStar, numUse)                                  ;Get the indices of the usable stars
  
  ; Grab the STATS for this image
  SKY, fixedImg, skyMode, skySig, /silent
  
  ; Show the image to the user
  TVIM, fixedImg, RANGE = skyMode + [-3, +10]*skySig
  
  XYOUTS, 0.5, 0.98, FILE_BASENAME(thisFile), /NORMAL, ALIGNMENT = 0.5
  XYOUTS, 0.5, 0.02, 'Click on highlighted star', /NORMAL, ALIGNMENT = 0.5
  
  OPLOT, xGuess, yGuess, PSYM=4, COLOR=255L                         ;Overplot the astrometry stars
  OPLOT, [xGuess[useInds[0]]], [yGuess[useInds[0]]], $              ;Mark the brightest usable star
    PSYM=6, COLOR=255L*255L, SYMSIZE = 2
  
  XYcen = PYPOL_FIND_CENTROID(fixedImg, RANGE = skyMode + [-3, +10]*skySig)

  shiftX  = XYcen[0] - xGuess[useInds[0]]                           ;Compute x-offset correction
  shiftY  = XYcen[1] - yGuess[useInds[0]]                           ;Compute y-offset correction
  xGuess += shiftX                                                  ;Update estimated star x-positions
  yGuess += shiftY                                                  ;Update estimated star y-positions
  useStar = (xGuess GT 20) AND (xGuess LT (sz[0] - 21)) $           ;Update star usage flags
    AND (yGuess GT 20) AND (yGuess LT (sz[1] - 21)) $               ;based on new star (x,y) positions
    AND (astroStars.HMAG LT 14) AND (astroStars.KMAG LT 14)         ;and based on magnitudes

  ; Locate the indices of the usable stars
  useInds = WHERE(useStar, numUse)

  ; Only keep the brightest 25 stars (that's all you'll need!)
  IF numUse GT 50 THEN BEGIN
    useInds = useInds[0:24]
    numUse = N_ELEMENTS(useInds)
  ENDIF

  IF ~(numUse GT 0) THEN STOP ELSE BEGIN                            ;If there are no usable stars, then stop
    xGuess      = xGuess[useInds]                                    ;Otherwise cull the star position estimates
    yGuess      = yGuess[useInds]                                    ;to only include the usable stars
    astroStars1 = astroStars[useInds]
  ENDELSE
  
  nStars = numUse                                                   ;Count the number of stars to find
  FWHMs  = FLTARR(nStars)                                           ;Array to store FWHM of each star
  
  xStars = xGuess                                                   ;Now that the stars have been "centroided"
  yStars = yGuess                                                   ;they are no longer "guesses"
  
  FOR k = 0, nStars - 1 DO BEGIN                                    ;Loop through each star and find its gcentrd value
    xOff     = (xGuess[k] - 20) > 0
    xRt      = (xOff  + 40) < (sz[0] - 1)
    yOff     = (yGuess[k] - 20) > 0
    yTop     = (yOff + 40)  < (sz[1] - 1)
    subArray = fixedImg[xOff:xRt, yOff:yTop]                        ;Cut out a subarry for fine tuning star position
    result   = GAUSS2DFIT(subArray, A, /TILT)                       ;Gaussian fit the star
    inArray  = (A[4] GT 5) AND (A[4] LT 35) $                       ;If the fit is located in the array
      AND (A[5] GT 5) AND (A[5] LT 35)
    okShape  = (A[2] GT 0.8) AND (A[2] LT 5) $                      ;and if its gaussian width is reasonable (not a hot pixel)
      AND (A[3] GT 0.8) AND (A[3] LT 5)
      
    IF inArray AND okShape THEN method = 'gaussianGuess' $          ;Define which method should be used first
    ELSE method = 'manualAssist'
    
    SWITCH method OF
    
      ;This method simply performs a gaussian fit and computes a gaussian centroid at the expected star position.
      ;If the two methods agree, then the we BREAK out of the SWITCH.
      ;If the two methods disagree or there are other problems, then we proceed with a user assisted centroid method.
      'gaussianGuess': BEGIN
        ;If everything is in order, then simply proceed with centroid
        FWHMs[k] = SQRT(ABS(A[2]*A[3]))*2.355                       ;Compute the FWHM for this star
        GCNTRD, subArray, A[4], A[5], xcen, ycen,  FWHMs[k]         ;Centroid this star (using estimated FWHM)
        methodDifference = SQRT((xCen - A[4])^2 + (yCen - A[5])^2)  ;Compute difference between the two locating methods
        IF (methodDifference LE 1E) AND FINITE(methodDifference) $  ;If the two methods have a consensus,
          THEN BREAK                                                ;then we have our answer! (exit SWTICH blocks)
      END
      
      ;This method will display the subarray to the user and have them help locate the star center
      'manualAssist': BEGIN
        XYcen = PYPOL_FIND_CENTROID(subArray, /IS_SUBIMG)
        xcen  = XYcen[0]
        ycen  = XYcen[1]
      END
    ENDSWITCH
    
    xStars[k] = xOff + xcen-1                                         ;Update the star x-position
    yStars[k] = yOff + ycen-1                                         ;Update the star y-position
  ENDFOR
  
  TVIM, fixedImg, RANGE = skyMode + [-3, +10]*skySig
  
  XYOUTS, 0.5, 0.98, FILE_BASENAME(thisFile), /NORMAL, ALIGNMENT = 0.5
  XYOUTS, 0.5, 0.02, 'Left click (approve) | Right click (repair)', /NORMAL, ALIGNMENT = 0.5
  
  OPLOT, xStars, yStars, PSYM=4, COLOR=255L                         ;Overplot the astrometry stars
  
  CURSOR, xJunk, yJunk, /DATA, /DOWN                         ;Click on the approximate location
  IF !MOUSE.BUTTON EQ 1 THEN BEGIN
    ; The left button was clicked
    PRINT, 'Astrometry approved for ', thisFile
  ENDIF ELSE IF !MOUSE.BUTTON EQ 4 THEN BEGIN
    PRINT, 'Uh Oh... you still did not like the astrometry?'
    PRINT, 'That is a bad sign.'
    STOP
  ENDIF

  ; Now solve the astrometry...
  astr = JM_SOLVE_ASTRO(astroStars1.RAJ2000, astroStars1.DEJ2000, xStars, yStars, $
    NAXIS1 = sz[0], NAXIS2 = sz[1])
  crpix = [511, 512]
  XY2AD, crpix[0], crpix[1], astr, crval1, crval2
  astr.crpix = (crpix + 1)                                        ;FITS convention is offset 1 pixel from IDL
  astr.crval = [crval1, crval2]                                   ;Store the updated reference pixel values
  PUTAST, thisHead, astr, EQUINOX = 2000                        ;Update the header with the new astrometry
  
  ; Make sure the output image is the FIXED image
  thisImg = fixedImg
  
END


;******************************************************************************
; THIS IS WHERE THE USER CAN DEFINE THE SPECIFICS OF THIS PPOL_PROJECT
;******************************************************************************
; Define the PPOL directory
PPOL_dir  = 'C:\Users\Jordan\FITS_data\Mimir_data\PPOL_Reduced\201611'


;******************************************************************************
; FINALLY, WE EXECUTE A SCRIPT TO ACTUALLY FIX THE ASTROMETRY
;******************************************************************************
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

; Loop through each group and do astrometry tests
FOR iGroup = 0, ((*G_PTR).N_GROUPS - 1) DO BEGIN
  thisGroupCount    = (*G_PTR).GROUP_NUMBERS[iGroup]  ; Grab the number of files in this group
  
  ; Acces the 2MASS catalog from online resources
  ; First grab the approximate image coordinates
  tmpHead  = HEADFITS((*G_PTR).GROUP_IMAGES[iGroup,0]) ; Grab the header from the first file
  thisBand = STRTRIM(SXPAR(tmpHead, 'FILTNME2'), 2)
  telRA    = STRSPLIT(SXPAR(tmpHead, 'TELRA'), ':', /EXTRACT)      ;Store the telescope RA pointing
  telRA    = 15D*TEN(telRA[0], telRA[1], telRA[2])                 ;Convert pointing to float (deg)
  telDec   = STRSPLIT(SXPAR(tmpHead, 'TELDEC'), ':', /EXTRACT)     ;Store the telescope Dec pointing
  telDec   = TEN(telDec[0], telDec[1], telDec[2])                  ;Convert pointing to float (deg)
  
  ;Grab all the good stars within 12-arcmin of the median pointing.
  ;Retrieve the 2MASS star info
  ;this_mirror = ['CfA', 'UK', 'CDS', 'CA', 'Jp']
  this_mirror = ['UK', 'CDS', 'CA', 'Jp']
  ;
  nloop = 0L
  WHILE nloop LT 4320L DO BEGIN                       ; 1 day at 20 sec per try
    ;
    vizier_flag = 0
    FOR imirror = 0, N_ELEMENTS(this_mirror)-1 DO BEGIN
      IF(vizier_flag EQ 0) THEN BEGIN
        astroStars = mp_QueryVizier('2MASS-PSC', [telRA, telDec], [16,16], $
          MIRROR=this_mirror[imirror], constraint='Qflg==AAA')
        test = size(astroStars, /TYPE)

        IF (test EQ 8) THEN BEGIN
          IF TOTAL(astroStars.DEJ2000 GT 90) EQ 0 THEN vizier_flag = 1
        ENDIF
      ENDIF
    ENDFOR
    
    IF(vizier_flag EQ 0) THEN BEGIN
      astroStars = 99   ;No Vizier servers available
      ;
      ; if no Vizier servers, wait 20s and retry
      ;
      PRINT, 'No Vizier Servers at ',SYSTIME(),' waiting 20s and retrying'
      WAIT, 20
      nloop++
    ENDIF ELSE nloop = 4321L                                ;Force the loop to close
  ENDWHILE

  ; Sort the astroStars array by the magnitude at THIS band
  IF thisBand EQ 'H' THEN BEGIN
    sortInds = SORT(astroStars.KMAG)
  ENDIF ELSE IF thisBand EQ 'Ks' THEN BEGIN
    sortInds = SORT(astroStars.HMAG)
  ENDIF
  astroStars = astroStars[sortInds]

  ; Now that the Vizier catalog info has been downloaded,
  ; display the astrometry to the user for evaluation
  FOR iFile = 0, (thisGroupCount - 1) DO BEGIN        ; Loop through all of the files in this group
    thisFile    = (*G_PTR).GROUP_IMAGES[iGroup, iFile]; Grab the BDP filename for this file
    thisS3file  = PPOL_dir + PATH_SEP() + $           ; Grab the S3 filename for this file
      'S3_Astrometry' + PATH_SEP() + FILE_BASENAME(thisFile)

    ; If an S3 file exists, then read it in and have the user check it's astrometry
    IF FILE_TEST(thisS3file) THEN BEGIN
      ; The S3 file exists, so let's check it out.
      thisImg = READFITS(thisS3file, thisHead)
      sz      = SIZE(thisImg, /DIMENSIONS)
      
      ; Grab the STATS for this image
      SKY, thisImg, skyMode, skySig, /silent
      
      ; Show the image to the user
      WINDOW, 0, xs = 800, ys  = 800
      TVIM, thisImg, RANGE = skyMode + [-3, +10]*skySig
      
      ; Overplot the 2MASS stars
      EXTAST, thisHead, astr
      AD2XY, astroStars.RAJ2000, astroStars.DEJ2000, $                  ;Position astrometry stars using astrometry guess
        astr, xStars, yStars
      
      ;Subtrcat 1 to account for indexing differences
      xStars -= 1
      yStars -= 1
      
      useStar = (xStars GT 80) AND (xStars LT (sz[0] - 81)) $           ;Only use stars more than 80 pixels from image edge
        AND (yStars GT 80) AND (yStars LT (sz[1] - 81))
      useInds = WHERE(useStar, numUse)                                  ;Get the indices of the usable stars
      
      ; Check that at least three star is on the image
      ; (otherwise there's probably an error)
      IF numUse LT 3 THEN BEGIN
        PRINT, 'There are no stars in this image!'
        PRINT, 'What went wrong?'
        STOP 
      ENDIF

      OPLOT, xStars, yStars, PSYM=4, COLOR=255L                         ;Overplot the astrometry stars
;      OPLOT, [xStars[useInds[0]]], [yStars[useInds[0]]], $              ;Mark the brightest usable star
;        PSYM=6, COLOR=255L*255L, SYMSIZE = 2

      
      XYOUTS, 0.5, 0.98, FILE_BASENAME(thisFile), /NORMAL, ALIGNMENT = 0.5
      XYOUTS, 0.5, 0.02, 'Left click (approve) | Right click (repair)', /NORMAL, ALIGNMENT = 0.5

      CURSOR, xJunk, yJunk, /DATA, /DOWN                         ;Click on the approximate location
      IF !MOUSE.BUTTON EQ 1 THEN BEGIN
        ; The left button was clicked
        PRINT, 'Astrometry approved for ', thisS3file
        ERASE
      ENDIF ELSE IF !MOUSE.BUTTON EQ 4 THEN BEGIN
        ; The right mouse button was clicked
        PYPOL_REPAIR_ASTROMETRY, thisFile, thisImg, thisHead, $
          ASTROSTARS = astroStars, PPOL_DIR = PPOL_dir
        
        ; Now that the astrometry has been written to the header,
        ; write the FITS image and header to the step 3 directory.
        WRITEFITS, thisS3file, thisImg, thisHead
      ENDIF
      
    ENDIF ELSE BEGIN
      ; In this case, no astrometry exists, so we will need perform a repair
      PRINT, 'No PPOL Step 3 file found... beginning astrometry repair.'
      PYPOL_REPAIR_ASTROMETRY, thisFile, thisImg, thisHead, $
        ASTROSTARS = astroStars, PPOL_DIR = PPOL_dir
      
      ; Now that the astrometry has been written to the header,
      ; write the FITS image and header to the step 3 directory.
      WRITEFITS, thisS3file, thisImg, thisHead

    ENDELSE
  ENDFOR
ENDFOR

PRINT, 'Done!'

END