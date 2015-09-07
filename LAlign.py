# -*- coding: utf-8 -*-
"""
Created on Sun Sep 06 14:23:23 2015

@author: anaiman
"""

import os
import cv2

#(F_SCALE, W_TARGET, H_TARGET) = (50.0, 120.0, 400.0)
#IN_DIR = '.\side_in_small'
#OUT_DIR = '.\side_out_small'
(F_SCALE, W_TARGET, H_TARGET) = (400.0, 1000.0, 3400.0)
IN_DIR = '.\side_in'
OUT_DIR = '.\side_out'

def align_images(in_dir):
    
    # Load an appropriate trained cascade classifier
    face_cascade = cv2.CascadeClassifier('.\haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print "Problem loading face cascade"
        return
    
    # Process each image in directory
    for f in os.listdir(in_dir):
        fpath = os.path.join(in_dir, f)
        print fpath
        img = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray)        
        print faces
        
#        for (x,y,w,h) in faces:
#            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 1)
#            roi_gray = gray[y:y+h, x:x+w]
#            roi_color = img[y:y+h, x:x+w]
#            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 2)
#            print eyes
#            for (ex,ey,ew,eh) in eyes:
#                cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1) 
        
        #small_img = cv2.resize(img, (0,0), fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
        
#        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
#        cv2.imshow('image', img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        
        # If there's more than one face detected, use the one with closest to expected size
        if len(faces) > 0:
            diff = F_SCALE
            for (fx,fy,fw,fh) in faces:
                if abs(fw - F_SCALE) < diff:
                    (x,y,w,h) = (fx,fy,fw,fh)
                    diff = abs(fw - F_SCALE)
        else:
            print "No faces detected"
            continue

        print x, y, w, h
        scale = F_SCALE/w
        # First, crop the image to a predetermined box size
        width = W_TARGET/scale
        height = H_TARGET/scale
        xmin = int(x - w/2)
        xmax = int(xmin + width)
        ymin = int(y - h/4)
        ymax = int(ymin + height)
        
        print ymin, ymax, xmin, xmax
        crop_img = img[ymin:ymax, xmin:xmax]
        
        # Then, scale the image based on the detected face size
        new_img = cv2.resize(crop_img, (0,0), fx=scale, fy=scale)
        
#        cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
#        cv2.imshow('image', new_img)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()
        
        outpath = os.path.join(OUT_DIR, f)
        cv2.imwrite(outpath, new_img)

if __name__ == "__main__":
    images_in_dir = IN_DIR
    
    align_images(images_in_dir)