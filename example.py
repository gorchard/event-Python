# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:43:46 2016

@author: tslgmo
"""
import eventvision as ev

#read in some data
TD, EM = ev.read_aer('0000.val')

#show the TD events
ev.show_td(TD)

#extract a region of interest... note this will also edit the event struct 'TD'
TD2 = ev.extract_roi(TD, [100,100], [50,50])

#implement a refractory period... note this will also edit the event struct 'TD2'
TD3 = ev.impement_refraction(TD2, 0.03)

#perform some noise filtering... note this will also edit the event struct 'TD3'
TD4 = ev.filter_td(TD3, 0.03)

#show the resulting data
ev.show_td(TD4)

#write the filtered data in a format jAER can understand
ev.write2jAER(TD4, 'jAERdata.aedat')
    
    
#show the grayscale data
ev.show_em(EM)



#perform camera calibration
#first show the calibration pattern on the screen and make some recordings:
num_squares = 10
squareSize_mm = ev.present_checkerboard(num_squares)

#state where the recordings are what format they are in
image_directory = 'path_to_calibration_images'
image_format = '.bmp'

#using a scale is useful for visualization
scale = 4

#call the calibration function and follow the instructions provided
ret, mtx, dist, rvecs, tvecs = ev.auto_calibrate(num_squares, squareSize_mm, scale, image_directory, image_format)

