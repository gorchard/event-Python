# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 17:43:46 2016

@author: tslgmo
"""
import eventvision as ev

num_squares = 10
scale = 4
image_directory = 'C:\\Users\\tslgmo\\Desktop\\raw_recordings'
image_format = '.bmp'
squareSize_mm = ev.present_checkerboard(num_squares)
ret, mtx, dist, rvecs, tvecs = ev.auto_calibrate(num_squares, squareSize_mm, scale, image_directory, image_format)

TD, EM = ev.read_aer('0000.val')
ev.show_td(TD)


TD2 = ev.extract_roi(TD, [100,100], [50,50]);
