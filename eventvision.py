# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
from win32api import GetSystemMetrics
import numpy as np
import cv2
import glob

class Events:
    def __init__(self):
        self.x = None
        self.y = None
        self.p = None
        self.ts = None 

def present_checkerboard(num_squares):
    screen_width_pixels = GetSystemMetrics(0)
    screen_height_pixels = GetSystemMetrics(1)
    
    #fixed parameters of the setup
    figure_borderSize = 30; #leave space of 100 pixels on each side of the axes for the figure controls etc
    # image_borderSize = 10; %within the image, create a border of size 10 pixels to ensure contrast with the outside rectangles
    
    #How big is each rectangle in units of pixels?
    Screen_size_pixels = np.array([screen_width_pixels, screen_height_pixels])
    Screen_size_mm = 0.00254*Screen_size_pixels/96
    squareSize_pixels = int(min(Screen_size_pixels - 2*figure_borderSize)/(num_squares+2))
    
    image_borderSize = np.array([1,2])
    image_borderSize[0] = (Screen_size_pixels[0] - figure_borderSize*2 - squareSize_pixels*(num_squares))/2
    image_borderSize[1] = (Screen_size_pixels[1] - figure_borderSize*2 - squareSize_pixels*(num_squares))/2
    
    #How big is each rectangle in units of millimeters?
    squareSize_mm = Screen_size_mm*squareSize_pixels/Screen_size_pixels
    
    #How big is the checkered part of the image
    image_inner_dim = num_squares*squareSize_pixels # the dimenstion of the inside of the image (not including the border)
    
    #Create a black image to fit both the checkerboard and the image border
    imgTemplate = np.ones((image_inner_dim+2*image_borderSize[1], image_inner_dim+2*image_borderSize[0]))
    
    ## create the checkerboard image
    img = imgTemplate
    
    for x in range(0, num_squares):
        for y in range((x)%2, num_squares, 2):
            minx = image_borderSize[1]+(x)*squareSize_pixels        
            maxx = image_borderSize[1]+(x+1)*squareSize_pixels
            miny = image_borderSize[0]+(y)*squareSize_pixels
            maxy = image_borderSize[0]+(y+1)*squareSize_pixels
            img[minx:maxx,miny:maxy] = 1
        
        for y in range((x+1)%2, num_squares, 2):
            minx = image_borderSize[1]+(x)*squareSize_pixels
            maxx = image_borderSize[1]+(x+1)*squareSize_pixels
            miny = image_borderSize[0]+(y)*squareSize_pixels
            maxy = image_borderSize[0]+(y+1)*squareSize_pixels
            img[minx:maxx,miny:maxy] = 0
            #xloc = range(image_borderSize+((x-1)*squareSize_pixels),(x*squareSize_pixels+image_borderSize))
            #yloc = range(image_borderSize+((y-1)*squareSize_pixels),(y*squareSize_pixels+image_borderSize))
            #img[[xloc],[yloc]] = 0
    
    
    
    # display
    cv2.imshow('image', img)
    print('Warning: Do not resize the checkerboard image window! It has been shown on the screen at a specific size which must be known for calibration')
    
    print('press any key when done recording images')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print('Checkerboard rectangle size is:')
    #print(['Vertical: ', num2str(squareSize_mm(2)), 'mm'])
    #print(['Horizontal: ', num2str(squareSize_mm(1)), 'mm'])
    
    #if num_flashes>1
    #    print('Press any button to begin flashing...\n');
    #    cv2.Waitkey(0)
    #    cv2.imshow('image', img)
    #    pause(1) %small pause
    #    
    #    % flash 'num_flashes' times
    #    for i = 1:num_flashes
    #        imshow(imgTemplate')
    #        drawnow;
    #        imshow(img')
    #        drawnow;
    #    end
    #end
    #
    #dX = squareSize_mm(1);
    #dY = squareSize_mm(2);

    return squareSize_mm
    


def auto_calibrate(num_squares, squareSize_mm, scale, image_directory, image_format):
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # how large are the squares?
    square_sidelength = squareSize_mm[1]; 
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros(((num_squares-1)*(num_squares-1),3), np.float32)
    objp[:,:2] = np.mgrid[0:(num_squares-1),0:(num_squares-1)].T.reshape(-1,2)*square_sidelength
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    
    images = glob.glob(image_directory +'\\*' + image_format)
    
    
    
    for fname in images:
        
        img_original = cv2.imread(fname)
        gray_original = cv2.cvtColor(img_original,cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(img_small, None, fx=1, fy=1, interpolation = cv2.INTER_CUBIC)
        img = cv2.resize(img_original, None, fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        
        #cv2.imshow('img',gray)
        #cv2.waitKey(0)
        
        #gray = cv2.equalizeHist(gray)
        threshold = 128;
        keypressed = 0
        while keypressed !=13:
            ret, gray_threshold = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            cv2.imshow('img',gray_threshold)
            keypressed = cv2.waitKey(0)
            if keypressed == 2490368:
                threshold = threshold+1
            if keypressed == 2621440:
                threshold = threshold-1
        
        gray = gray_threshold
        # Find the chess board corners
        #ret, corners = cv2.findChessboardCorners(gray, (9,9), flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret, corners = cv2.findChessboardCorners(gray, ((num_squares-1),(num_squares-1)), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)
        
        # If found, add object points, image points (after refining them)
        if ret == True:
        #if not (corners is None):
            objpoints.append(objp)
    
            corners2 = cv2.cornerSubPix(gray,corners,(4,4),(-1,-1),criteria)
            imgpoints.append(corners2/scale)
    
    
            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, ((num_squares-1),(num_squares-1)), corners2,ret)
            cv2.imshow('img',img)
            cv2.waitKey(0)
    
    cv2.destroyAllWindows()
    
    # perform the calibration
    
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_original.shape[::-1],None,None)
    
    # calculate the error
    tot_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
        tot_error += error
    
    print "mean error: ", tot_error/len(objpoints)
    
    return ret, mtx, dist, rvecs, tvecs

#EM.x = np.array(range(10))
#EM.y = np.array(range(10))
#EM.ts = np.array(range(10))
#EM.p = np.array(range(10))
    
    
def extract_indices(events, logical_indices):
    eventsOut = Events()
    if sum(logical_indices)>0:
        eventsOut = events
        for i in eventsOut.__dict__.keys():
            temp = getattr(eventsOut, i)
            temp = temp[logical_indices]
            setattr(eventsOut, i, temp)
    return eventsOut
    

def read_aer(filename):
    TD = Events()    
    EM = Events()
    ALLevts = Events()
    f = open(filename, 'rb')
    #raw_data = np.fromfile(f, dtype=np.uint8, count=-1)
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint16(raw_data)
    
    ALLevts.y = raw_data[3::4]
    ALLevts.x = ((raw_data[1::4] & 32)<<3)| raw_data[2::4] #bit 5
    ALLevts.p = (raw_data[1::4] & 128)>>7 #bit 7
    ALLevts.ts = raw_data[0::4] | ((raw_data[1::4] & 31)<<8) # bit 4 downto 0    
    Type = (raw_data[1::4] & 64)>>6 #bit 6
    ALLevts.ts = ALLevts.ts.astype('uint')
    
    timeOffset = 0;
    for i in range(len(ALLevts.ts)):
        if ((ALLevts.y[i] == 240) and (ALLevts.x[i] ==305)):
            Type[i] = 2;
            timeOffset = timeOffset + 2**13;
        else:
            ALLevts.ts[i] = ALLevts.ts[i] + timeOffset;
    
    valid_indices = Type != 2
    ALLevts = extract_indices(ALLevts, valid_indices)
    Type = Type[valid_indices]
    
    valid_indices = Type==1
    EM = extract_indices(ALLevts, valid_indices)
    valid_indices = Type==0
    TD = extract_indices(ALLevts, valid_indices)
    
    return TD, EM

def show_em(em_events):
    max_x = max(em_events.x)+1
    max_y = max(em_events.y)+1
    thr_valid = np.zeros((max_y, max_x))
    thr_l = np.zeros((max_y, max_x))
    thr_h = np.zeros((max_y, max_x))

    frame_length = 24e3;
    t_max = len(em_events.ts)-1
    frame_end = em_events.ts[1] + frame_length
    i=0
    while i<t_max:
        while (em_events.ts[i] < frame_end) and (i<t_max):
            if em_events.p[i] == 0:
                thr_valid[em_events.y[i], em_events.x[i]] = 1;
                thr_l[em_events.y[i], em_events.x[i]] = em_events.ts[i]
            else:
                if thr_valid[em_events.y[i], em_events.x[i]] == 1:
                    thr_valid[em_events.y[i], em_events.x[i]] = 0
                    thr_h[em_events.y[i], em_events.x[i]] = em_events.ts[i] - thr_l[em_events.y[i], em_events.x[i]]
            i = i+1
            
        maxVal = 1.16e5
        minVal = 1.74e3
    
        img = 255*(1-(thr_h-minVal)/(maxVal-minVal))
        #thr_h = cv2.adaptiveThreshold(thr_h, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
        img[img<0] = 0
        img[img>255] = 255
        img = img.astype('uint8')
        cv2.imshow('img',img)
        cv2.waitKey(1)
        frame_end = frame_end+frame_length
        
    cv2.destroyAllWindows()
    return


def show_td(td_events):
    max_x = max(td_events.x)+1
    max_y = max(td_events.y)+1

    frame_length = 24e3;
    t_max = len(td_events.ts)-1
    frame_end = td_events.ts[1] + frame_length
    i=0
    while i<t_max:
        td_img = 0.5*np.ones((max_y, max_x))
        while (td_events.ts[i] < frame_end) and (i<t_max):
            td_img[td_events.y[i], td_events.x[i]] = td_events.p[i]
            i = i+1
    
        img = 255*td_img
        img = img.astype('uint8')
        cv2.imshow('img',img)
        cv2.waitKey(1)
        frame_end = frame_end+frame_length
        
    cv2.destroyAllWindows()
    return

def filter_td(td_events, us_time):
   
    max_x = max(td_events.x)
    max_y = max(td_events.y)
    T0 = np.ones((max_x+1,max_y+1))-us_time-1
    x_prev = 0;
    y_prev = 0;
    p_prev = 0;
    
    valid_indices = np.ones(len(td_events.ts), np.bool)

    for i in range(len(td_events.ts)):
        if x_prev != td_events.x[i] | y_prev != td_events.y[i] | p_prev != td_events.p[i]:
            T0[td_events.x[i], td_events.y[i]]=  -us_time
            min_x_sub = max(0, td_events.x[i]-1)
            max_x_sub = min(max_x, td_events.x[i]+1)
            min_y_sub = max(0, td_events.y[i]-1)
            max_y_sub = min(max_y, td_events.y[i]+1)
            
            T0temp = T0[min_x_sub:(max_x_sub+1), min_y_sub:(max_y_sub+1)]
             
            if min(td_events.ts[i]-T0temp.reshape(-1,1)) > us_time:
                valid_indices[i] = 0

        T0[td_events.x[i], td_events.y[i]] =  td_events.ts[i]
        x_prev = td_events.x[i]
        y_prev = td_events.y[i]
        p_prev = td_events.p[i]
    
    return  extract_indices(td_events, valid_indices.astype('bool'))

def sort_order(events):
    print('the function sort_order has not yet been thoroughly tested')
    inds = events.ts.argsort()
    eventsOut = events
    for i in eventsOut.__dict__.keys():
        temp = getattr(eventsOut, i)
        temp = temp[inds]
        setattr(eventsOut, i, temp)
    return eventsOut

def extract_roi(td_events, top_left, size):
    valid_indices = (td_events.x > top_left[0]) & (td_events.y > top_left[1]) & (td_events.x < (size[0]+top_left[0])) & (td_events.y > (top_left[1]+size[1]))
    return  extract_indices(td_events, valid_indices.astype('bool'))


def implement_refraction(td_events, us_time):
    max_x = max(td_events.x)
    max_y = max(td_events.y)
    T0 = np.ones((max_x+1,max_y+1))-us_time-1
   
    valid_indices = np.ones(len(td_events.ts), np.bool)

    for i in range(len(td_events.ts)):
        if (td_events.ts[i] - T0[td_events.x[i], td_events.y[i]]) < us_time:
            valid_indices[i] = 0
        else:
            valid_indices[i] = 1
            T0[td_events.x[i], td_events.y[i]] =  td_events.ts[i]
    
    return  extract_indices(td_events, valid_indices.astype('bool'))

def write2jAER(td_events, filename):
    import time
    y = 479-td_events.y
    #y = td_events.y
    y_shift = 22+32

    x = 639-td_events.x
    #x = td_events.x
    x_shift = 12+32

    p = td_events.p
    p_shift = 11+32

    ts_shift = 0
    
    y_final= y.astype(dtype=np.uint64)<<y_shift
    x_final= x.astype(dtype=np.uint64)<<x_shift
    p_final= p.astype(dtype=np.uint64)<<p_shift
    ts_final = td_events.ts.astype(dtype=np.uint64)<<ts_shift
    vector_all =  np.array(y_final + x_final + p_final + ts_final, dtype=np.uint64)
    aedat_file=open(filename,'wb')

    version='2.0'
    aedat_file.write('#!AER-DAT'+ version+'\r\n')
    aedat_file.write('# This is a raw AE data file - do not edit\r\n')
    aedat_file.write('# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n')
    aedat_file.write('# Timestamps tick is 1 us\r\n')
    aedat_file.write('# created ' + time.strftime("%d/%m/%Y") + ' ' + time.strftime("%H:%M:%S") + ' by the Python function "write2jAER"\r\n')
    aedat_file.write('# This function fakes the format of DAVIS640 to allow for the full ATIS address space to be used (304x240)\r\n')
    ##aedat_file.write(vector_all.astype(dtype='>u8').tostring())    
    to_write = bytearray(vector_all[::-1])
    to_write.reverse()
    aedat_file.write(to_write)
    #aedat_file.write(vector_all)
    #vector_all.tofile(aedat_file)
    aedat_file.close()

def read_dataset(filename):
    TD = Events()    
    f = open(filename, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)
    
    TD.x = raw_data[0::5]
    TD.y = raw_data[1::5]
    TD.p = (raw_data[2::5]& 128)>>7 #bit 7
    TD.ts = ((raw_data[2::5]& 127)<<16) | (raw_data[3::5]<<8) | (raw_data[4::5])
    Type = np.zeros(len(TD.ts), dtype=np.uint8)
    timeOffset = 0;
    for i in range(len(TD.ts)):
        if ((TD.y[i] == 240) and (TD.x[i] ==305)):
            Type[i] = 2;
            timeOffset = timeOffset + 2**13;
        else:
            TD.ts[i] = TD.ts[i] + timeOffset;
    valid_indices = Type != 2
    TD = extract_indices(TD, valid_indices)
    return TD

print('Event-based vision module imported')