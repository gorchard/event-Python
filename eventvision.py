# -*- coding: utf-8 -*-
"""
This module contains classes, functions and an example (main) for handling AER vision data.
"""
import cv2
import glob
import numpy as np
from win32api import GetSystemMetrics

class Events(object):
    """
    Temporal Difference events. 
    data: a NumPy Record Array with the following named fields
        x: pixel x coordinate, unsigned 16bit int
        y: pixel y coordinate, unsigned 16bit int
        p: polarity value, boolean. False=off, True=on
        ts: timestamp in microseconds, unsigned 64bit int
    """
    def __init__(self, num_events):
        """num_spikes: number of events this instance will initially contain"""
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.uint64)], shape=(num_events))

    def show_em(self):
        """Displays the EM events (grayscale ATIS events)"""
        max_x = self.data.x.max() + 1
        max_y = self.data.y.max() + 1
        thr_valid = np.zeros((max_y, max_x))
        thr_l = np.zeros((max_y, max_x))
        thr_h = np.zeros((max_y, max_x))

        frame_length = 24e3
        t_max = len(self.data) - 1
        frame_end = self.data[1].ts + frame_length
        i = 0
        while i < t_max:
            while (self.data[i].ts < frame_end) and (i < t_max):
                datum = self.data[i]
                if datum.p == 0:
                    thr_valid[datum.y, datum.x] = 1
                    thr_l[datum.y, datum.x] = datum.ts
                elif thr_valid[datum.y, datum.x] == 1:
                    thr_valid[datum.y, datum.x] = 0
                    thr_h[datum.y, datum.x] = datum.ts - thr_l[datum.y, datum.x]
                i = i + 1

            max_val = 1.16e5
            min_val = 1.74e3

            img = 255 * (1 - (thr_h - min_val) / (max_val - min_val))
            #thr_h = cv2.adaptiveThreshold(thr_h, 255,
            #cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
            img[img < 0] = 0
            img[img > 255] = 255
            img = img.astype('uint8')
            cv2.imshow('img', img)
            cv2.waitKey(1)
            frame_end = frame_end + frame_length

        cv2.destroyAllWindows()
        return

    def show_td(self, wait_delay=1):
        """Displays the TD events (change detection ATIS or DVS events)
        waitDelay: milliseconds
        """
        max_x = self.data.x.max() + 1
        max_y = self.data.y.max() + 1

        frame_length = 24e3
        t_max = len(self.data) - 1
        frame_end = self.data[1].ts + frame_length
        i = 0
        while i < t_max:
            td_img = 0.5 * np.ones((max_y, max_x))
            while (self.data[i].ts < frame_end) and (i < t_max):
                datum = self.data[i]
                td_img[datum.y, datum.x] = datum.p
                i = i + 1

            img = 255 * td_img
            img = img.astype('uint8')
            cv2.imshow('img', img)
            cv2.waitKey(wait_delay)
            frame_end = frame_end + frame_length

        cv2.destroyAllWindows()
        return

    def filter_td(self, us_time):
        """Generate a filtered set of event data.
        Does not modify instance data
        Uses a background activity filter on the events, such that only events which are
        correlated with a neighbouring event within 'us_time' microseconds will be allowed
        through the filter.
        us_time: microseconds
        """
        max_x = self.data.x.max()
        max_y = self.data.y.max()
        t0 = np.ones((max_x + 1, max_y + 1)) - us_time - 1
        x_prev = 0
        y_prev = 0
        p_prev = 0

        valid_indices = np.ones(len(self.data), np.bool_)
        i = 0

        for datum in np.nditer(self.data):
            datum_ts = datum['ts'].item(0)
            datum_x = datum['x'].item(0)
            datum_y = datum['y'].item(0)
            datum_p = datum['p'].item(0)
            if x_prev != datum_x | y_prev != datum_y | p_prev != datum_p:
                t0[datum_x, datum_y] = -us_time
                min_x_sub = max(0, datum_x - 1)
                max_x_sub = min(max_x, datum_x + 1)
                min_y_sub = max(0, datum_y - 1)
                max_y_sub = min(max_y, datum_y + 1)

                t0_temp = t0[min_x_sub:(max_x_sub + 1), min_y_sub:(max_y_sub + 1)]

                if min(datum_ts - t0_temp.reshape(-1, 1)) > us_time:
                       valid_indices[i] = 0

            t0[datum_x, datum_y] = datum_ts
            x_prev = datum_x
            y_prev = datum_y
            p_prev = datum_p
            i = i + 1

        return self.data[valid_indices.astype('bool')]

    def sort_order(self):
        """Generate data sorted by ascending ts
        Does not modify instance data
        Will look through the struct events, and sort all events by the field 'ts'.
        In other words, it will ensure events_out.ts is monotonically increasing,
        which is useful when combining events from multiple recordings.
        """
        #chose mergesort because it is a stable sort, at the expense of more memory usage
        self.data = np.sort(self.data, order='ts', kind='mergesort')
        inds = self.ts.argsort()
        events_out = self
        for i in events_out.__dict__.keys():
            temp = getattr(events_out, i)
            temp = temp[inds]
            setattr(events_out, i, temp)
        return events_out

    def extract_roi(self, top_left, size, is_normalize=False):
        """Extract Region of Interest
        Does not modify instance data
        Generates a set of td_events which fall into a rectangular region of interest with
        top left corner at 'top_left' and size 'size'
        top_left: [x: int, y: int]
        size: [width, height]
        is_normalize: bool. If True, x and y values will be normalized to the cropped region
        """
        min_x = top_left[0]
        min_y = top_left[1]
        max_x = size[0] + min_x
        max_y = size[1] + min_y
        extracted_data = self.data[(self.data.x >= min_x)]
        extracted_data = extracted_data[extracted_data.y >= min_y]
        extracted_data = extracted_data[extracted_data.x < max_x]
        extracted_data = extracted_data[extracted_data.y < max_y]

        if is_normalize:
            extracted_data.x = extracted_data.x - min_x
            extracted_data.y = extracted_data.y - min_y

        return extracted_data

    def apply_refraction(self, us_time):
        """Implements a refractory period for each pixel.
        Does not modify instance data
        In other words, if an event occurs within 'us_time' microseconds of
        a previous event at the same pixel, then the second event is removed
        us_time: time in microseconds
        """
        max_x = self.data.x.max()
        max_y = self.data.y.max()
        t0 = np.ones((max_x + 1, max_y + 1)) - us_time - 1

        valid_indices = np.ones(len(self.data), np.bool_)
        i = 0

        for datum in np.nditer(self.data):
            datum_ts = datum['ts'].item(0)
            datum_x = datum['x'].item(0)
            datum_y = datum['y'].item(0)
            if datum_ts - t0[datum_x, datum_y] < us_time:
                valid_indices[i] = 0
            else:
                valid_indices[i] = 1
                t0[datum_x, datum_y] = datum_ts

            i = i + 1

        return self.data[valid_indices.astype('bool')]

    def write_j_aer(self, filename):
        """
        writes the td events in 'td_events' to a file specified by 'filename'
        which is compatible with the jAER framework.
        To view these events in jAER, make sure to select the DAVIS640 sensor.
        """
        import time
        y = 479 - self.data.y
        #y = td_events.y
        y_shift = 22 + 32

        x = 639 - self.data.x
        #x = td_events.x
        x_shift = 12 + 32

        p = self.data.p + 1
        p_shift = 11 + 32

        ts_shift = 0

        y_final = y.astype(dtype=np.uint64) << y_shift
        x_final = x.astype(dtype=np.uint64) << x_shift
        p_final = p.astype(dtype=np.uint64) << p_shift
        ts_final = self.data.ts.astype(dtype=np.uint64) << ts_shift
        vector_all = np.array(y_final + x_final + p_final + ts_final, dtype=np.uint64)
        aedat_file = open(filename, 'wb')

        version = '2.0'
        aedat_file.write('#!AER-DAT' + version + '\r\n')
        aedat_file.write('# This is a raw AE data file - do not edit\r\n')
        aedat_file.write \
            ('# Data format is int32 address, int32 timestamp (8 bytes total), repeated for each event\r\n')
        aedat_file.write('# Timestamps tick is 1 us\r\n')
        aedat_file.write('# created ' + time.strftime("%d/%m/%Y") \
            + ' ' + time.strftime("%H:%M:%S") \
            + ' by the Python function "write2jAER"\r\n')
        aedat_file.write \
            ('# This function fakes the format of DAVIS640 to allow for the full ATIS address space to be used (304x240)\r\n')
        ##aedat_file.write(vector_all.astype(dtype='>u8').tostring())
        to_write = bytearray(vector_all[::-1])
        to_write.reverse()
        aedat_file.write(to_write)
        #aedat_file.write(vector_all)
        #vector_all.tofile(aedat_file)
        aedat_file.close()

def present_checkerboard(num_squares):
    """
    Presents a checkerboard pattern of size num_squares*num_squares on the screen.
    The function will automatically detect the screen size in pixels and assume a
    resolution of 96 dpi to provide the square size in mm.
    """
    screen_width_pixels = GetSystemMetrics(0)
    screen_height_pixels = GetSystemMetrics(1)

    #fixed parameters of the setup
    figure_border_size = 30 #leave space of 100 pixels on each side of the axes for the figure
                            #controls etc
    #image_border_size = 10 #within the image, create a border of size 10
                            #pixels to ensure contrast with the outside
                                                       #rectangles

    #How big is each rectangle in units of pixels?
    screen_size_pixels = np.array([screen_width_pixels, screen_height_pixels])
    screen_size_mm = 0.00254 * screen_size_pixels / 96
    square_size_pixels = int(min(screen_size_pixels - 2 * figure_border_size) / (num_squares + 2))

    image_border_size = np.array([1, 2])
    image_border_size[0] = (screen_size_pixels[0] - figure_border_size * 2 - square_size_pixels * (num_squares)) / 2
    image_border_size[1] = (screen_size_pixels[1] - figure_border_size * 2 - square_size_pixels * (num_squares)) / 2

    #How big is each rectangle in units of millimeters?
    square_size_mm = screen_size_mm * square_size_pixels / screen_size_pixels

    #How big is the checkered part of the image
    image_inner_dim = num_squares * square_size_pixels # the dimenstion of the inside of the image (not including the border)

    #Create a black image to fit both the checkerboard and the image border
    img_template = np.ones((image_inner_dim + 2 * image_border_size[1], image_inner_dim + 2 * image_border_size[0]))

    ## create the checkerboard image
    img = img_template

    for x in range(0, num_squares):
        for y in range((x) % 2, num_squares, 2):
            minx = image_border_size[1] + (x) * square_size_pixels
            maxx = image_border_size[1] + (x + 1) * square_size_pixels
            miny = image_border_size[0] + (y) * square_size_pixels
            maxy = image_border_size[0] + (y + 1) * square_size_pixels
            img[minx:maxx, miny:maxy] = 1

        for y in range((x + 1) % 2, num_squares, 2):
            minx = image_border_size[1] + (x) * square_size_pixels
            maxx = image_border_size[1] + (x + 1) * square_size_pixels
            miny = image_border_size[0] + (y) * square_size_pixels
            maxy = image_border_size[0] + (y + 1) * square_size_pixels
            img[minx:maxx, miny:maxy] = 0
            #xloc =
            #range(image_borderSize+((x-1)*squareSize_pixels),(x*squareSize_pixels+image_borderSize))
            #yloc =
            #range(image_borderSize+((y-1)*squareSize_pixels),(y*squareSize_pixels+image_borderSize))
            #img[[xloc],[yloc]] = 0

    # display
    cv2.imshow('image', img)
    print 'Warning: Do not resize the checkerboard image window! It has been shown on the screen at a specific size which must be known for calibration'

    print 'press any key when done recording images'
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    #print('Checkerboard rectangle size is:')
    #print(['Vertical: ', num2str(squareSize_mm(2)), 'mm'])
    #print(['Horizontal: ', num2str(squareSize_mm(1)), 'mm'])

    #if num_flashes>1
    #    print('Press any button to begin flashing...\n')
    #    cv2.Waitkey(0)
    #    cv2.imshow('image', img)
    #    pause(1) %small pause
    #
    #    % flash 'num_flashes' times
    #    for i = 1:num_flashes
    #        imshow(imgTemplate')
    #        drawnow
    #        imshow(img')
    #        drawnow
    #    end
    #end
    #
    #dX = squareSize_mm(1)
    #dY = squareSize_mm(2)

    return square_size_mm

def auto_calibrate(num_squares, square_size_mm, scale, image_directory, image_format):
    """
    Will read in images of extension 'image_format' from the directory 'image_directory' for calibration.
    Each image should contain a checkerboard with 'num_squares'*'num_squares' squares,
    each of size 'squareSize_mm'.
    'scale' is an optional argument to rescale images before calibration
    because ATIS/DVS have very low resolution and calibration algorithms are used to handling larger images (use 4)
    """
    # termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # how large are the squares?
    square_sidelength = square_size_mm[1]

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros(((num_squares - 1) * (num_squares - 1), 3), np.float32)
    objp[:, :2] = np.mgrid[0:(num_squares - 1), 0:(num_squares - 1)].T.reshape(-1, 2) * square_sidelength

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.

    images = glob.glob(image_directory + '\\*' + image_format)



    for fname in images:
        img_original = cv2.imread(fname)
        gray_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2GRAY)
        #img = cv2.resize(img_small, None, fx=1, fy=1, interpolation =
        #cv2.INTER_CUBIC)
        img = cv2.resize(img_original, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #cv2.imshow('img',gray)
        #cv2.waitKey(0)

        #gray = cv2.equalizeHist(gray)
        threshold = 128
        keypressed = 0
        while keypressed != 13:
            ret, gray_threshold = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            cv2.imshow('img', gray_threshold)
            keypressed = cv2.waitKey(0)
            if keypressed == 2490368:
                threshold = threshold + 1
            if keypressed == 2621440:
                threshold = threshold - 1

        gray = gray_threshold
        # Find the chess board corners
        #ret, corners = cv2.findChessboardCorners(gray, (9,9),
        #flags=cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret, corners = cv2.findChessboardCorners(gray, ((num_squares - 1), (num_squares - 1)), flags=cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if ret:
        #if not (corners is None):
            objpoints.append(objp)

            corners2 = cv2.cornerSubPix(gray, corners, (4, 4), (-1, -1), criteria)
            imgpoints.append(corners2 / scale)


            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, ((num_squares - 1), (num_squares - 1)), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    # perform the calibration

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_original.shape[::-1], None, None)

    # calculate the error
    tot_error = 0
    for i in xrange(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        tot_error += error

    print "mean error: ", tot_error / len(objpoints)

    return ret, mtx, dist, rvecs, tvecs

def read_aer(filename):
    """Reads in the ATIS file specified by 'filename' and returns the TD and EM events.
    This only works for ATIS recordings directly from the GUI.
    If you are working with the N-MNIST or N-CALTECH101 datasets, use read_dataset(filename) instead
    """
    f = open(filename, 'rb')
    #raw_data = np.fromfile(f, dtype=np.uint8, count=-1)
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint16(raw_data)

    all_y = raw_data[3::4]
    all_x = ((raw_data[1::4] & 32) << 3) | raw_data[2::4] #bit 5
    all_p = (raw_data[1::4] & 128) >> 7 #bit 7
    all_ts = raw_data[0::4] | ((raw_data[1::4] & 31) << 8) # bit 4 downto 0
    all_event_type = (raw_data[1::4] & 64) >> 6 #bit 6
    all_ts = all_ts.astype('uint')
    td_event_indices = np.zeros(len(all_y), dtype=np.bool_)
    em_event_indices = np.copy(td_event_indices)
    0.
    # Iterate through the events, looking out for time stamp overflow events
    # And update the td and em event indices at the same time
    time_offset = 0
    for i in range(len(all_ts)):
        if (all_y[i] == 240) and (all_x[i] == 305):
            #timestamp overflow, increment the time offset
            time_offset = time_offset + 2 ** 13
        else:
            #apply time offset
            all_ts[i] = all_ts[i] + time_offset
            
            #update the td and em event indices
            if all_event_type[i] == 1:
                em_event_indices[i] = True
            else:
                td_event_indices[i] = True

    em = Events(em_event_indices.sum())
    em.data.x = all_x[em_event_indices]
    em.data.y = all_y[em_event_indices]
    em.data.ts = all_ts[em_event_indices]
    em.data.p = all_p[em_event_indices]

    td = Events(td_event_indices.sum())
    td.data.x = all_x[td_event_indices]
    td.data.y = all_y[td_event_indices]
    td.data.ts = all_ts[td_event_indices]
    td.data.p = all_p[td_event_indices]

    return td, em

def read_dataset(filename):
    """Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
    f = open(filename, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    all_y = raw_data[1::5]
    all_x = raw_data[0::5]
    all_p = (raw_data[2::5] & 128) >> 7 #bit 7
    all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
    td_indices = np.zeros(len(all_ts), dtype=np.bool_)

    #Iterate through the events, looking out for time stamp overflow events
    #And update td indices at the same time (by excluding time stamp events)
    time_offset = 0
    for i in range(len(all_ts)):
        if (all_y[i] == 240) and (all_x[i] == 305):
            #timestamp overflow, increment the time offset
            time_offset = time_offset + 2 ** 13
        else:
            #apply time offset
            all_ts[i] = all_ts[i] + time_offset
            td_indices[i] = True

    td = Events(td_indices.sum())
    td.data.x = all_x[td_indices]
    td.data.y = all_y[td_indices]
    td.data.ts = all_ts[td_indices]
    td.data.p = all_p[td_indices]
    return td

def read_bin_linux(filename):
    """
    Reads in ATIS .bin files generated by the linux interface.
    If working with N-MNIST or N-CALTECH101 datasets, use read_dataset(filename).
    If working with recordings from the GUI, use read_aer(filename).
    Returns TD, containing:
        data: a NumPy Record Array with the following named fields
            x: pixel x coordinate, unsigned 16bit int
            y: pixel y coordinate, unsigned 16bit int
            p: polarity value, boolean. False = off event, True = on event
            ts: timestamp in microseconds, unsigned 64bit int
    """
    
    with open(filename, 'rb') as f:
    
        # Strip header

        header_line = f.readline()
   
        while(header_line[0] == '#'):
            header_line = f.readline()

        raw_data = np.fromfile(f, dtype = np.uint8)

    # file already closed since using 'with' statement

    total_events = len(raw_data)

    full_x = np.zeros(total_events)
    full_y = np.zeros(total_events)
    full_p = np.zeros(total_events)
    full_ts = np.zeros(total_events)
    full_f = np.zeros(total_events)    
    
    TD_indices = np.zeros(total_events, dtype=np.bool_)

    total_events = 0
    buffer_location = 0
    start_evt_ind = 0

    while(buffer_location < len(raw_data)):
    
        num_events = ((raw_data[buffer_location+3].astype(np.uint32)<<24)
                        + (raw_data[buffer_location+2].astype(np.uint32)<<16)
                        + (raw_data[buffer_location+1].astype(np.uint32)<<8)
                        + raw_data[buffer_location])

        buffer_location = buffer_location + 4
    
        start_time = ((raw_data[buffer_location+3].astype(np.uint32)<<24)
                        + (raw_data[buffer_location+2].astype(np.uint32)<<16)
                        + (raw_data[buffer_location+1].astype(np.uint32)<<8)
                        + raw_data[buffer_location])

        buffer_location = buffer_location + 8

        # Note renaming (since original is a Python built-in): 
	# type = evt_type and subtype = evt_subtype 

        evt_type = raw_data[buffer_location:(buffer_location+8*num_events):8]
        evt_subtype = raw_data[(buffer_location + 1):(buffer_location + 8*num_events +1 ):8]
        y = raw_data[(buffer_location + 2):(buffer_location + 8*num_events + 2):8]
        x = ((raw_data[(buffer_location + 5):(buffer_location + 8*num_events + 5):8].astype(np.uint16)<<8)
             + (raw_data[(buffer_location + 4):(buffer_location + 8*num_events + 4):8]))
        ts = ((raw_data[(buffer_location + 7):(buffer_location + 8*num_events + 7):8].astype(np.uint32)<<8)
              + (raw_data[(buffer_location + 6):(buffer_location + 8*num_events + 6):8]))

        buffer_location = buffer_location + num_events*8;

        ts = ts + start_time

        overflows = np.where(evt_type == 2)

        for i in range(0, len(overflows[0])):
            overflow_loc = overflows[0][i]
            ts[overflow_loc:] = ts[overflow_loc:] + 65536

        locations = np.where((evt_type == 0) | (evt_type == 3))
        TD_indices[start_evt_ind:(start_evt_ind + num_events)][locations] = True

        full_x[start_evt_ind:(start_evt_ind + num_events)] = x
        full_y[start_evt_ind:(start_evt_ind + num_events)] = y
        full_p[start_evt_ind:(start_evt_ind + num_events)] = evt_subtype
        full_ts[start_evt_ind:(start_evt_ind + num_events)] = ts
        full_f[start_evt_ind:(start_evt_ind + num_events)] = evt_type

        start_evt_ind = start_evt_ind + num_events

    TD = Events(len(full_x[TD_indices]))

    # If intefacing with Matlab, 1 must be added to x and y indices.
    # due to Matlab's index convention.

    TD.data.x = full_x[TD_indices]  # + 1       
    TD.data.y = full_y[TD_indices]  # + 1
    TD.data.ts = full_ts[TD_indices]
    TD.data.p = full_p[TD_indices]

    return TD


def main():
    """Example usage of eventvision"""
    #read in some data
    #td, em = read_aer('0000.val')
    td = read_dataset('trainReduced/0/00002.bin')

    #show the TD events
    td.show_td(100)

    #extract a region of interest...
    #note this will also edit the event struct 'TD'
    #td.data = ev.extract_roi(TD, [50,50], [150,150])
    td.data = td.extract_roi([3, 3], [20, 20])

    #implement a refractory period...
    #note this will also edit the event #struct 'TD2'
    td.data = td.apply_refraction(0.03)

    #perform some noise filtering...
    #note this will also edit the event struct 'TD3'
    td.data = td.filter_td(0.03)

    #show the resulting data
    td.show_td(100)

    #write the filtered data in a format jAER can understand
    td.write_j_aer('jAERdata.aedat')


    #show the grayscale data
    em.show_em()


    #perform camera calibration
    #first show the calibration pattern on the screen and make some recordings:
    num_squares = 10
    square_size_mm = ev.present_checkerboard(num_squares)

    #state where the recordings are what format they are in
    image_directory = 'path_to_calibration_images'
    image_format = '.bmp'

    #using a scale is useful for visualization
    scale = 4

    #call the calibration function and follow the instructions provided
    ret, mtx, dist, rvecs, tvecs = ev.auto_calibrate(num_squares, square_size_mm, scale, image_directory, image_format)

if __name__ == "__main__":
    main()

print 'Event-based vision module imported'
