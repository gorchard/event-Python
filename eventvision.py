# -*- coding: utf-8 -*-
"""
This module contains classes, functions and an example (main) for handling AER vision data.
"""
import glob
import cv2
import numpy as np
from win32api import GetSystemMetrics
import timer

class Events(object):
    """
    Temporal Difference events.
    data: a NumPy Record Array with the following named fields
        x: pixel x coordinate, unsigned 16bit int
        y: pixel y coordinate, unsigned 16bit int
        p: polarity value, boolean. False=off, True=on
        ts: timestamp in microseconds, unsigned 64bit int
    width: The width of the frame. Default = 304.
    height: The height of the frame. Default = 240.
    """
    def __init__(self, num_events, width=304, height=240):
        """num_spikes: number of events this instance will initially contain"""
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('p', np.bool_), ('ts', np.uint64)], shape=(num_events))
        self.width = width
        self.height = height

    def show_em(self):
        """Displays the EM events (grayscale ATIS events)"""
        frame_length = 24e3
        t_max = self.data.ts[-1]
        frame_start = self.data[0].ts
        frame_end = self.data[0].ts + frame_length
        max_val = 1.16e5
        min_val = 1.74e3
        val_range = max_val - min_val

        thr = np.rec.array(None, dtype=[('valid', np.bool_), ('low', np.uint64), ('high', np.uint64)], shape=(self.height, self.width))
        thr.valid.fill(False)
        thr.low.fill(frame_start)
        thr.high.fill(0)

        def show_em_frame(frame_data):
            """Prepare and show a single frame of em data to be shown"""
            for datum in np.nditer(frame_data):
                ts_val = datum['ts'].item(0)
                thr_data = thr[datum['y'].item(0), datum['x'].item(0)]

                if datum['p'].item(0) == 0:
                    thr_data.valid = 1
                    thr_data.low = ts_val
                elif thr_data.valid == 1:
                    thr_data.valid = 0
                    thr_data.high = ts_val - thr_data.low

            img = 255 * (1 - (thr.high - min_val) / (val_range))
            #thr_h = cv2.adaptiveThreshold(thr_h, 255,
            #cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 0)
            img = np.piecewise(img, [img <= 0, (img > 0) & (img < 255), img >= 255], [0, lambda x: x, 255])
            img = img.astype('uint8')
            cv2.imshow('img', img)
            cv2.waitKey(1)

        while frame_start < t_max:
            #with timer.Timer() as em_playback_timer:
            frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
            show_em_frame(frame_data)
            frame_start = frame_end + 1
            frame_end += frame_length + 1
            #print 'showing em frame took %s seconds' %em_playback_timer.secs

        cv2.destroyAllWindows()
        return

    def show_td(self, wait_delay=1):
        """Displays the TD events (change detection ATIS or DVS events)
        waitDelay: milliseconds
        """
        frame_length = 24e3
        t_max = self.data.ts[-1]
        frame_start = self.data[0].ts
        frame_end = self.data[0].ts + frame_length
        td_img = np.ones((self.height, self.width), dtype=np.uint8)
        while frame_start < t_max:
            frame_data = self.data[(self.data.ts >= frame_start) & (self.data.ts < frame_end)]
            
            if frame_data.size > 0:
                td_img.fill(128)

                #with timer.Timer() as em_playback_timer:
                for datum in np.nditer(frame_data):
                    td_img[datum['y'].item(0), datum['x'].item(0)] = datum['p'].item(0)
                #print 'prepare td frame by iterating events took %s seconds'
                #%em_playback_timer.secs

                td_img = np.piecewise(td_img, [td_img == 0, td_img == 1, td_img == 128], [0, 255, 128])
                cv2.imshow('img', td_img)
                cv2.waitKey(wait_delay)

            frame_start = frame_end + 1
            frame_end = frame_end + frame_length + 1

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
        max_x = self.width - 1
        max_y = self.height - 1
        t0 = np.ones((self.width, self.height)) - us_time - 1
        x_prev = 0
        y_prev = 0
        p_prev = 0

        valid_indices = np.ones(len(self.data), np.bool_)
        i = 0

        with timer.Timer() as ref_timer:
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
        print 'filtering took %s seconds' % ref_timer.secs
        return self.data[valid_indices.astype('bool')]

    def sort_order(self):
        """Generate data sorted by ascending ts
        Does not modify instance data
        Will look through the struct events, and sort all events by the field 'ts'.
        In other words, it will ensure events_out.ts is monotonically increasing,
        which is useful when combining events from multiple recordings.
        """
        #chose mergesort because it is a stable sort, at the expense of more
        #memory usage
        events_out = np.sort(self.data, order='ts', kind='mergesort')
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
        extracted_data = self.data[(self.data.x >= min_x) & (self.data.x < max_x) & (self.data.y >= min_y) & (self.data.y < max_y)]

        if is_normalize:
            self.width = size[0]
            self.height = size[1]
            extracted_data = np.copy(extracted_data)
            extracted_data = extracted_data.view(np.recarray)
            extracted_data.x -= min_x
            extracted_data.y -= min_y

        return extracted_data

    def apply_refraction(self, us_time):
        """Implements a refractory period for each pixel.
        Does not modify instance data
        In other words, if an event occurs within 'us_time' microseconds of
        a previous event at the same pixel, then the second event is removed
        us_time: time in microseconds
        """
        t0 = np.ones((self.width, self.height)) - us_time - 1
        valid_indices = np.ones(len(self.data), np.bool_)

        #with timer.Timer() as ref_timer:
        i = 0
        for datum in np.nditer(self.data):
            datum_ts = datum['ts'].item(0)
            datum_x = datum['x'].item(0)
            datum_y = datum['y'].item(0)
            if datum_ts - t0[datum_x, datum_y] < us_time:
                valid_indices[i] = 0
            else:
                t0[datum_x, datum_y] = datum_ts

            i += 1
        #print 'Refraction took %s seconds' % ref_timer.secs

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
    with timer.Timer() as read_aer_timer:
        file_handle = open(filename, 'rb')
        #raw_data = np.fromfile(f, dtype=np.uint8, count=-1)
        raw_data = np.fromfile(file_handle, dtype=np.uint8)
        file_handle.close()
    print '=> Reading .val file took %s s' % read_aer_timer.secs

    with timer.Timer() as read_aer_timer:
        raw_data = np.uint16(raw_data)
        all_y = raw_data[3::4]
        all_x = ((raw_data[1::4] & 32) << 3) | raw_data[2::4] #bit 5
        all_p = (raw_data[1::4] & 128) >> 7 #bit 7
        #all_ts = raw_data[0::4] | ((raw_data[1::4] & 31) << 8) # bit 4 downto
                                                   #0
        all_ts2 = raw_data[0::4] | ((raw_data[1::4] & 31) << 8) # bit 4 downto 0
        #all_event_type = (raw_data[1::4] & 64) >> 6 #bit 6
        all_event_type2 = (raw_data[1::4] & 64) >> 6 #bit 6
        #all_ts = all_ts.astype('uint')
        all_ts2 = all_ts2.astype('uint')
    print '=> Parsing .val data took %s s' % read_aer_timer.secs

    time_increment = 2 ** 13
    ##old way, much slower
    #with timer.Timer() as read_aer_timer:
    #    td_event_indices = np.zeros(len(all_y), dtype=np.bool_)
    #    em_event_indices = np.copy(td_event_indices)
    #    time_offset = 0
    #    for i, y_val in enumerate(all_y):
    #        if (y_val == 240) and (all_x[i] == 305):
    #            #timestamp overflow, increment the time offset
    #            time_offset += time_increment
    #        else:
    #            #apply time offset
    #            all_ts[i] += time_offset

    #            #update the td and em event indices
    #            em_event_indices[i] = all_event_type[i]
    #            td_event_indices[i] = not em_event_indices[i]
    #print '=> Processing .val data old way took %s s' % read_aer_timer.secs

    # Process time stamp overflow events,
    # then generate the td and em event indices
    with timer.Timer() as read_aer_timer:
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts2[overflow_index:] += time_increment

        all_event_type2[overflow_indices] = 2
        em_event_indices2 = np.where(all_event_type2 == 1)[0]
        td_event_indices2 = np.where(all_event_type2 == 0)[0]
    print '=> Processing .val data new way took %s s' % read_aer_timer.secs

    #em = Events(em_event_indices.sum())
    #em.data.x = all_x[em_event_indices]
    #em.data.y = all_y[em_event_indices]
    #em.data.ts = all_ts[em_event_indices]
    #em.data.p = all_p[em_event_indices]

    #td = Events(td_event_indices.sum())
    #td.data.x = all_x[td_event_indices]
    #td.data.y = all_y[td_event_indices]
    #td.data.ts = all_ts[td_event_indices]
    #td.data.p = all_p[td_event_indices]

    em2 = Events(em_event_indices2.size)
    em2.data.x = all_x[em_event_indices2]
    em2.data.y = all_y[em_event_indices2]
    em2.data.ts = all_ts2[em_event_indices2]
    em2.data.p = all_p[em_event_indices2]

    #if datum.p == 0:
    #    thr_valid[datum.y, datum.x] = 1
    #    thr_l[datum.y, datum.x] = datum.ts
    #elif thr_valid[datum.y, datum.x] == 1:
    #    thr_valid[datum.y, datum.x] = 0
    #    thr_h[datum.y, datum.x] = datum.ts - thr_l[datum.y, datum.x]

    td2 = Events(td_event_indices2.size)
    td2.data.x = all_x[td_event_indices2]
    td2.data.y = all_y[td_event_indices2]
    td2.data.ts = all_ts2[td_event_indices2]
    td2.data.p = all_p[td_event_indices2]

    ##test correctness of new way
    #print np.array_equal(em, em2)
    #print np.array_equal(td, td2)

    # It appears that the polarity needs to be flipped (when results are compared with Matlab output).
    # Change the polarity: 0 events become 1 events and vice versa.

    td2.data.p = np.abs(td2.data.p - 1)

    return td2, em2

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

    #Process time stamp overflow events
    time_increment = 2 ** 13
    overflow_indices = np.where(all_y == 240)[0]
    for overflow_index in overflow_indices:
        all_ts[overflow_index:] += time_increment

    #Everything else is a proper td spike
    td_indices = np.where(all_y != 240)[0]

    td = Events(td_indices.size, 34, 34)
    td.data.x = all_x[td_indices]
    td.width = td.data.x.max() + 1
    td.data.y = all_y[td_indices]
    td.height = td.data.y.max() + 1
    td.data.ts = all_ts[td_indices]
    td.data.p = all_p[td_indices]
    return td

def read_bin_linux(filename):
    """Reads in ATIS .bin files generated by the linux interface.
    If working with N-MNIST or N-CALTECH101 datasets, use read_dataset(filename).
    If working with recordings from the GUI, use read_aer(filename).
    Returns TD, containing:
        data: a NumPy Record Array with the following named fields
            x: pixel x coordinate, unsigned 16bit int
            y: pixel y coordinate, unsigned 16bit int
            p: polarity value, boolean. False = off event, True = on event
            ts: timestamp in microseconds, unsigned 64bit int
		width: The width of the frame.
        height: The height of the frame. 
    """

    with open(filename, 'rb') as f:
        # Strip header
        header_line = f.readline()
        
        while header_line[0] == '#':
            header_line = f.readline()

        raw_data = np.fromfile(f, dtype=np.uint8)

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

    while buffer_location < len(raw_data):
        num_events = ((raw_data[buffer_location + 3].astype(np.uint32) << 24) + (raw_data[buffer_location + 2].astype(np.uint32) << 16) + (raw_data[buffer_location + 1].astype(np.uint32) << 8) + raw_data[buffer_location])
        buffer_location = buffer_location + 4
        start_time = ((raw_data[buffer_location + 3].astype(np.uint32) << 24) + (raw_data[buffer_location + 2].astype(np.uint32) << 16) + (raw_data[buffer_location + 1].astype(np.uint32) << 8) + raw_data[buffer_location])
        buffer_location = buffer_location + 8

        # Note renaming (since original is a Python built-in):
        evt_type = raw_data[buffer_location:(buffer_location + 8 * num_events):8]
        evt_subtype = raw_data[(buffer_location + 1):(buffer_location + 8 * num_events + 1):8]
        y = raw_data[(buffer_location + 2):(buffer_location + 8 * num_events + 2):8]
        x = ((raw_data[(buffer_location + 5):(buffer_location + 8 * num_events + 5):8].astype(np.uint16) << 8) + (raw_data[(buffer_location + 4):(buffer_location + 8 * num_events + 4):8]))
        ts = ((raw_data[(buffer_location + 7):(buffer_location + 8 * num_events + 7):8].astype(np.uint32) << 8) + (raw_data[(buffer_location + 6):(buffer_location + 8 * num_events + 6):8]))
        buffer_location = buffer_location + num_events * 8
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

    # It appears that the polarity needs to be flipped (when results are compared with Matlab output).
    # Change the polarity: 0 events become 1 events and vice versa.

    TD.data.p = np.abs(TD.data.p - 1)
    
    return TD


def main():
    """Example usage of eventvision"""
    #read in some data
    td, em = read_aer('0001.val')
    #td = read_dataset('trainReduced/0/00002.bin')

    #show the TD events
    td.show_td()

    #extract a region of interest...
    td.data = td.extract_roi([75, 75], [50, 50], True)
    #td.data = td.extract_roi([3, 3], [28, 28])

    #implement a refractory period...
    td.data = td.apply_refraction(0.03)

    #perform some noise filtering...
    td.data = td.filter_td(0.03)

    #show the resulting data
    td.show_td()

    #write the filtered data in a format jAER can understand
    td.write_j_aer('jAERdata.aedat')


    #show the grayscale data
    em.show_em()


    #perform camera calibration
    #first show the calibration pattern on the screen and make some recordings:
    num_squares = 10
    square_size_mm = present_checkerboard(num_squares)

    #state where the recordings are what format they are in
    image_directory = 'path_to_calibration_images'
    image_format = '.bmp'

    #using a scale is useful for visualization
    scale = 4

    #call the calibration function and follow the instructions provided
    ret, mtx, dist, rvecs, tvecs = auto_calibrate(num_squares, square_size_mm, scale, image_directory, image_format)

if __name__ == "__main__":
    main()

print 'Event-based vision module imported'
