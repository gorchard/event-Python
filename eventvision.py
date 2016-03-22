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
    Temporal Difference events. The arrays are of equal size
    x: Numpy array of pixel x coordinates
    y: Numpy array of pixel y coordinates
    p: Numpy array of polarity values. 1 means off, 2 means on
    ts: Numpy array of timestamps in microseconds
    """
    def __init__(self):
        self.x = None
        self.y = None
        self.p = None
        self.ts = None

    def show_em(self):
        """
        Displays the EM events (grayscale ATIS events)
        """
        max_x = max(self.x) + 1
        max_y = max(self.y) + 1
        thr_valid = np.zeros((max_y, max_x))
        thr_l = np.zeros((max_y, max_x))
        thr_h = np.zeros((max_y, max_x))

        frame_length = 24e3
        t_max = len(self.ts) - 1
        frame_end = self.ts[1] + frame_length
        i = 0
        while i < t_max:
            while (self.ts[i] < frame_end) and (i < t_max):
                if self.p[i] == 0:
                    thr_valid[self.y[i], self.x[i]] = 1
                    thr_l[self.y[i], self.x[i]] = self.ts[i]
                else:
                    if thr_valid[self.y[i], self.x[i]] == 1:
                        thr_valid[self.y[i], self.x[i]] = 0
                        thr_h[self.y[i], self.x[i]] = self.ts[i] - thr_l[self.y[i], self.x[i]]
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
        """
        Displays the TD events (change detection ATIS or DVS events)
        waitDelay: milliseconds
        """
        max_x = max(self.x) + 1
        max_y = max(self.y) + 1

        frame_length = 24e3
        t_max = len(self.ts) - 1
        frame_end = self.ts[1] + frame_length
        i = 0
        while i < t_max:
            td_img = 0.5 * np.ones((max_y, max_x))
            while (self.ts[i] < frame_end) and (i < t_max):
                td_img[self.y[i], self.x[i]] = self.p[i]
                i = i + 1

            img = 255 * td_img
            img = img.astype('uint8')
            cv2.imshow('img', img)
            cv2.waitKey(wait_delay)
            frame_end = frame_end + frame_length

        cv2.destroyAllWindows()
        return

    def filter_td(self, us_time):
        """
        Apply a background activity filter on the events, such that only events which are
        correlated with a neighbouring event within 'us_time' microseconds will be allowed
        through the filter.
        us_time: microseconds
        """
        max_x = max(self.x)
        max_y = max(self.y)
        t0 = np.ones((max_x + 1, max_y + 1)) - us_time - 1
        x_prev = 0
        y_prev = 0
        p_prev = 0

        valid_indices = np.ones(len(self.ts), np.bool)

        for i in range(len(self.ts)):
            if x_prev != self.x[i] | y_prev != self.y[i] | p_prev != self.p[i]:
                t0[self.x[i], self.y[i]] = -us_time
                min_x_sub = max(0, self.x[i] - 1)
                max_x_sub = min(max_x, self.x[i] + 1)
                min_y_sub = max(0, self.y[i] - 1)
                max_y_sub = min(max_y, self.y[i] + 1)

                t0_temp = t0[min_x_sub:(max_x_sub + 1), min_y_sub:(max_y_sub + 1)]

                if min(self.ts[i] - t0_temp.reshape(-1, 1)) > us_time:
                    valid_indices[i] = 0

            t0[self.x[i], self.y[i]] = self.ts[i]
            x_prev = self.x[i]
            y_prev = self.y[i]
            p_prev = self.p[i]

        return extract_indices(self, valid_indices.astype('bool'))

    def sort_order(self):
        """
        Will look through the struct events, and sort all events by the field 'ts'.
        In other words, it will ensure events_out.ts is monotonically increasing,
        which is useful when combining events from multiple recordings.
        """
        print 'the function sort_order has not yet been thoroughly tested'
        inds = self.ts.argsort()
        events_out = self
        for i in events_out.__dict__.keys():
            temp = getattr(events_out, i)
            temp = temp[inds]
            setattr(events_out, i, temp)
        return events_out

    def extract_roi(self, top_left, size, is_normalize=False):
        """
        Extracts td_events which fall into a rectangular region of interest with
        top left corner at 'top_left' and size 'size'
        top_left: [x: int, y: int]
        size: [width, height]
        is_normalize: bool. If true, x and y values will be normalized to the cropped region
        """
        #TODO(cedricseah): implement normalization
        valid_indices = (self.x >= top_left[0]) \
        & (self.y >= top_left[1]) \
        & (self.x < (size[0] + top_left[0])) \
        & (self.y < (top_left[1] + size[1]))
        return extract_indices(self, valid_indices.astype('bool'))

    def apply_refraction(self, us_time):
        """
        Apply refraction ala the biological neuron behaviour of a refractory (enforced rest)
        period before a neuron is able to spike
        us_time: time in microseconds
        """
        max_x = max(self.x)
        max_y = max(self.y)
        t0 = np.ones((max_x + 1, max_y + 1)) - us_time - 1

        valid_indices = np.ones(len(self.ts), np.bool)

        for i in range(len(self.ts)):
            if (self.ts[i] - t0[self.x[i], self.y[i]]) < us_time:
                valid_indices[i] = 0
            else:
                valid_indices[i] = 1
                t0[self.x[i], self.y[i]] = self.ts[i]

        return extract_indices(self, valid_indices.astype('bool'))

    def write_j_aer(self, filename):
        """
        writes the td events in 'td_events' to a file specified by 'filename'
        which is compatible with the jAER framework.
        To view these events in jAER, make sure to select the DAVIS640 sensor.
        """
        import time
        y = 479 - self.y
        #y = td_events.y
        y_shift = 22 + 32

        x = 639 - self.x
        #x = td_events.x
        x_shift = 12 + 32

        p = self.p
        p_shift = 11 + 32

        ts_shift = 0

        y_final = y.astype(dtype=np.uint64) << y_shift
        x_final = x.astype(dtype=np.uint64) << x_shift
        p_final = p.astype(dtype=np.uint64) << p_shift
        ts_final = self.ts.astype(dtype=np.uint64) << ts_shift
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

def extract_indices(events, logical_indices):
    """
    Take in the event structure 'events' and remove any events corresponding to locations where logical_indices = 0
    """
    events_out = Events()
    if sum(logical_indices) > 0:
        events_out = events
        for i in events_out.__dict__.keys():
            temp = getattr(events_out, i)
            temp = temp[logical_indices]
            setattr(events_out, i, temp)
    return events_out

def read_aer(filename):
    """
    Reads in the ATIS file specified by 'filename' and returns the TD and EM events.
    This only works for ATIS recordings directly from the GUI.
    If you are working with the N-MNIST or N-CALTECH101 datasets, use read_dataset(filename) instead
    """
    td = Events()
    em = Events()
    all_events = Events()
    f = open(filename, 'rb')
    #raw_data = np.fromfile(f, dtype=np.uint8, count=-1)
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint16(raw_data)

    all_events.y = raw_data[3::4]
    all_events.x = ((raw_data[1::4] & 32) << 3) | raw_data[2::4] #bit 5
    all_events.p = (raw_data[1::4] & 128) >> 7 #bit 7
    all_events.ts = raw_data[0::4] | ((raw_data[1::4] & 31) << 8) # bit 4 downto 0
    dtype = (raw_data[1::4] & 64) >> 6 #bit 6
    all_events.ts = all_events.ts.astype('uint')

    time_offset = 0
    for i in range(len(all_events.ts)):
        if (all_events.y[i] == 240) and (all_events.x[i] == 305):
            dtype[i] = 2
            time_offset = time_offset + 2 ** 13
        else:
            all_events.ts[i] = all_events.ts[i] + time_offset

    valid_indices = dtype != 2
    all_events = extract_indices(all_events, valid_indices)
    dtype = dtype[valid_indices]

    valid_indices = dtype == 1
    em = extract_indices(all_events, valid_indices)
    valid_indices = dtype == 0
    td = extract_indices(all_events, valid_indices)

    return td, em

def read_dataset(filename):
    """
    Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'
    """
    td = Events()
    f = open(filename, 'rb')
    raw_data = np.fromfile(f, dtype=np.uint8)
    f.close()
    raw_data = np.uint32(raw_data)

    td.x = raw_data[0::5]
    td.y = raw_data[1::5]
    td.p = (raw_data[2::5] & 128) >> 7 #bit 7
    td.ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
    dtype = np.zeros(len(td.ts), dtype=np.uint8)
    time_offset = 0
    for i in range(len(td.ts)):
        if (td.y[i] == 240) and (td.x[i] == 305):
            dtype[i] = 2
            time_offset = time_offset + 2 ** 13
        else:
            td.ts[i] = td.ts[i] + time_offset
    valid_indices = dtype != 2
    td = extract_indices(td, valid_indices)
    return td

def main():
    """Example usage of eventvision"""
    #read in some data
    td, em = ev.read_aer('0000.val')

    #show the TD events
    td.show_td(100)

    #extract a region of interest...
    #note this will also edit the event struct 'TD'
    td2 = td.extract_roi([100, 100], [50, 50])

    #implement a refractory period...
    #note this will also edit the event #struct 'TD2'
    td3 = td2.apply_refraction(0.03)

    #perform some noise filtering...
    #note this will also edit the event struct 'TD3'
    td4 = td3.filter_td(0.03)

    #show the resulting data
    td4.show_td(100)

    #write the filtered data in a format jAER can understand
    td4.write_j_aer('jAERdata.aedat')


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
