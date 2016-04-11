"""
This module contains classes and functions for annotating trackpoints in AER vision data.
"""
import eventvision
import cv2
import numpy as np

class Trackpoints(object):
    """
    Manually annotated trackpoints.
    Trackpoints are stored in a record array, 'Trackpoints', containing
    the following fields:
        data.x  = The x location (in pixels) of each trackpoint (column in frame).
        data.y  = The y location (in pixels) of each trackpoint (row in frame).
        data.ts = The time-stamp of each trackpoint (in microseconds).
    """
    def __init__(self, num_events):
        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('ts', np.uint64)], shape=(num_events))

def annotate(event, x, y, flags, param):
    """
    Callback for function 'annotate_tracks'.
    Detects if mouse position is to be saved as
    a trackpoint.
    """
    if event == cv2.EVENT_MOUSEMOVE:
        global px, py
        px, py = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        global read
        read = 1
        
    if event == cv2.EVENT_LBUTTONUP:
        read = 0

    
def annotate_tracks(TD, frame_length):
    """
    Allows user to manually annotate sequential locations in a TD
    record array containing AER data.

    Takes in:

        TD, containing:
            data: a NumPy Record Array with the following named fields:
                x:  pixel x coordinate, unsigned 16bit int.
                y:  pixel y coordinate, unsigned 16bit int.
                p:  polarity value, boolean. False = off event, True = on event.
                ts: timestamp in microseconds, unsigned 64bit int.
                
        frame_length:   an integer specifying the length of time (in microseconds)
                        in which to accumulate spiking events to be shown in each
                        frame.

    User input:

        'f':    Pressing the 'f' key doubles frame_length (effectively
                doubling the speed at which data is displayed). Use this
                to fast forward through unimportant sections of recordings.
                
        's':    Pressing the 's' key halves frame_length (effectively
                halving the speed at which data is displayed). Use this to
                slow down where accurate tracking is required.
                
        Mouse:  Clicking and holding down the left-hand button of the mouse
                initiates tracking and saves a trackpoint at the location of the
                crosshairs of the mouse at the beginning of each frame.
                Locations outside of the frame are ignored.

    Displays:

        A video is displayed as a sequence of frames of accumulated events.
        On events are red, and off events are blue.

    Returns:

        Trackpoints, containing:
            data: a NumPy Record Array with the following named fields:
                data.x  = The x location (in pixels) of each trackpoint(column in frame).
                data.y  = The y location (in pixels) of each trackpoint (row in frame).
                data.ts = The time-stamp of each trackpoint (in microseconds). 
    """
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', annotate) 

    tmax = np.max(TD.data.ts)
    tmin = np.min(TD.data.ts)

    n_frames = np.ceil((tmax-tmin)/frame_length)

    global px
    global py
    global read

    px = 0
    py = 0
    read = 0

    k = 0
    j = 0
    i = 0

    trackpoints_x = np.ones(n_frames)*-1
    trackpoints_y = np.ones(n_frames)*-1
    trackpoints_ts = np.ones(n_frames)*-1

    while(i < n_frames):
        
        t1_ind = np.min(np.where(TD.data.ts >= j))
        t2_ind = np.max(np.where(TD.data.ts < j + frame_length))
        j = j + frame_length
         
        current_frame = np.zeros((240,304,3), np.uint8)
   
        current_frame[TD.data.y[t1_ind:t2_ind][TD.data.p[t1_ind:t2_ind] == 1], TD.data.x[t1_ind:t2_ind][TD.data.p[t1_ind:t2_ind] == 1], :] = [100, 100, 255] 
        current_frame[TD.data.y[t1_ind:t2_ind][TD.data.p[t1_ind:t2_ind] == 0], TD.data.x[t1_ind:t2_ind][TD.data.p[t1_ind:t2_ind] == 0], :] = [255, 255, 30]

        if(read):
            trackpoints_x[k] = px
            trackpoints_y[k] = py
            trackpoints_ts[k] = t1_ind 
            k = k + 1
            print px, py

        cv2.imshow('frame',current_frame)
        key = cv2.waitKey(5)
    
        if(key == ord('f')):
            print 'Faster'
            frame_length = np.round(frame_length*2)
            tmax = np.max(TD.data.ts)
            tmin = j
            n_frames = np.ceil((tmax-tmin)/frame_length)
    
        if(key == ord('s')):
            print 'Slower'
            frame_length = np.round(frame_length/2)
            tmax = np.max(TD.data.ts)
            tmin = j
            n_frames = np.ceil((tmax-tmin)/frame_length)
  
        i = i + 1

    valid_points = np.where((trackpoints_x >= 0) & (trackpoints_y >= 0) & (trackpoints_x <= 304) & (trackpoints_x <= 240))

    Tracks = Trackpoints(len(valid_points[0]))

    Tracks.data.x = trackpoints_x[valid_points]
    Tracks.data.y = trackpoints_y[valid_points]
    Tracks.data.ts = trackpoints_ts[valid_points]

    plt.plot(Tracks.data.x, Tracks.data.y)
    plt.show()

    return Tracks
