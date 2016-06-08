"""This module contains classes and functions for annotating trackpoints in AER vision data.
"""
import eventvision
import cv2
import os
import numpy as np
import h5py
import pickle
import sys
import timer

class Track(object):
    """Manually annotated trackpoints.
    Trackpoints are stored in an object, Track, containing
    the following fields:
        data: a NumPy record array containing:
            x  = The x location (in pixels) of each trackpoint (column in frame).
            y  = The y location (in pixels) of each trackpoint (row in frame).
            ts = The time-stamp of each trackpoint (in microseconds).
        width: The width of the frame containing the track. Default = 304.
        height: The height of the frame containing the track. Default = 240.
        track_type: The type of object being tracked. Default = 0.
        source_file: The path (at the time of annotation) of the recording for 
                     which this track was created.
        annotated_by: The name of the user who annotated this particular track.
    """
    
    def __init__(self, num_events, width = 304, height = 240, annotated_by = '', source_file = '', track_type = 0 ):

        self.data = np.rec.array(None, dtype=[('x', np.uint16), ('y', np.uint16), ('ts', np.uint64)], shape=(num_events))
        self.width = width
        self.height = height
        self.track_type = track_type
        self.source_file = source_file
        self.annotated_by = annotated_by

    def save(self, filename):
        """Saves the current track object into an hdf5 file. 
        The hdf5 format can be read with the Matlab function h5read.
        Takes in:
            filename:   The path to the file to be written, excluding
                        extensions.
        """
        
        with h5py.File(filename + '.hdf5', 'w') as f:
            x = f.create_dataset('x', data = self.data.x, dtype='i')
            y = f.create_dataset('y', data = self.data.y, dtype='i')
            ts = f.create_dataset('ts', data = self.data.ts, dtype='i')
            track_type = f.create_dataset('track_type', data = self.track_type, dtype='i')
            width = f.create_dataset('width', data = self.width, dtype='i')
            height = f.create_dataset('height', data = self.height, dtype='i')
            annotated_by = f.create_dataset('annotated_by', data = self.annotated_by)
            source_file = f.create_dataset('source_file', data = self.source_file)

    def review(self, TD_object):
        """Displays the TD recording overlaid with the annotated track.
        On events are red, and off events are blue.
        Takes in:
            TD_object:  An Events object (see eventvision module).
        """
        
        cv2.namedWindow('review_frame')

        for i in range(1, len(self.data.ts)):

            current_frame = np.zeros((TD_object.height,TD_object.width,3), np.uint8)
            tmin = self.data.ts[i-1]
            tmax = self.data.ts[i]
            tminind = np.min(np.where(TD_object.data.ts >= tmin))
            tmaxind = np.max(np.where(TD_object.data.ts <= tmax))

            # Populate the current frame with all the events which occur between successive timestamps of the 
            # annotated track events. Track event which was saved at the end of the current frame is shown.
            current_frame[TD_object.data.y[tminind:tmaxind][TD_object.data.p[tminind:tmaxind] == 1], TD_object.data.x[tminind:tmaxind][TD_object.data.p[tminind:tmaxind] == 1], :] = [100, 100, 255] 
            current_frame[TD_object.data.y[tminind:tmaxind][TD_object.data.p[tminind:tmaxind] == 0], TD_object.data.x[tminind:tmaxind][TD_object.data.p[tminind:tmaxind] == 0], :] = [255, 255, 30]
            cv2.circle(current_frame, (self.data.x[i], self.data.y[i]), 10, (0,255,0), 2)
            cv2.imshow('review_frame', current_frame)
            key = cv2.waitKey(1)

        cv2.destroyWindow('review_frame')

def load_track(track_file):
    """Reads a given hdf5 track file and creates a corresponding track object.
    Takes in:
        track_file: The path to the file to be read, including extension.
    """
    
    with h5py.File(track_file,'r') as f:
        track_object = Track(len(f['ts'][:]))
        track_object.data.x = f['x'][:]
        track_object.data.y = f['y'][:]
        track_object.data.ts = f['ts'][:]
        track_object.track_type = f['track_type'][()]
        track_object.width = f['width'][()]
        track_object.height = f['height'][()]
        track_object.source_file = f['source_file'][()]
        track_object.annotated_by = f['annotated_by'][()]

    return track_object
  
def annotate(event, x, y, flags, param):
    """Callback for function 'annotate_tracks'.
    Tracks cursor and detects if mouse position is to be saved as
    a trackpoint. Track points are saved once per frame if the 
    left mouse button is held down.
    """
    
    global is_read
    global px, py
    
    if event == cv2.EVENT_MOUSEMOVE:
        px, py = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        is_read = 1
        
    if event == cv2.EVENT_LBUTTONUP:
        is_read = 0

    
def annotate_tracks(TD, frame_length):
    """Allows the user to manually annotate sequential locations in a TD object.

    Takes in:

        TD, an object containing:
        
            data: a NumPy Record Array with the following named fields:
                x:  pixel x coordinate, unsigned 16bit int.
                y:  pixel y coordinate, unsigned 16bit int.
                p:  polarity value, boolean. False = off event, True = on event.
                ts: timestamp in microseconds, unsigned 64bit int.
            width: The width of the frame.
            height: The height of the frame.
                
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

        A Track object, containing:

            data: a NumPy Record Array with the following named fields:
                data.x  = The x location (in pixels) of each trackpoint(column in frame).
                data.y  = The y location (in pixels) of each trackpoint (row in frame).
                data.ts = The time-stamp of each trackpoint (in microseconds).
            width: The width of the frame containing the track. Default = 304.
            height: The height of the frame containing the track. Default = 240.
            track_type: The type of object tracked. Default = 0.
            source_file: The path (at the time of annotation) to the recording for 
                         which this track was created.
            annotated_by: The name of the user who annotated this particular track.
    """
    
    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', annotate) 
    print 'Click in window and press spacebar to begin tracking'
    
    while(True):
        key1 = cv2.waitKey(5)
        if(key1 == ord(' ')):
            break
        
    tmax = np.max(TD.data.ts)
    tmin = np.min(TD.data.ts)
    n_frames = np.ceil((tmax-tmin)/frame_length)
    # x and y locations of current track point.
    global px 
    global py
    # Indicates that current cursor position should be saved as a track point.
    global is_read
    px = 0
    py = 0
    is_read = 0
    frame_num = 0 # Current frame number.
    frame_ts = 0 # The start time of the current frame (in microseconds).
    trackpoint_ind = 0 # The index in the trackpoint arrays where the current trackpoint should be saved.

    # Initialise numpy arrays to hold 16x the initial estimate of the
    # number of frames. If the user were to slow the frame rate by half
    # twice at the beginning, this would require 4x the initial
    # estimate of the number of frames.
    trackpoints_x = np.ones(n_frames*16)*-1
    trackpoints_y = np.ones(n_frames*16)*-1
    trackpoints_ts = np.ones(n_frames*16)*-1

    while(frame_num < n_frames):
        t1_ind = np.min(np.where(TD.data.ts >= frame_ts))
        t2_ind = np.min(np.where(TD.data.ts > frame_ts + frame_length))
        frame_ts = frame_ts + frame_length
        current_frame = np.zeros((TD.height,TD.width,3), np.uint8)
        current_frame[TD.data.y[t1_ind:t2_ind][TD.data.p[t1_ind:t2_ind] == 1], TD.data.x[t1_ind:t2_ind][TD.data.p[t1_ind:t2_ind] == 1], :] = [100, 100, 255] 
        current_frame[TD.data.y[t1_ind:t2_ind][TD.data.p[t1_ind:t2_ind] == 0], TD.data.x[t1_ind:t2_ind][TD.data.p[t1_ind:t2_ind] == 0], :] = [255, 255, 30]
        cv2.imshow('frame',current_frame)
        key = cv2.waitKey(5)
        
        if(is_read):
            trackpoints_x[trackpoint_ind] = px
            trackpoints_y[trackpoint_ind] = py
            trackpoints_ts[trackpoint_ind] = TD.data.ts[t2_ind] # Track point time stamp is the time at the end of the current frame. 
            trackpoint_ind += 1
            if((px > 0) & (py > 0) & (px < TD.width) & (py < TD.height)):
                print px, py
    
        if(key == ord('f')):
            print 'Faster'
            frame_length = np.round(frame_length*2)
            tmax = np.max(TD.data.ts)
            tmin = frame_ts
            n_frames = frame_num + np.ceil((tmax-tmin)/frame_length)
    
        if(key == ord('s')):
            print 'Slower'
            frame_length = np.round(frame_length/2)
            tmax = np.max(TD.data.ts)
            tmin = frame_ts
            n_frames = frame_num + np.ceil((tmax-tmin)/frame_length)
  
        frame_num += 1

    valid_points = np.where((trackpoints_x >= 0) & (trackpoints_y >= 0) & (trackpoints_x <= TD.width) & (trackpoints_y <= TD.height))
    Trackpoints = Track(len(valid_points[0]), width=TD.width, height=TD.height) 
    Trackpoints.data.x = trackpoints_x[valid_points]
    Trackpoints.data.y = trackpoints_y[valid_points]
    Trackpoints.data.ts = trackpoints_ts[valid_points]
    return Trackpoints


class Annotation_State(object):
    """Holds information about which files have been annotated, and which still need to be annotated.
    Contains:
        source_files:   A list of strings, each element of which is 
                        the file path (at the time of annotation)
                        to a file that has been completely annotated.
        output_subfolders:  A list of strings, each element of which 
                            is the output folder in which the track 
                            files associated with the corresponding 
                            file name in source_files are to be saved.
        is_complete:    A list of boolean values which are true if the 
                        corresponding file in source_files has been 
                        completely annotated, and false if not. 
    """
    
    def __init__(self, source_files, output_subfolders, is_complete ):

        self.source_files = source_files
        self.output_subfolders = output_subfolders
        self.is_complete = is_complete

    def save_state(self, filename):
        """Save the current annotation state.
        """    
        with open(filename, 'wb') as output_state_file:
            pickle.dump(self, output_state_file, pickle.HIGHEST_PROTOCOL)
    
    def update_state(self, input_folder, output_folder):
        """Updates the state file.
        NOTE: Changing the paths of any existing files in the 
        directory will result in them being considered as new
        files in need of annotation.
        """
        
        next_folder_number = len(self.output_subfolders) # Named from 0, counted normally here.
        
        for root, dirs, files in os.walk(input_folder):
            for inner_file in files:
                # Check if the current file path has been allocated an output folder.
                if os.path.join(root,inner_file) in self.source_files:
                    continue
                
                # If not, create a new folder for it and update the annotation state.               
                new_input_file = os.path.join(root, inner_file)
                self.output_subfolders.append(os.path.join(output_folder, str(next_folder_number)))
                self.source_files.append(new_input_file)
                self.is_complete.append(False)
                os.makedirs(os.path.join(output_folder, str(next_folder_number)))

                print 'Added file ' + new_input_file
                next_folder_number += 1

        print 'State file updated.'


def load_state(filename):
    """Load the annotation state stored in filename.
    """  
    with open(filename, 'rb') as input_state_file:
        loaded_state = pickle.load(input_state_file) 
    return loaded_state
    

def link_files(input_folder, output_folder, state_file_name):
    """Creates the directory structure into which track files are to be saved,
    and also creates an annotation state file which holds information 
    about which files still need to be annotated (Annotation_State).
    
    Each annotated file is allocated a folder, the name of which is 
    given by a sequentially incremented integer. Each folder contains 
    the track files associated with that particular file.
    
    Takes in:
        input_folder:   The file path to the directory containing all the 
                        TD files to be annotated. This directory may contain
                        a combination of subdirectories.
        output_folder:  The directory which will contain the folders associated
                        with the files in the input_folder directory. This 
                        directory should be empty to avoid conflicts.
        state_file_name: The name of the state file which is to hold information
                         about which files have been completely annotated. 
    """
    
    print 'Creating output directory structure and state file...'
    output_subfolders = []
    source_files = []
    is_complete = []
    output_folder_number = 0
    annotation_state = Annotation_State(source_files, output_subfolders, is_complete)
    annotation_state.update_state(input_folder, output_folder)
    annotation_state.save_state(os.path.join(output_folder, state_file_name))
    return annotation_state
                    
def load_TD(file_path):
    """Returns the TD file given by file_path.
    Can be either .val files generated by the windows GUI,
    or .bin files generated by the linux C++ framework.
    """
    
    if(file_path.endswith('.val')):
        print 'Reading event file: ' + file_path
        TD = eventvision.read_aer(file_path)[0]
        return TD
        
    if(file_path.endswith('.bin')):
        print 'Reading event file: ' + file_path
        TD = eventvision.read_bin_linux(file_path)
        return TD

def add_track(folder, track_num, TD, annotated_by, source_file):
    """Seeks user input to create and save a track file.
    
    Takes in:
        folder: The file path to the folder in which the current track file
                is to be saved.
        track_num:  The number of the current track associated with a 
                    particular input file. 
        TD: The original TD object for which the current track is being created.
        annotated_by: The name of the user who is annotating this track.
        source_file: The file path to the file from which the TD object was read.                 
    """
    
    print 'Annotating track number ' + str(track_num)
    track = annotate_tracks(TD, 100000)
    is_under_review = True

    while(is_under_review):
        redo = raw_input('Press [r] to re-annotate this track, [v] to view the track or [s] to save and continue:')

        if((redo == 'r') | (redo == 'R')):
            track = annotate_tracks(TD, 100000)

        if((redo == 'v') | (redo == 'V')):
            track.review(TD)

        if((redo == 's') | (redo == 'S')): 
            is_under_review = False

    while(True):
        try:
            track_type = int(raw_input('Please enter the track type (integer value):'))
        except ValueError:
            print 'Invalid - track type must be an integer.'
        else:
            break

    track.track_type = track_type
    track.annotated_by = annotated_by
    track.source_file = source_file
    track.save(os.path.join(folder,'Track_' + str(track_num)))
    print 'Track saved.'

def main():
    """Program to annotate the tracks of moving features in TD recordings.
    
    An input folder is required, which contains the recordings to be annotated.
    This folder may contain folders and subfolders as well. An output folder is 
    required, which must be empty before first run (see the method link_files).
    
    Note that there are limitations: 
    
    If you wish to re-annotate a file that has been previously marked as 
    completely annotated, you will need to do the following:
    
        1.  Call annotate.load_state to read the current annotation_state
            file. Examine the fields of the returned 
            Annotation_State object and set the is_complete flag (corresponding 
            to the file in question) to zero. Save the Annotation_State object 
            in place of the original one using Annotation_State.save  
    
        2.  Manually delete the track files, contained within the
            folder corresponding to the file that was previously 
            flagged as completely annotated, but which must now be 
            re-annotated. 
    
    Another important limitation:
    
    Changing the paths of any existing files in the input directory 
    will result in them being considered as new files in need of annotation.
    """

    input_folder = 'C:\Input_Recordings'
    output_folder = 'C:\Output_Tracks'
    state_file_name = 'annotation_state.pkl'

    while(not os.path.isdir(input_folder)):
        print 'Unable to find input directory ' + input_folder

        # Pause before exit in case program not called from command line.
        while(True):
            should_exit = raw_input('Press [x] to exit.')
            if(should_exit == 'x'):
                sys.exit()

    while(not os.path.isdir(output_folder)):
        print 'Unable to find output directory ' + output_folder

        # Pause before exit in case program not called from command line.
        while(True):
            should_exit = raw_input('Press [x] to exit.')
            if(should_exit == 'x'):
                sys.exit()

    # Look for state file.
    # If state file does not exist, create it as well as the output
    # directory structure which will hold the track files.
    if(os.path.isfile(os.path.join(output_folder, state_file_name))):
        print 'State file found.'
        annotation_state = load_state(os.path.join(output_folder, state_file_name))
        # Update in case any new files added to input directory:
        annotation_state.update_state(input_folder, output_folder)
    else:
        annotation_state = link_files(input_folder, output_folder, state_file_name)

    # Iterate over all the files which have not been flagged as completely annotated.
    incomplete_indices = [i for i, individual_status in enumerate(annotation_state.is_complete) if not individual_status]

    if(len(incomplete_indices) > 0):
        annotated_by = raw_input('Please enter your name: ') # Only ask if there are still files to annotate.

    for i in range(len(incomplete_indices)):
        num_track_files = 0

        # Check if there are any existing track files, and update
        # num_track_files accordingly.
        incomplete_file_ind = incomplete_indices[i]

        for track_file in os.listdir(annotation_state.output_subfolders[incomplete_file_ind]):  
            if(track_file.endswith('.hdf5')):
                num_track_files += 1 # Track files are named from zero, but counted normally here.

        # Load TD from current source file
        TD = load_TD(annotation_state.source_files[incomplete_file_ind])

        # If track files already exist, give the user the option to 
        # review them.
        if(num_track_files > 0):              
            print 'Track files have been found.'
            is_review_tracks = True
            
            while(is_review_tracks):
                check_tracks = raw_input('Press [v] to view existing tracks or [c] continue:')

                if((check_tracks == 'v') | (check_tracks == 'V')):
                    for track_file in os.listdir(annotation_state.output_subfolders[incomplete_file_ind]):
                        current_track = load_track(os.path.join(annotation_state.output_subfolders[incomplete_file_ind],track_file))
                        print 'Reviewing track ' + track_file
                        current_track.review(TD)

                if((check_tracks == 'c') | (check_tracks == 'C')):
                    is_review_tracks = False

        is_add_track = True

        while(is_add_track):   
            check_tracks = raw_input('Press [a] to add a track, [c] to mark this file as complete and continue to the next file, or [x] to exit.')  

            if((check_tracks == 'a') | (check_tracks == 'A')):
                add_track(annotation_state.output_subfolders[incomplete_file_ind], num_track_files, TD, annotated_by, annotation_state.source_files[incomplete_file_ind])
                num_track_files += 1
                
            if((check_tracks == 'c') | (check_tracks == 'C')):
                annotation_state.is_complete[incomplete_file_ind] = True
                annotation_state.save_state(os.path.join(output_folder, state_file_name)) # Flag current file as completely annotated.
                is_add_track = False

            if((check_tracks == 'x') | (check_tracks == 'X')):
                sys.exit() # Any tracks have been saved already so can exit here.

    print 'There are no more files to annotate in the input directory.'

    # Pause before exit in case program not called from command line.
    while(True):
        should_exit = raw_input('Press [x] to exit.')
        if(should_exit == 'x'):
            break


if __name__ == '__main__':
    main()
