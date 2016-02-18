# event-Python
Python code for event based vision

This is a first attempt at Python code for handling AER vision data. The code is written in Anaconda Python 2.7, and makes extensive use of OpenCV, and numpy.

Functions included:
	
	TD, EM = read_aer(filename):
		Reads in the ATIS file specified by 'filename' and returns the TD and EM events. This only works for ATIS recordings directly from the GUI. If you are working with the N-MNIST or N-CALTECH101 datasets, use  read_dataset(filename) instead
		
	TD = read_dataset(filename):
		Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'
	
	write2jAER(td_events, filename):
		writes the td events in 'td_events' to a file specified by 'filename' which is compatible with the jAER framework. To view these events in jAER, make sure to select the DAVIS640 sensor.
		
	show_td(TD):
		displays the TD events (change detection ATIS or DVS events)
		
	show_em(EM):
		displays the EM events (grayscale ATIS events)
	
	event_ROI = extract_roi(td_events, top_left, size):
		extracts td_events which fall into a rectangular region of interest with top left corner at 'top_left' and size 'size'
	
	events_out = filter_td(td_events, us_time):
		implements a background activity filter on 'td_events', where only events which are supported by a neighbouring event within 'us_time' microseconds are allowed through the filter.
	
	events_out = implement_refraction(td_events, us_time):
		implements a refractory period of 'us_time' microseconds on all events in 'td_events'. Any time an valid event is received, any further events from the same pixel within 'us_time' microseconds are considered invalid
		
	events_out = extract_indices(events, logical_indices):
		will take in the event structure 'events' and remove any events corresponding to locations where logical_indices = 0
	
	events_out = sort_order(events):
		will look through the struct events, and sort all events by the field 'ts'. In other words, it will ensure events_out.ts is monotonically increasing, which is useful when combining events from multiple recordings.
	
	squareSize_mm = present_checkerboard(num_squares):
		Presents a checkerboard pattern of size num_squares*num_squares on the screen. The function will automatically detect the screen size in pixels and assume a resolution of 96 dpi to provide the square size in mm.
			
	
	ret, mtx, dist, rvecs, tvecs = auto_calibrate(num_squares, squareSize_mm, scale, image_directory, image_format):
		Will read in images of extension 'image_format' from the directory 'image_directory' for calibration.
		Each image should contain a checkerboard with 'num_squares'*'num_squares' squares, each of size 'squareSize_mm'
		'scale' is an optional argument to rescale images before calibration because ATIS/DVS have very low resolution and calibration algorithms are used to handling larger images (use 4)
		