"""This module contains classes, functions and a program (main) for reading neuromorphic datasets, processing the data and then saving into a caffe lmdb"""
import glob
import math
import os
import cv2
import numpy as np
from scipy import ndimage
from scipy.io import savemat
import caffe_lmdb
import datum_pb2
import eventvision as ev
import timer

def stabilize(td):
    """Compensate for motion of the ATIS sensor during recording of the Neuromorphic datasets
    Applies to the N-MNIST and N-Caltech101 datasets.
    The image motion is originally induced by egorotation of the ATIS sensor
    td: eventvision.Events
    """
    assert isinstance(td, ev.Events)

    def correct_saccade1(data):
        data.x -= np.rint(3.5 * data.ts / 105e3).astype(np.uint16)
        data.y -= np.rint(7 * data.ts / 105e3).astype(np.uint16)
        return data

    def correct_saccade2(data):
        data.x -= np.rint(3.5 + 3.5 * (data.ts - 105e3) / 105e3).astype(np.uint16)
        data.y -= np.rint(7 - 7 * (data.ts - 105e3) / 105e3).astype(np.uint16)
        return data

    def correct_saccade3(data):
        data.x -= np.rint(7 - 7 * (data.ts - 210e3) / 105e3).astype(np.uint16)
        return data

    copy = np.piecewise(td.data,\
        [td.data.ts <= 105e3, (td.data.ts > 105e3) & (td.data.ts <= 210e3), (td.data.ts > 210e3)],\
        [correct_saccade1, correct_saccade2, correct_saccade3]).view(np.recarray)

    # after saccades, we might end up with invalid x and y values, have to
    # correct these
    x_vals = copy.x
    y_vals = copy.y
    copy.x = np.piecewise(x_vals,\
        [x_vals >= 65000, (x_vals < 65000) & (x_vals >= td.width), x_vals < td.width],\
        [0, td.width - 1, lambda x: x])
    copy.y = np.piecewise(y_vals,\
        [y_vals >= 65000, (y_vals < 65000) & (y_vals >= td.height), y_vals < td.height],\
        [0, td.height - 1, lambda y: y])

    return copy

def apply_tracking1(td, alpha=0.98, threshold=-1):
    """Alternative to stabilization. Compensate for motion of a single "object" by tracking its movement
    The concept is fairly simple:
    0: The tracker starts at the center of the event recording
    1: For each incoming event, calculate its distance to the tracker.
    2: If the distance is less than a threshold then update the tracker location using
    3: tracker_location = tracker_location*alpha + event_location*(1-alpha)
        
    You may find the tracker is quite erratic because it moves with every incoming event. It may be a good idea to smooth the motion somewhat which would be another step.%

    td: eventvision.Events
    alpha: alpha is a number between 0 and 1. Typically quite high. Default 0.9
    threshold: distance in pixels for the tracker to be updated. Default = 0.5 * height of td
    """
    assert(alpha >= 0)
    assert(alpha <= 1)
    mix = 1 - alpha

    #with timer.Timer() as my_timer:
    track_x = center_x = td.width / 2
    track_y = center_y = td.height / 2
    threshold_sq = math.floor(center_y**2)

    if (threshold > 0):
        threshold_sq = math.floor(threshold**2)
    
    copy = np.copy(td.data).view(np.recarray)
    for i in range(copy.size):
        datum = copy[i]
        y_val = datum.y
        x_val = datum.x
        distance = (track_x - x_val)**2 + (track_y - y_val)**2

        if (distance <= threshold_sq):
            track_x = track_x * alpha + x_val * mix
            track_y = track_y * alpha + y_val * mix
  
        datum.y = round(y_val - track_y + center_y)
        datum.x = round(x_val - track_x + center_x)
    #print 'Applying tracker took %s seconds' % my_timer.secs
    # remove the events that are out of bounds
    return copy[(copy.x >= 0) & (copy.y >= 0) & (copy.x < td.width) & (copy.y < td.height)]
    

def apply_tracking2(td, num_spikes = 20, alpha=0.5, threshold=-1):
    """Work as well as tracking 1, but faster
    Alternative to stabilization. Compensate for motion of a single "object" by tracking its movement every "num_spike" spikes
    1: Filter spikes far from the tracker
    2: tracker location = tracker_location*alpha + filtered_average*(1-alpha)

    td: eventvision.Events
    num_spikes: number of spike to process in a batch
    alpha: between 0 and 1. How much weight to give to previous tracker value
    threshold: distance in number of pixels. Spikes within this distance shall be included in the tracker location computation. It is half of the event height by default.
    """
    assert(alpha >= 0)
    assert(alpha <= 1)
    mix = 1 - alpha
    track_x = center_x = float(td.width / 2)
    track_y = center_y = float(td.height / 2)
    threshold_sq = math.floor(center_y**2)

    if (threshold > 0):
        threshold_sq = math.floor(threshold**2)

    copy = np.copy(td.data).view(np.recarray)
    offset_x_arr = np.zeros(copy.size, np.float32)
    offset_y_arr = np.zeros(copy.size, np.float32)

    for spike_index in range(0, copy.size, num_spikes):
        frame_data = copy[spike_index:spike_index + num_spikes]
        distances = ((frame_data.x - track_x) ** 2) + ((frame_data.y - track_y) ** 2)
        valid_data = frame_data[distances < threshold_sq]

        if valid_data.size > 0:
            x_avg =  float(np.sum(valid_data.x)) / valid_data.size
            y_avg =  float(np.sum(valid_data.y)) / valid_data.size
            track_x = (track_x * alpha) + (x_avg * mix)
            track_y = (track_y * alpha) + (y_avg * mix)
            offset_x = int(round(center_x - track_x))
            offset_y = int(round(center_y - track_y))
            offset_x_arr[spike_index:spike_index + num_spikes] = offset_x
            offset_y_arr[spike_index:spike_index + num_spikes] = offset_y

    offset_x_arr[spike_index:] = offset_x
    offset_y_arr[spike_index:] = offset_y
    copy.x = (copy.x + offset_x_arr).astype(np.uint8)
    copy.y = (copy.y + offset_y_arr).astype(np.uint8)
    # remove the events that are out of bounds
    return copy[(copy.x >= 0) & (copy.y >= 0) & (copy.x < td.width) & (copy.y < td.height)]


def apply_tracking3(td, time_us=1000, alpha=0.7, threshold=-1):
    """Work as well as tracking 1, but faster
    Alternative to stabilization. Compensate for motion of a single "object" by tracking its movement every time_us microseconds
    1: Filter spikes far from the tracker
    2: tracker location = tracker_location*alpha + filtered_average*(1-alpha)

    td: eventvision.Events
    time_us: batch spikes by time (in microseconds). Default = 1 millisecond
    alpha: between 0 and 1. How much weight to give to previous tracker value
    threshold: distance in number of pixels. Spikes within this distance shall be included in the tracker location computation. It is half of the event height by default.
    """
    assert(alpha >= 0)
    assert(alpha <= 1)
    mix = 1 - alpha
    track_x = center_x = float(td.width / 2)
    track_y = center_y = float(td.height / 2)
    threshold_sq = math.floor(center_y**2)

    if (threshold > 0):
        threshold_sq = math.floor(threshold**2)

    copy = np.copy(td.data).view(np.recarray)
    offset_x = offset_y = 0
    offset_x_arr = np.zeros(copy.size, np.float32)
    offset_y_arr = np.zeros(copy.size, np.float32)
    offset_index = 0 # used to keep track of the offsets we are writing to

    for start_ts in range(copy[0].ts, copy[-1].ts, time_us):
        end_ts = start_ts + time_us
        frame_data = copy[(copy.ts >= start_ts) & (copy.ts < end_ts)]
        distances = ((frame_data.x - track_x) ** 2) + ((frame_data.y - track_y) ** 2)
        valid_data = frame_data[distances < threshold_sq]

        if valid_data.size > 0:
            x_avg =  float(np.sum(valid_data.x)) / valid_data.size
            y_avg =  float(np.sum(valid_data.y)) / valid_data.size
            track_x = (track_x * alpha) + (x_avg * mix)
            track_y = (track_y * alpha) + (y_avg * mix)

            offset_x = int(round(center_x - track_x))
            offset_y = int(round(center_y - track_y))
            offset_x_arr[offset_index:offset_index + frame_data.size] = offset_x
            offset_y_arr[offset_index:offset_index + frame_data.size] = offset_y
            offset_index += frame_data.size

    offset_x_arr[offset_index:] = offset_x
    offset_y_arr[offset_index:] = offset_y
    copy.x = (copy.x + offset_x_arr).astype(np.uint8)
    copy.y = (copy.y + offset_y_arr).astype(np.uint8)
    # remove the events that are out of bounds
    return copy[(copy.x >= 0) & (copy.y >= 0) & (copy.x < td.width) & (copy.y < td.height)]

def make_td_images(td, num_spikes, step_factor=1):
    """Generate set of images from the Temporal Difference (td) events by reading a number of unique spikes
    td is read from a binary file (refer to eventvision.Readxxx functions)
    td: eventvision.Events
    num_spikes: number of unique spikes to accumulate before generating an image
    step_factor: proportional amount to shift before generating the next image.
        1 would result in no overlapping events between images
        0.6 would result in the next image overlapping with 40% of the previous image

    returns array of images
    """
    assert isinstance(td, ev.Events)
    assert isinstance(num_spikes, (int, long))
    assert num_spikes > 0
    assert step_factor > 0

    #with timer.Timer() as my_timer:
    event_offset = 0
    images = []
    while event_offset + num_spikes < td.data.size:
        image = np.zeros((td.height, td.width), dtype=np.uint8)
        unique_spike_count = 0
        index_ptr = event_offset
        while (unique_spike_count < num_spikes) & (index_ptr < td.data.size):
            event = td.data[index_ptr]
            y = event.y
            x = event.x
            if image[y, x] == 0:
                image[y, x] = 255
                unique_spike_count += 1

            index_ptr += 1

        #cv2.imshow('img', img)
        #cv2.waitKey(1)
        if unique_spike_count < num_spikes:
            break

        images.append(image)

        #offset next image
        total_spikes_traversed = index_ptr - event_offset
        event_offset += math.floor(total_spikes_traversed * step_factor) + 1
    #print 'Making images out of bin file took %s seconds' % my_timer.secs

    return images

def make_td_probability_image(td, skip_steps=0, is_normalize = False):
    """Generate image from the Temporal Difference (td) events with each pixel value indicating probability of a spike within a 1 millisecond time step. 0 = 0%. 255 = 100%
    td is read from a binary file (refer to eventvision.Readxxx functions)
    td: eventvision.Events
    skip_steps: number of time steps to skip (to allow tracker to init to a more correct position)
    is_normalize: True to make the images more obvious (by scaling max probability to pixel value 255)
    """
    assert isinstance(td, ev.Events)

    #with timer.Timer() as my_timer:
    event_offset = 0
    combined_image = np.zeros((td.height, td.width), np.float32)
    offset_ts = td.data[0].ts + (skip_steps * 1000)
    num_time_steps = math.floor((td.data[-1].ts - offset_ts) / 1000)
    
    current_frame = np.zeros((td.height, td.width), np.uint8)
    for start_ts in range(int(offset_ts), td.data[-1].ts, 1000):
        end_ts = start_ts + 1000
        frame_data = td.data[(td.data.ts >= start_ts) & (td.data.ts < end_ts)]
        current_frame.fill(0)
        current_frame[frame_data.y, frame_data.x] = 1 
        combined_image = combined_image + current_frame        

    #print 'Making image out of bin file took %s seconds' % my_timer.secs
    if (is_normalize):
        combined_image = (combined_image / np.max(combined_image))
    else:
        combined_image = (combined_image / num_time_steps)

    return combined_image

def prepare_n_mnist(filename, is_filter, num_spikes, step_factor=1):
    """Creates images from the specified n mnist recording
    filename: path to the recording
    is_filter: True if median filtering should be applied to the constructed image
    num_spikes: number of unique spikes per image
    step_factor: proportional amount to shift before generating the next image
        1 would result in no overlapping events between images
        0.6 would result in the next image overlapping with 40% of the previous image
    returns: list of images, where each image is a 2d numpy array (height, width)
    """
    td = ev.read_dataset(filename)
    #td.show_td(100)
    td.data = stabilize(td)
    td.data = td.extract_roi([3, 3], [28, 28], True)
    images = make_td_images(td, num_spikes, step_factor)

    if is_filter:
        images = ndimage.median_filter(images, 3)

    #for image in images:
    #    cv2.imshow('img', image)
    #    cv2.waitKey(70)
    return images

def prepare_n_mnist_continuous(filename, is_filter, is_normalize=False):
    """Creates image with pixel values indicating probability of a spike
    filename: path to the recording
    is_filter: True if median filtering should be applied to the constructed image
    is_normalize: If True, the probabilities will be normalized to make the image more obvious
    returns: image (2d numpy array (height, width))
    """
    td = ev.read_dataset(filename)
    #td.show_td(100)
    td.data = stabilize(td)
    td.data = td.extract_roi([0, 0], [28, 28], True)
    #td.data = apply_tracking1(td)
    #td.data = apply_tracking2(td)
    #td.data = apply_tracking3(td)
    #td.data = td.extract_roi([3, 3], [28, 28], True)
    image = make_td_probability_image(td, 9, is_normalize)

    if is_filter:
        image = ndimage.median_filter(image, 3)

    #cv2.imshow('img', image)
    #cv2.waitKey(1)
    return image

def add_images_to_dataset(image_dataset, images, add_index, label, width, height):
    """Add/replace images to a image dataset at a specified index"""
    if isinstance(images, list):
        idx = add_index
        for image in images:
            image_dataset[idx].height = height
            image_dataset[idx].width = width
            image_dataset[idx].image_data = image
            image_dataset[idx].label = label
            idx += 1

    else:
        image_dataset[add_index].height = height
        image_dataset[add_index].width = width
        image_dataset[add_index].image_data = images
        image_dataset[add_index].label = label

def save_to_lmdb(image_dataset, output_lmdb, is_float_data):
    """Save contents of image dataset to an lmdb
    image_dataset: images in a numpy record array
    output_lmdb: path to output lmdb

    returns caffe_lmdb instance
    """
    # shuffle the images before storing in the lmdb
    # np.random.shuffle(image_dataset) # does not work
    lmdb_size = 5L * image_dataset.height[0] * image_dataset.width[0] * image_dataset.size

    if is_float_data:
        lmdb_size = lmdb_size * 4L;

    shuffled_indices = range(image_dataset.size)
    np.random.shuffle(shuffled_indices)

    image_database = caffe_lmdb.CaffeLmdb(output_lmdb, lmdb_size)
    image_database.start_write_transaction()
    count = 0
    key = 0

    for i in shuffled_indices:
        count += 1
        key += 1
        image = image_dataset[i]
        datum = datum_pb2.Datum()
        datum.channels = 1 #always one for neuromorphic images
        datum.height = image['height'].item(0)
        datum.width = image['width'].item(0)

        if is_float_data:
            float_img = image['image_data'].flatten().tolist()
            datum.float_data.extend(float_img)
        else:
            datum.data = image['image_data'].tobytes()  # or .tostring() if numpy < 1.9

        datum.label = image['label'].item(0)
        str_id = '{:08}'.format(key)

        image_database.write_datum(str_id, datum)

        #Interim commit every 1000 images
        if count % 1000 == 0:
            image_database.commit_write_transaction()
            image_database.start_write_transaction()

    image_database.commit_write_transaction()
    return image_database

def save_to_mat(image_dataset, output_mat):
    """Save contents of image dataset to an matlab format
    image_dataset: images in a numpy record array
    output_mat: path to output mat

    returns void
    """
    # shuffle the images before storing in the dataset
    shuffled_indices = range(image_dataset.size)
    np.random.shuffle(shuffled_indices)
    num_images = image_dataset.size    
    num_features = image_dataset.height[0]*image_dataset.width[0]
    num_labels = 10

    # Numpy Array which will be written into output matrix (Pre-allocate memory)
    # Assumes image_dataset[i]['image_data'] is of the format = [height, width]
    images = np.zeros((num_images,num_features), dtype=np.float) # np.uint8 is sufficient if space is an issue
    labels = np.zeros((num_images,num_labels), dtype=np.float)

    key = 0
    for i in shuffled_indices:
        image = image_dataset[i]
        flat_image = image['image_data'].T
        images[key,:] = flat_image.flatten()
        labels[key,:] = np.zeros((1,num_labels))
        labels[key,image['label'].item(0)] = 1
        key += 1

    savemat(output_mat, {'labels' : labels, 'data' : images}, appendmat=True,do_compression=True)

def generate_nmnist_dataset(initial_size, input_dir, num_spikes, step_factor):
    """Parse the specified directory containing nmnist files to generate an image dataset
    initial_size: initial size of the image dataset.
        Set this to an appropriately high value to avoid expensive reallocation
    input_dir: input directory.
        Should contain folders 0 to 9, each containing a set of bin files (n mnist recordings)
    num_spikes: number of unique spikes per image
    step_factor: proportional amount to shift before generating the next image
        1 would result in no overlapping events between images
        0.6 would result in the next image overlapping with 40% of the previous image
    """
    image_dataset = np.rec.array(None, dtype=[('height', np.uint16), ('width', np.uint16), ('image_data', 'object'), ('label', np.uint32)], shape=(initial_size))
    num_images = 0

    # loop through each folder within the test directories
    for i in range(0, 10):
        current_dir = input_dir + os.path.sep + str(i) + os.path.sep + '*.bin'
        print 'Processing %s...' %current_dir
        for filename in glob.iglob(current_dir):
            images = prepare_n_mnist(filename, True, num_spikes, step_factor)
            if num_images + len(images) >= image_dataset.size:
                image_dataset = np.resize(image_dataset, (num_images + len(images)) * 2)
            add_images_to_dataset(image_dataset, images, num_images, i, 28, 28)
            num_images += len(images)

    return image_dataset[0:num_images]

def generate_nmnist_continuous_dataset(initial_size, input_dir):
    """Parse the specified directory containing nmnist files to generate an image dataset meant for training
    initial_size: initial size of the image dataset.
        Set this to an appropriately high value to avoid expensive reallocation
    input_dir: input directory.
        Should contain folders 0 to 9, each containing a set of bin files (n mnist recordings)
    """
    image_dataset = np.rec.array(None, dtype=[('height', np.uint16), ('width', np.uint16), ('image_data', 'object'), ('label', np.uint32)], shape=(initial_size))
    num_images = 0

    # loop through each folder within the test directories
    for i in range(0, 10):
        current_dir = input_dir + os.path.sep + str(i) + os.path.sep + '*.bin'
        print 'Processing %s...' %current_dir
        for filename in glob.iglob(current_dir):
            image = prepare_n_mnist_continuous(filename, False, False)
            if num_images + 1 >= image_dataset.size:
                image_dataset = np.resize(image_dataset, (num_images * 2))
            add_images_to_dataset(image_dataset, image, num_images, i, 28, 28)
            num_images += 1

    return image_dataset[0:num_images]

def show_lmdb_datum(key, datum):
    flat_image = np.fromstring(datum.data, dtype=np.uint8)
    if (datum.channels == 1):
        image = flat_image.reshape(datum.height, datum.width)
    else:
        image = flat_image.reshape(datum.channels, datum.height, datum.width)
    label = datum.label
    print label
    cv2.imshow('img', image)
    cv2.waitKey(1)

def main():
    #"""Prepare neuromorphic MNIST image datasets for use in caffe
    #Each dataset will be generated with different number of unique spikes
    #"""
    #initial_size = 1e6 #best to make this big enough avoid expensive re-allocation
    #test_dir = os.path.abspath('testFull')
    #train_dir = os.path.abspath('trainFull')

    #for num_spikes in range(150, 260, 10):
    #    #test directory
    #    image_dataset = generate_nmnist_dataset(initial_size, test_dir, num_spikes, 0.75)
    #    output_lmdb = 'testlmdb' + str(num_spikes)
    #    database = save_to_lmdb(image_dataset, output_lmdb)
    #    #database.process_all_data(show_lmdb_datum)

    #    #train directory
    #    image_dataset = generate_nmnist_dataset(initial_size, train_dir, num_spikes, 0.75)
    #    output_lmdb = 'trainlmdb' + str(num_spikes)
    #    save_to_lmdb(image_dataset, output_lmdb)

    ##TD = ev.read_dataset(os.path.abspath('trainReduced/0/00002.bin'))

    """Prepare neuromorphic MNIST image datasets for use in caffe
    Datasets generated are for continuous spike processing by TrueNorth layers
    """
    initial_size = 6e5 #best to make this big enough avoid expensive re-allocation
    test_dir = os.path.abspath('testFull')
    train_dir = os.path.abspath('trainFull')

    #test directory
    image_dataset = generate_nmnist_continuous_dataset(initial_size, test_dir)
    database = save_to_lmdb(image_dataset, 'testlmdb_continuous', True)
    save_to_mat(image_dataset, 'MNIST_continuous_test.mat');
    #database.process_all_data(show_lmdb_datum)

    #train directory
    image_dataset = generate_nmnist_continuous_dataset(initial_size, train_dir)
    save_to_lmdb(image_dataset, 'trainlmdb_continuous', True)
    save_to_mat(image_dataset, 'MNIST_continuous_train.mat');

    #TD = ev.read_dataset(os.path.abspath('trainReduced/0/00002.bin'))
if __name__ == "__main__":
    main()
