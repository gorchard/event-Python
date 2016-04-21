"""This module contains classes, functions and a program (main) for reading neuromorphic datasets, processing the data and then saving into a caffe lmdb"""
import glob
import math
import os
import cv2
import numpy as np
from scipy import ndimage
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

def make_td_images(td, num_spikes, step_factor=1):
    """Generate set of images from the Temporal Difference (td) events
    td is read from a binary file (refer to eventvision.Readxxx functions)
    td: eventvision.Events
    num_spikes: number of unique spikes to accumulate before generating an image
    step_factor: proportional amount to shift before generating the next image.
        1 would result in no overlapping events between images
        0.6 would result in the next image overlapping with 40% of the previous image
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

def add_images_to_dataset(image_dataset, images, add_index, label, width, height):
    """Add/replace images to a image dataset at a specified index"""
    idx = add_index
    for image in images:
        image_dataset[idx].height = height
        image_dataset[idx].width = width
        image_dataset[idx].image_data = image
        image_dataset[idx].label = label

        idx += 1

def save_to_lmdb(image_dataset, output_lmdb):
    """Save contents of image dataset to an lmdb
    image_dataset: images in a numpy record array
    output_lmdb: path to output lmdb

    returns caffe_lmdb instance
    """
    # shuffle the images before storing in the lmdb
    # np.random.shuffle(image_dataset) # does not work
    lmdb_size = 5L * image_dataset.height[0] * image_dataset.width[0] * image_dataset.size
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
    """Prepare lots of neuromorphic MNIST datasets for use in caffe
    Each dataset will be generated with different number of unique spikes
    """
    initial_size = 1e6 #best to make this big enough avoid expensive re-allocation
    test_dir = os.path.abspath('testFull')
    train_dir = os.path.abspath('trainFull')

    for num_spikes in range(150, 260, 10):
        #test directory
        image_dataset = generate_nmnist_dataset(initial_size, test_dir, num_spikes, 0.75)
        output_lmdb = 'testlmdb' + str(num_spikes)
        database = save_to_lmdb(image_dataset, output_lmdb)
        #database.process_all_data(show_lmdb_datum)

        #train directory
        image_dataset = generate_nmnist_dataset(initial_size, train_dir, num_spikes, 0.75)
        output_lmdb = 'trainlmdb' + str(num_spikes)
        save_to_lmdb(image_dataset, output_lmdb)

    #TD = ev.read_dataset(os.path.abspath('trainReduced/0/00002.bin'))
if __name__ == "__main__":
    main()
