"""This module contains classes, functions and an example (main) for preparing image data for
caffe"""
import numpy as np
import lmdb
import datum_pb2


class CaffeLmdb(object):
    """Encapsulation of an lmdb meant to store images for Caffe.
    Note that LMDB allows only one concurrent write transaction and multiple concurrent readers"""

    def __init__(self, lmdb_path, map_size=10737418240):
        """
        lmdb_path: str
        map_size: int, defaults to 10 Gigabytes. There is little drawback to
            setting this too big but it will fail catastrophically if the size is
            too small to hold all the data.
        """
        self.lmdb = lmdb.open(lmdb_path, map_size)
        self._write_txn = None


    def __del__(self):
        #if hasattr(self, '_write_txn'):
        if self._write_txn is not None:
            self._write_txn.abort()

        del self._write_txn
        self.lmdb.close()

    def start_write_transaction(self):
        """Start a write transaction so that data can be written to this lmdb.
       Refer to WriteDatum"""
        #if not hasattr(self, '_write_txn'):
        if self._write_txn is None:
            self._write_txn = self.lmdb.begin(write=True)
        else:
            print "Transaction is already open"

    def commit_write_transaction(self):
        """Commit the transaction so that data is written to the lmdb"""
        #if hasattr(self, '_write_txn'):
        if self._write_txn is not None:
            self._write_txn.commit()
            self._write_txn = None
        else:
            print "No transaction present"

    def close_write_transaction(self):
        """Close the transaction. Any data that has not been committed yet will be discarded"""
        #if hasattr(self, '_write_txn'):
        if self._write_txn is not None:
            self._write_txn.abort()
            self._write_txn = None
        else:
            print "No transaction present"

    def write_datum(self, key, datum):
        """
        Insert/Update a single datum to the lmdb.
            Transaction must be started before this, and closed after all data has been written
        key: str
        datum: datum_pb2.Datum()
        """
        #if not hasattr(self, '_write_txn'):
        if self._write_txn is None:
            print "Transaction is not started yet"
        elif key is None:
            print "Key cannot be None"
        elif not isinstance(datum, datum_pb2.Datum):
            print "datum must be of type datum_pb2.Datum"
        else:
            self._write_txn.put(key.encode('ascii'), datum.SerializeToString(), overwrite=True)

    def read_data(self, *keys):
        """Retrieve the data corresponding to the specified keys.
        data will be returned as a dictionary of key-Datum pairs
        Concurrent read transactions are allowed."""
        data = {}
        with self.lmdb.begin() as txn:
            for key in range(keys):
                datum = datum_pb2.Datum()
                datum.ParseFromString(txn.get(key))
                data[key] = datum
        return data

    def process_all_data(self, process_function):
        """Iterate over all the data in the lmdb and apply the
        specified function on each datum.
        The lmdb contents will not be modified"""
        with self.lmdb.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                datum = datum_pb2.Datum()
                datum.ParseFromString(value)
                process_function(key, datum)

    def read_all_data(self):
        """Retrieve all data from the lmdb as a dictionary of key-datum pairs"""
        data = {}
        with self.lmdb.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                datum = datum_pb2.Datum()
                datum.ParseFromString(value)
                data[key] = datum
        return data

def main():
    """Example of saving and reading grayscale image data (datum) to a caffe lmdb"""
    num_images = 1000

    # Let's pretend these are grayscale MNist images
    images = np.zeros((num_images, 1, 28, 28), dtype=np.uint8) # 1 channel, 8 bits
    labels = np.empty(num_images, dtype=np.uint8) # random labels
    labels = labels.astype(int) % 10 # labels will be between 0-9


    # We need to prepare the database for the size.
    # If you still run into problem after raising
    # this, you might want to try saving fewer entries
    # in a single transaction.
    map_size = images.nbytes * 10

    image_database = CaffeLmdb('mylmdb', map_size)
    image_database.start_write_transaction()
    for i in range(num_images):
        datum = datum_pb2.Datum()
        datum.channels = images.shape[1]
        datum.height = images.shape[2]
        datum.width = images.shape[3]
        datum.data = images[i].tobytes()  # or .tostring() if numpy < 1.9
        datum.label = labels[i]
        str_id = '{:08}'.format(i)

        image_database.write_datum(str_id, datum)

    image_database.commit_write_transaction()
    image_database.close_write_transaction()

    def print_function(key, datum):
        """Print the key and label of the datum"""
        print 'key: {0}\n\tvalue: {1}'.format(key, datum.label)

    image_database.process_all_data(print_function)

if __name__ == "__main__":
    main()
