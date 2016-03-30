"""
This timer module provides stop watch functionality for measuring how long loops / methods take to execute.
"""
import time

class Timer(object):
    """Timer for making ad-hoc timing measurements"""
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.secs = 0
        self.msecs = 0
        self.start = 0
        self.end = 0

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs
        if self.verbose:
            print 'elapsed time: %f ms' % self.msecs

def main():
    """Example of how timer is used"""
    with Timer() as my_timer:
        #Do some time consuming work here
        pass
    print 'elapsed time is %s ms' % my_timer.msecs

if __name__ == "__main__":
    main()
