import time

class Timer(object):
    """Timer for making ad-hoc timing measurements"""
    def __init__(self, verbose=False):
        self.verbose = verbose

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
        with Timer() as t:
            #Do some time consuming work here
            pass
        print 'elapsed time is %s ms' % t.msecs

    if __name__ == "__main__":
        main()