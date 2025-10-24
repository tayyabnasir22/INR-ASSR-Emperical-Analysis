from time import time

class Timer():
    def __init__(self):
        self.start = time.time()

    def Reset(self):
        self.start = time.time()

    def Elapsed(self):
        return time.time() - self.start