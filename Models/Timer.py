import time

class Timer():
    def __init__(self):
        self.start = time.time()

    def Reset(self):
        self.start = time.time()

    def Elapsed(self):
        return time.time() - self.start
    
    @staticmethod
    def ConvertTimeToText(t):
        if t >= 3600:
            return '{:.1f}h'.format(t / 3600)
        elif t >= 60:
            return '{:.1f}m'.format(t / 60)
        else:
            return '{:.1f}s'.format(t)