class RunningAverage():

    def __init__(self):
        self.samples = 0.0
        self.values = 0.0

    def SetItem(self, v, n=1.0):
        self.values = (self.values * self.samples + v * n) / (self.samples + n)
        self.samples += n

    def GetItem(self):
        return self.values