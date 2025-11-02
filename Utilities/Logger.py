import os
from tensorboardX import SummaryWriter

class Logger:
    _log_path = None
    
    @classmethod
    def SetLogPath(cls, path):
        cls._log_path = path

    @classmethod
    def GetLogPath(cls):
        return cls._log_path

    @staticmethod
    def TimeToLogText(t):
        if t >= 3600:
            return '{:.1f}h'.format(t / 3600)
        elif t >= 60:
            return '{:.1f}m'.format(t / 60)
        else:
            return '{:.1f}s'.format(t)
        
    @staticmethod
    def Log(obj, filename='log.txt'):
        print(obj)
        if Logger._log_path is not None:
            with open(os.path.join(Logger._log_path, filename), 'a') as f:
                print(obj, file=f)

    @staticmethod
    def LogSummaryWriter(writer: SummaryWriter, tag_prefix: str, metrics: dict, epoch: int, step: int, step_per_epoch: int):
        global_step = (epoch - 1) * step_per_epoch + step
        for name, value in metrics.items():
            writer.add_scalars(name, {tag_prefix: value}, global_step)