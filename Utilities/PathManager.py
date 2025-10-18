from Utilities.Logger import Logger
import os
from tensorboardX import SummaryWriter
import shutil

class PathManager:
    @staticmethod
    def CheckPathExists(path, remove=True):
        basename = os.path.basename(path.rstrip('/'))
        if os.path.exists(path):
            if remove and (basename.startswith('_')
                    or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
                shutil.rmtree(path)
                os.makedirs(path)
        else:
            os.makedirs(path)

    @staticmethod
    def SetModelSavePath(save_path, remove=True):
        PathManager.CheckPathExists(save_path, remove=remove)
        Logger.set_log_path(save_path)
        writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
        return Logger.Log, writer