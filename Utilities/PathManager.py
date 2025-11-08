from Models.DecoderType import DecoderType
from Models.EncoderType import EncoderType
from Models.RecipeType import RecipeType
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
        Logger.SetLogPath(save_path)
        writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
        return Logger.Log, writer
    
    @staticmethod
    def GetModelSavePath(
        encoder: EncoderType, 
        decoder: DecoderType, 
        recipe: RecipeType, 
        input_patch: int = 48, 
        scale_range: list[int, int] = [1, 4]
    ):
        return '_'.join(
            [
                './model_states',
                encoder.name,
                decoder.name,
                recipe.name,
                'Patch',
                str(input_patch),
                'Scale',
                str(scale_range[0]),
                str(scale_range[1]),

            ]
        )