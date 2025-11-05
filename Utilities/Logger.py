from datetime import datetime
import os
from tensorboardX import SummaryWriter
import pandas as pd
from Models.RunningAverage import RunningAverage
from Models.TestingStrategy import TestingStrategy
from Models.Timer import Timer
from Models.BenchmarkType import BenchmarkType

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
    def Log(*args, sep=' ', end='\n', filename='log.txt'):
        # Print to console
        print(*args, sep=sep, end=end)
        if Logger.GetLogPath() is not None:
            log_file = os.path.join(Logger.GetLogPath(), filename)
            with open(log_file, 'a', encoding='utf-8') as f:
                print(*args, sep=sep, end=end, file=f)

    @staticmethod
    def LogSummaryWriter(
        writer: SummaryWriter, 
        tag_prefix: str, 
        metrics: dict, 
        epoch: int, 
        step: int, 
        step_per_epoch: int
    ):
        global_step = (epoch - 1) * step_per_epoch + step
        for name, value in metrics.items():
            writer.add_scalars(name, {tag_prefix: value}, global_step)

    @staticmethod
    def LogTestResultsToCSV(
        results: dict[str, RunningAverage], 
        dataset: BenchmarkType, 
        scale: int, 
        out_path: str, 
        timer: Timer, 
        model_path: str, 
        test_type: TestingStrategy = TestingStrategy.Simple
    ):
        for k, v in results.items():
            results[k] = v.GetItem()

        results['Elapsed'] = Timer.ConvertTimeToText(timer.Elapsed())
        results['Dataset'] = dataset
        results['Scale'] = scale
        results['ModelPath'] = model_path
        results['TestType'] = test_type
        results['CreatedDate'] = datetime.now()

        row_df = pd.DataFrame([results])

        # If you already have a DataFrame 'df', append
        # Otherwise, create a new one
        os.makedirs(out_path, exist_ok=True)
        OUT_PATH = os.path.join(out_path, test_type.value + '_' + "metrics.csv")
        try:
            df = pd.read_csv(OUT_PATH)   # load existing file
        except:
            df = pd.DataFrame()               # empty DataFrame if no file yet

        # Append the new row
        df = pd.concat([df, row_df], ignore_index=True)

        # Save back to CSV
        df.to_csv(OUT_PATH, index=False)

        return OUT_PATH