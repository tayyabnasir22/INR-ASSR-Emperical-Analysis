from dataclasses import dataclass

from Configurations.BenchmarkType import BenchmarkType
from Configurations.TrainingDataConfigurations import TrainingDataConfigurations

@dataclass
class ValidationDataConfigurations(TrainingDataConfigurations):
    benchmark_type: BenchmarkType
    eval_batch_size: int
    eval_scale: int