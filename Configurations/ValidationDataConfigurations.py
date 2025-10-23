from dataclasses import dataclass

from Configurations.BenchmarkType import BenchmarkTypes
from Configurations.TrainingDataConfigurations import TrainingDataConfigurations

@dataclass
class ValidationDataConfigurations(TrainingDataConfigurations):
    benchmark_type: BenchmarkTypes
    eval_batch_size: int
    eval_scale: int