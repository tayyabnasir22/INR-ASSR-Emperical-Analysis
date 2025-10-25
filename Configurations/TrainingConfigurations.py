from dataclasses import dataclass
from Configurations.TrainingDataConfigurations import TrainingDataConfigurations
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations

@dataclass
class TrainingConfigurations:
    optimizer: dict
    data_configurations: TrainingDataConfigurations
    validation_data_configurations: ValidationDataConfigurations
    lr_scheduler: dict
    epochs: int
    save_path: str
    resume_path: str
    epoch_val: int
    epoch_save: int
    monitor_metric: str