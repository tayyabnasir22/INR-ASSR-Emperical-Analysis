from dataclasses import dataclass
from Configurations.TrainingDataConfigurations import TrainingDataConfigurations
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations
from Decoders.DecoderBase import DecoderBase
from Encoders.EncoderBase import EncoderBase

@dataclass
class TrainingConfigurations:
    optimizer: dict
    data_configurations: TrainingDataConfigurations
    validation_data_configurations: ValidationDataConfigurations
    lr_schedular: dict
    epochs: int
    save_path: str
    resume_path: str
    epoch_val: int
    epoch_save: int