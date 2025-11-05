from dataclasses import dataclass
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations

@dataclass
class ValidationConfigurations:
    model_path: str
    logger_path: str
    data_configurations: ValidationDataConfigurations