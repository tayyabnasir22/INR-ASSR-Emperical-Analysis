from dataclasses import dataclass
from Configurations.ValidationDataConfigurations import ValidationDataConfigurations

@dataclass
class ValidationConfigurations:
    model_path: str
    save_path: str
    data_configurations: ValidationDataConfigurations