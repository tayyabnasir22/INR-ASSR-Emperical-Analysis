from dataclasses import dataclass
from Configurations.ValidationConfigurations import ValidationConfigurations

@dataclass
class PatchedValidationConfigurations(ValidationConfigurations):
    breakdown_patch_size: int