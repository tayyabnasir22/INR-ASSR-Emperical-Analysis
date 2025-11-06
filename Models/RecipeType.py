from enum import Enum

class RecipeType(Enum):
    Simple = "Simple"
    GradLoss = 'GradLoss'
    GramL1Loss = 'GramL1Loss'
    SGDR = 'SGDR'
