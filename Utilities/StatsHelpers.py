

class StatsHelpers:
    @staticmethod
    def GetMulitStepMilestones(epochs: int, count_milestones: int = 4) -> list[int]:
        step = epochs / (count_milestones + 1)
        milestones = [int(step * (i + 1)) for i in range(count_milestones)]
        return milestones
