from tap import Tap


class TrainArgumentParser(Tap):
    def __init__(self):
        super().__init__(underscores_to_dashes=True)
    lr: float = 1e-3
