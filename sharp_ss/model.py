from dataclasses import dataclass, field
import numpy as np

@dataclass
class Bookkeeping:
    totalSteps:     int = int(1e6)
    burnInSteps:    int = None
    nSaveModels:    int = 100
    actionsPerStep: int = 2
    fitNoise:       bool = True

    def __post_init__(self):
        if self.burnInSteps is None:
            self.burnInSteps = int(self.totalSteps // 2)

@dataclass
class Prior:
    stdP: float

    maxN: int = 2
    tlen: int = None # G half length in sample points
    dt: float = None
    minSpace: float = 1.0
    ampRange: tuple = (-0.5, 0.5)
    widRange: tuple = None

    locStd: float = 1.0
    ampStd: float = None
    widStd: float = None
    sigStd: float = None
    nc1Std: float = None
    nc2Std: float = None

    negOnly: bool = False
    align: bool = False

    def __post_init__(self):
        if self.ampStd is None:
            self.ampStd = 0.2 * (self.ampRange[1] - self.ampRange[0])
        if self.widStd is None:
            self.widStd = 0.2 * (self.widRange[1] - self.widRange[0])

@dataclass
class Model:
    Nphase: int
    loc: np.ndarray
    amp: np.ndarray
    wid: np.ndarray
    sig: float
    nc1: float = 0.25
    nc2: float = 1.40

    @classmethod
    def create_empty(cls, prior: Prior, nc1=0.25, nc2=1.40):
        return cls(
            Nphase=0,
            loc=np.array([]),
            amp=np.array([]),
            wid=np.array([]),
            sig=prior.stdP,
            nc1=nc1,
            nc2=nc2
        )
    
    # @classmethod
    # def create_random(cls, prior:Prior):
    #     pass