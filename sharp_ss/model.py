from dataclasses import dataclass, field
import numpy as np

@dataclass
class Bookkeeping:
    totalSteps:     int = int(1e6)
    burnInSteps:    int = None
    nSaveModels:    int = 100
    actionsPerStep: int = 2
    fitNoise:       bool = False

    def __post_init__(self):
        if self.burnInSteps is None:
            self.burnInSteps = int(self.totalSteps // 2)
@dataclass
class Prior:
    stdP: float = 0.1
    maxN: int = 2
    tlen: int = None # G half length in sample points
    dt: float = None
    minSpace: float = 1.0
    ampRange: tuple = (-0.2, 0.2)
    widRange: tuple = None
    rhoRange: tuple = (1.7, 2.0)

    locStd: float = 1.0
    ampStd: float = None
    widStd: float = None
    rhoStd: float = None
    sigStd: float = None
    nc1Std: float = None
    nc2Std: float = None

    negOnly: bool = False
    align: bool = False

    def __post_init__(self):
        if self.ampStd is None:
            self.ampStd = 0.05 * (self.ampRange[1] - self.ampRange[0])
        if self.widStd is None:
            self.widStd = 0.05 * (self.widRange[1] - self.widRange[0])
        if self.rhoStd is None:
            self.rhoStd = 0.05 * (self.rhoRange[1] - self.rhoRange[0])
@dataclass
class Model:
    Nphase: int
    loc: np.ndarray
    amp: np.ndarray
    wid: np.ndarray
    sig: float = 0.1
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

@dataclass
class Model2:
    Nphase: int
    locPP: np.ndarray
    ampPP: np.ndarray
    widPP: np.ndarray
    ampSS: np.ndarray
    widSS: np.ndarray
    rho: np.ndarray

    @classmethod
    def create_empty(cls):
        return cls(
            Nphase=0,
            locPP=np.array([]),
            ampPP=np.array([]),
            widPP=np.array([]),
            ampSS=np.array([]),
            widSS=np.array([]),
            rho=np.array([])
        )
    
    # @classmethod
    # def create_random(cls, prior:Prior):
    #     pass
