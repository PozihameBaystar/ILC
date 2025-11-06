import numpy as np
from scipy.signal import medfilt

class mf_ILCController:
    def __init__(
            self,
            N: int,
            Kp: float,
            Kd: float,
    ) -> None:
        self.Kp = Kp
        self.Kd = Kd
        self.inputBuffer = np.zeros(N)
        self.errorBuffer = np.zeros(N)
        self.prevErrorBuffer = np.zeros(N)

    def control(
            self,
            ref: np.ndarray,
            y_sec: np.ndarray,
            smoothing: bool = False,
    ) -> np.ndarray:
        # 参照軌道と今回の出力から入力を修正する
        self.prevErrorBuffer = self.errorBuffer
        self.errorBuffer = y_sec - ref

        delta_input = -self.Kp * self.errorBuffer \
            -self.Kd*(self.errorBuffer - self.prevErrorBuffer)
        
        u_ilc = self.inputBuffer + delta_input

        if smoothing:
            u_ilc = medfilt(u_ilc, kernel_size=5)

        self.inputBuffer = u_ilc      
        return u_ilc