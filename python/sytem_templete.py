import numpy as np
from scipy.signal import StateSpace, lsim


class LTVSystem:
    """
    線形時不変システムクラス
    """
    def __init__(
            self,
            A: np.ndarray,
            B: np.ndarray,
            C: np.ndarray,
            D: np.ndarray,
    ) -> None:
        self.sys = StateSpace(A,B,C,D)

    def solve_next_state(
            self,
            dt: float,
            x: float | list | np.ndarray,
            u: float | list | np.ndarray = None,
    ) -> tuple[np.ndarray,np.ndarray]:
        """
        指定されたdt分だけ微分方程式を計算し、新たな状態変数xとyを求める関数
        """
        t = np.array([0,dt])
        U = np.vstack([u,u])
        tout, y_next, x_next = lsim(self.sys, U=U, T=t, X0=x, interp=True)
        return y_next, x_next
    
    def __call__(
            self,
            dt: float,
            x: float | list | np.ndarray,
            u: float | list | np.ndarray = None,
    ) -> np.ndarray:
        y_next, _ = self.solve_next_state(dt,x,u)
        return y_next
    


class springMassDamper1d(LTVSystem):
    """
    ばねますダンパ系システム（マスが1個のタイプ）
    """
    def __init__(
            self,
            m: float,
            c: float,
            k: float,
    ) -> None:
        A = np.array([[0.0, 1.0],
                      [-k/m, -c/m]])
        B = np.array([[0.0],
                      [1.0/m]])
        C = np.array([1, 0])
        D = np.array([0, 0])

        super().__init__(A,B,C,D)



class springMassDamper2d(LTVSystem):
    """
    ばねますダンパ系システム（マスが2個のタイプ）
    """
    def __init__(
            self,
            m1: float,
            m2: float,
            c1: float,
            c2: float,
            k1: float,
            k2: float,
    ) -> None:
        A = np.array([[0.0, 1.0, 0.0, 0.0],
                      [-(k1 + k2)/m1, -(c1 + c2)/m1, k2/m1, c2/m1],
                      [0.0, 0.0, 0.0, 1.0],
                      [k2/m2, c2/m2, -k2/m2, -c2/m2]])
        B = np.array([[0.0, 0.0],
                      [1.0/m1, 0.0],
                      [0.0, 0.0],
                      [0.0,  1.0/m2]])
        C = np.array([[1.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 1.0, 0.0]])
        D = np.zeros((2, 2))
        super().__init__(A,B,C,D)