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
        tout, y_next, x_next = lsim(self.sys, U=U, T=t, X0=x, interp=False)
        return y_next[-1], x_next[-1]
    
    def solve_sequence(
            self,
            t_sequence: np.ndarray,
            x0: np.ndarray,
            us: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        連続した時刻分計算する
        """
        N_seq = len(t_sequence)
        if N_seq-1 == len(us):
            us = np.vstack([us,us[-1]])
        tout, ys, xs = lsim(self.sys, U=us, T=t_sequence, X0=x0, interp=False)
        return tout, ys, xs



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
        C = np.array([1.0, 0.0])
        D = np.array([0.0, 0.0])

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