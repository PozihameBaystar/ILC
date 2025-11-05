import numpy as np
from scipy.integrate import solve_ivp


class system:

    """
    制御対象のひな型となるクラスを作る
    """

    def __init__(self):
        pass

    def state_function(self,t,y):
        """
        子クラスではここを上書きする
        ここに状態方程式を書く
        末尾に入力を持ってくるものとする
        """
        pass

    def solve_next_state(
            self,
            dt: float,
            x: float | list | np.ndarray,
            u: float | list | np.ndarray,
    ):
        """
        指定されたdt分だけ微分方程式を計算し、当たらな状態変数xを求める関数
        """
        u_len = len(u)
        y = np.hstack((x,u))
        sol = solve_ivp(self.state_function,(0,dt),y)
        y_next = sol.y.T[-1]
        x_next = y_next[:-u_len]
        return x_next