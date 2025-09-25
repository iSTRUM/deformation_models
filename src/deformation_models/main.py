from __future__ import annotations

import abc
from typing import Callable

import numpy as np
import numpy.typing as npt


class MaterialModel(abc.ABC):
    def __init__(self, Ju: float, tau_m: float):
        self.Ju = Ju
        self.tau_m = tau_m

    @abc.abstractmethod
    def J_t(self, t: float | npt.NDArray) -> float | npt.NDArray:
        pass

    def M_t(self, t: float | npt.NDArray) -> float | npt.NDArray:
        return 1 / self.J_t(t)


class MaxwellModel(MaterialModel):
    def __init__(self, Ju: float, tau_m: float):
        """Maxwell model

        Parameters
        ----------
        Ju: unrelaxed compliance
        tau_m: characteristic maxwell time
        """
        super().__init__(Ju, tau_m)

    def J_t(self, t: float | npt.NDArray) -> float | npt.NDArray:
        return self.Ju * (1 + t / self.tau_m)


class AndradeModel(MaterialModel):
    def __init__(
        self, Ju: float, tau_m: float, beta: float = 1e-5, alpha: float = 1.0 / 3
    ):
        """Andrade model

        Parameters
        ----------
        Ju: unrelaxed compliance
        tau_m: characteristic maxwell time
        beta: optional beta-factor of the andrade model, default 1e-5
        alpha: optional alpha-factor of the andrade model, default 1/3

        """
        super().__init__(Ju, tau_m)
        self.beta = beta
        self.alpha = alpha

    def J_t(self, t: float | npt.NDArray) -> float | npt.NDArray:
        return self.Ju + self.beta * t**self.alpha + self.Ju * t / self.tau_m


class SLS(MaterialModel):
    def __init__(self, Ju_1: float, tau_m: float, Ju_2: float):
        """Standard Linear Solid (SLS) or Zener model

        Parameters
        ----------
        Ju_1: unrelaxed compliance
        tau_m: characteristic maxwell time
        Ju_2: unrelaxed compliance of the maxwell-element

        """
        super().__init__(Ju_1, tau_m)
        self.Ju_2 = Ju_2

    def J_t(self, t: float | npt.NDArray) -> float | npt.NDArray:
        return self.Ju + self.Ju_2 * (1 - np.exp(-t / self.tau_m))


class Burgers(MaterialModel):
    def __init__(self, Ju1: float, tau_m1: float, Ju2: float, tau_m2: float):
        """Burgers model

        Parameters
        ----------
        Ju_1: unrelaxed compliance of first element
        tau_m1: characteristic maxwell time of first element
        Ju_2: unrelaxed compliance of second element
        tau_m2: characteristic maxwell time of second element
        """
        super().__init__(Ju1, tau_m1)
        self.Ju_2 = Ju2
        self.tau_m_2 = tau_m2

    def J_t(self, t: float | npt.NDArray) -> float | npt.NDArray:
        return self.Ju * (1 + t / self.tau_m) + self.Ju_2 * (
            1 - np.exp(t / self.tau_m_2)
        )


class DeformationExperiment:
    def __init__(
        self,
        stress_func: Callable,
        solid: MaterialModel,
        t_max: float,
        dt_max: float,
        t_min: float = 0.001,
        n_t: int = 1000,
        delayed_solve: bool = True,
    ):
        """
        A deformation experiment with specified stress history

        Parameters
        ----------
        stress_func:
            A function to use for calculating stress at a given time. Must except an
            array of arbitrary times and return the stresses at those times.
        solid:
            the deformation_model solid instance to calculate strain for.
        t_max:
            the max time to calculate for
        dt_max:
            the max allowed timestep when convolving stress with creep function
        t_min:
            the starting time (default 0.001)
        n_t:
            number of timesteps (default 1000)
        delayed_solve:
            if True (default), you must call .solve() after instantiation to
            calculate strain history. Otherwise, solution is started immediately.

        Notes
        -----
        After calling .solve(), the following attributes will contain the
        results of interest:

        t_actual:
            array with the time steps where strain history was evaluated
        stress_vals:
            array with the stress values at t_actual
        strain_vals:
            array with the strain values at t_actual
        """

        self.stress_func = stress_func
        self.solid = solid

        self.t_max = t_max
        self.dt_max = dt_max
        self.t_min = t_min
        self.n_t = n_t

        self.t_actual = np.linspace(self.t_min, self.t_max, self.n_t)
        self.strain_vals: npt.NDArray
        self.stress_vals: npt.NDArray

        if not delayed_solve:
            self.solve()

    def _stress_steps(self, tstar: npt.NDArray) -> npt.NDArray:
        stress_dot_t = self.stress_func(tstar)
        return stress_dot_t[1:] - stress_dot_t[:-1]

    def _J_signal(self, t: npt.NDArray, tstar: npt.NDArray) -> npt.NDArray:
        t_diff = t - tstar
        return self.solid.J_t(t_diff)[0:-1]

    def solve(self):
        stain_vals = []
        stress_vals = []
        stress_i = 0
        for t in self.t_actual:
            ntstar = int(t / self.dt_max) + 10
            tstar = np.linspace(self.t_min, t, ntstar)

            f1 = self._stress_steps(tstar)
            f2 = self._J_signal(t, tstar)
            stress_i = np.sum(f1)
            strain_i = np.sum(f1 * f2)

            stain_vals.append(strain_i)
            stress_vals.append(stress_i)

        self.strain_vals = np.array(stain_vals)
        self.stress_vals = np.array(stress_vals)
