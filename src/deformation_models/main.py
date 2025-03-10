from __future__ import annotations

import abc

import numpy as np
from scipy.special import gamma


class MaterialModel(abc.ABC):
    def __init__(self, Ju, tau_m):
        self.Ju = Ju
        self.tau_m = tau_m

    @abc.abstractmethod
    def J_t(self, t):
        pass

    @abc.abstractmethod
    def J1_w(self, w):
        pass

    @abc.abstractmethod
    def J2_w(self, w):
        pass

    def M_t(self, t):
        return 1 / self.J_t(t)

    def J_w(self, w):
        """The full complex compliance"""
        return self.J1_w(w) - 1 * self.J2_w(w) * 1j

    def M_w(self, w):
        """The full complex modulus"""
        return 1 / self.J_w(w)

    def M1_w(self, w):
        """Real part of the complex modulus"""
        return np.real(self.M_w(w))

    def M2_w(self, w):
        """Imaginary part of the complex modulus"""
        return np.imag(self.M_w(w))

    def Q_w_approx(self, w):
        """Q with the small Q approximation"""
        return self.J2_w(w) / self.J1_w(w)

    def Q_w(self, w):
        J1 = self.J1_w(w)
        J2 = self.J2_w(w)
        Qfac = (1.0 + np.sqrt(1.0 + (J2 / J1) ** 2)) / 2.0
        return self.Q_w_approx(w) * Qfac


class MaxwellModel(MaterialModel):
    def J_t(self, t):
        return self.Ju * (1 + t / self.tau_m)

    def J1_w(self, w=None):
        _ = w  # frequency independent
        return self.Ju

    def J2_w(self, w):
        return self.Ju / (w * self.tau_m)


class AndradeModel(MaterialModel):
    def __init__(self, Ju, tau_m, beta=1e-5, alpha=1.0 / 3):
        super().__init__(Ju, tau_m)
        self.beta = beta
        self.alpha = alpha

    def J_t(self, t):
        return self.Ju + self.beta * t**self.alpha + self.Ju * t / self.tau_m

    def J1_w(self, w):
        alf = self.alpha
        J_fac = 1 + self.beta * gamma(1 + alf) * np.cos(alf * np.pi / 2) / (w**alf)
        return self.Ju * J_fac

    def J2_w(self, w):
        alf = self.alpha
        J_fac = 1.0 / (w * self.tau_m) + self.beta * gamma(1 + alf) * np.sin(
            alf * np.pi / 2
        ) / (w**alf)
        return self.Ju * J_fac


class SLS(MaterialModel):
    # zener model
    def __init__(self, Ju_1, tau_m, Ju_2):
        super().__init__(Ju_1, tau_m)
        self.Ju_2 = Ju_2

    def J_t(self, t):
        return self.Ju + self.Ju_2 * (1 - np.exp(-t / self.tau_m))

    def _w_tau_fac(self, w):
        return 1 + (w * self.tau_m) ** 2

    def J1_w(self, w):
        return self.Ju + self.Ju_2 / self._w_tau_fac(w)

    def J2_w(self, w):
        return self.Ju_2 * w * self.tau_m / self._w_tau_fac(w)


class Burgers(MaterialModel):
    def __init__(self, Ju1, tau_m1, Ju2, tau_m2):
        super().__init__(Ju1, tau_m1)
        self.Ju_2 = Ju2
        self.tau_m_2 = tau_m2

    def J_t(self, t):
        return self.Ju * (1 + t / self.tau_m) + self.Ju_2 * (
            1 - np.exp(t / self.tau_m_2)
        )

    def J1_w(self, w):
        return self.Ju + self.Ju_2 / (1 + (self.tau_m_2 * w) ** 2)

    def J2_w(self, w):
        return self.Ju * 1 / (self.tau_m * w) + self.tau_m_2 * w / (
            1 + (self.tau_m_2 * w) ** 2
        )
