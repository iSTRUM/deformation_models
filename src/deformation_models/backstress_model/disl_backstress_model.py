from __future__ import annotations

import numpy as np
from unyt import unyt_array, unyt_quantity

gas_constant = unyt_quantity(8.314472, "m**2 * kg / (s**2) / K / mol")


class ModelContainer:
    # all the constants
    beta = unyt_quantity(2, "")
    burgers_vec = unyt_quantity(5, "nm")
    peierls_barrier_hgt = unyt_quantity(3.1, "GPa")
    activation_E_glide = unyt_quantity(450.0, "kJ / mol")  # dF
    prefactor_gb_diff = unyt_quantity(10**3.53, "1 / s / (Pa**4)")
    prefactor_pipe_diff = unyt_quantity(10**-0.95, "1 / s / (Pa**5)")
    prefactor_plasticity = unyt_quantity(10**6.94, "m**2 / s")
    sig_p_max = unyt_quantity(1.8, "GPa")
    shear_mod = unyt_quantity(65.0, "GPa")  # should be T dependent
    hardening_mod = unyt_quantity(135.0, "GPa") # any T dependence?

    def __init__(
        self,
        T: unyt_array,
        grain_size: unyt_array,
        stress_omega: unyt_quantity,
        stress_amp: unyt_quantity,
        stress_bias: unyt_quantity,
    ):
        self.T = T.to("Kelvin")
        self.grain_size = grain_size
        self.A_gb = self.arrhenius(
            self.T, self.prefactor_gb_diff, self.activation_E_glide
        )
        self.A_pipe = self.arrhenius(
            self.T, self.prefactor_pipe_diff, self.activation_E_glide
        )
        self.A_plastic = self.arrhenius(
            self.T, self.prefactor_plasticity, self.activation_E_glide
        )
        self.sig_ref = self.sig_ref_func()
        self.sig_d = self.sig_d_func()
        self.A_plastic_prime = self.A_plastic_prime_func()
        self.stress_omega = stress_omega
        self.stress_amp = stress_amp
        self.stress_bias = stress_bias

    @staticmethod
    def arrhenius(
        T: unyt_array, pre_factor: unyt_quantity, activation_energy: unyt_quantity
    ):
        return pre_factor * np.exp(-activation_energy / (gas_constant * T))

    def sig_ref_func(self) -> unyt_array:
        return (
            self.peierls_barrier_hgt * gas_constant * self.T / self.activation_E_glide
        )

    def sig_d_func(self) -> unyt_array:
        return self.beta * self.shear_mod * self.burgers_vec / self.grain_size

    def A_plastic_prime_func(self) -> unyt_array:
        return self.A_plastic  # add the other coeffs

    def update_T(self, T: unyt_array):
        self.T = T.to("Kelvin")
        self.A_gb = self.arrhenius(
            self.T, self.prefactor_gb_diff, self.activation_E_glide
        )
        self.A_pipe = self.arrhenius(
            self.T, self.prefactor_pipe_diff, self.activation_E_glide
        )
        self.A_plastic = self.arrhenius(
            self.T, self.prefactor_plasticity, self.activation_E_glide
        )
        self.sig_ref = self.sig_ref_func()
        self.sig_d = self.sig_d_func()

    def stress_t(self, t):
        tvar = np.sin(t * self.stress_omega / (2 * np.pi))
        return tvar * self.stress_amp + self.stress_bias


def epsdot_p(t, sig_p, mc: ModelContainer):
    sigma = mc.stress_t(t)
    sinh_arg = (sigma - sig_p - mc.sig_d) / mc.sig_ref
    return mc.A_plastic_prime * sig_p**2 * np.sinh(sinh_arg)
