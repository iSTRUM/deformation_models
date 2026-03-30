from __future__ import annotations

import numpy as np
from unyt import unyt_array, unyt_quantity
from scipy.integrate import solve_ivp

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
    taylor_constant = unyt_array(2.46, "") # see Breithaupt appendix

    # assuming here that: hardening modulus = factor * shear modulus 
    # following Breithaupt. Using values from Hein to calculate what
    # that ratio is.
    hardening_shear_mod_ratio_factor = unyt_array(135./65., "") 

    def __init__(
        self,
        T: unyt_quantity,
        grain_size: unyt_quantity,
        stress_omega: unyt_quantity,
        stress_amp: unyt_quantity,
        stress_bias: unyt_quantity,
    ):
        """A container for coefficients and constants for the backstress model

        Paramaters
        ----------
        T (unyt_quantity): 
            temperature
        grain_size (unyt_quantity):
            grain size
        stress_omega (unyt_quantity): 
            applied stress oscillation frequency
        stress_amp (unyt_quantity):
            applied stress amplitude
        stress_bias (unyt_quantity):
            applied stress bias
        """
        self.T = T.to("Kelvin")
        self.grain_size = grain_size

        self.hardening_mod = self.shear_mod * self.hardening_shear_mod_ratio_factor
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
        T: unyt_quantity, pre_factor: unyt_quantity, activation_energy: unyt_quantity
    ) -> unyt_quantity: 
        return pre_factor * np.exp(-activation_energy / (gas_constant * T.to('Kelvin')))

    def sig_ref_func(self) -> unyt_quantity:
        return (
            self.peierls_barrier_hgt * gas_constant * self.T / self.activation_E_glide
        )

    def sig_d_func(self) -> unyt_quantity:
        return self.beta * self.shear_mod * self.burgers_vec / self.grain_size

    def A_plastic_prime_func(self) -> unyt_quantity:
        coeffs = (self.taylor_constant * self.shear_mod * self.burgers_vec) ** 2
        return self.A_plastic / coeffs

    def update_T(self, T: unyt_quantity):
        """ set temperature and update all the temperature-dependent quantities"""
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
        self.A_plastic_prime = self.A_plastic_prime_func()

    def stress_t(self, t: unyt_quantity) -> unyt_quantity:
        """ calculate a stress for a given time """
        tvar = np.sin(t * self.stress_omega)
        return tvar * self.stress_amp + self.stress_bias


def epsdot_p(t: unyt_quantity, sig_p: unyt_quantity, mc: ModelContainer) -> unyt_quantity:
    sigma = mc.stress_t(t)
    sinh_arg = (sigma - sig_p - mc.sig_d) / mc.sig_ref
    return mc.A_plastic_prime * sig_p**2 * np.sinh(sinh_arg)


def sigmadot_p(t: float, sig_p: float, mc: ModelContainer) -> np.ndarray:
    """

    Parameters:
    -----------

    t: float
        time in seconds
    sig_p: float 
        current sigma_p in Mpa 
    mc: ModelContainer
        the model container for the solution

    Returns:
    --------
    np.ndarray
        value of sigmadot_p for the inputs
    """

    t_s = unyt_quantity(t, 's')
    sig_p_MPa = unyt_quantity(sig_p, 'MPa')

    # calculate strain rate for curent time, taylor stress
    epsdot = epsdot_p(t_s, sig_p_MPa, mc) 
    # print(f"epsdot is {epsdot}")

    # calculate taylor stress rate
    fac1 = (mc.sig_d + sig_p_MPa)/sig_p_MPa * epsdot
    fac2 = -1. *  sig_p_MPa / mc.sig_p_max * np.abs(epsdot)
    fac3 = -1. * mc.A_pipe * (sig_p_MPa ** 5)
    fac4 = -1. * mc.A_gb * (sig_p_MPa ** 3) * mc.sig_d
    M = mc.hardening_mod
    # print(f"{fac1=}, {fac2=}, {fac3=}, {fac4=}")

    result = M * (fac1 + fac2 + fac3 + fac4)
    result = np.atleast_1d(result.to('MPa/s').d)
    return result


# https://docs.scipy.org/doc/scipy/reference/integrate.html
# https://docs.scipy.org/doc/scipy/reference/integrate.html#solving-initial-value-problems-for-ode-systems
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.BDF.html#scipy.integrate.BDF


