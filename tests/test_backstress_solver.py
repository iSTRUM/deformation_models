from deformation_models.backstress_model import ModelContainer, sigmadot_p
from unyt import unyt_quantity
from scipy.integrate import solve_ivp
import numpy as np 

def test_backstress_sigmadot_p():

    mc_T = ModelContainer(
        T=unyt_quantity(1500, 'degC'),
        grain_size=unyt_quantity(1, 'mm'),
        stress_omega=unyt_quantity(60, 'Hz'), 
        stress_amp=unyt_quantity(3, 'MPa'), 
        stress_bias=unyt_quantity(100, 'kPa')
    )

    sigdot = sigmadot_p(0, 0.1, mc_T)
    assert not np.isnan(sigdot)

def test_backstress_solve():
    mc_T = ModelContainer(
        T=unyt_quantity(1150, 'degC'),
        grain_size=unyt_quantity(1, 'mm'),
        stress_omega=unyt_quantity(50, 'Hz'), 
        stress_amp=unyt_quantity(100, 'MPa'), 
        stress_bias=unyt_quantity(30, 'MPa')
    )

    tspan = [0.0, 60 * 60]
    nsteps = 100
    solution = solve_ivp(sigmadot_p, 
                        t_span = tspan, 
                        y0 = (0.0000001,), 
                        t_eval=np.linspace(tspan[0], tspan[1], nsteps),
                        method = 'BDF', 
                        args=(mc_T, ))
    assert np.all(np.isreal(solution.y))

