from Orb_Class import Orbit
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import openmdao.api as om

def dynEqn(t, State):
    theta = State[0]
    r = State[1]
    v_theta = State[2]
    v_r = State[3]

    dStatedt = np.zeros(4)
    dStatedt[0] = v_theta/r
    dStatedt[1] = v_r
    dStatedt[2] = -v_r*v_theta/r
    dStatedt[3] = v_theta**2/r - 1.0/(r**2)
    return dStatedt

def prop_orbit(delta_v_theta, delta_v_r):
    ## Initial Conditions
    theta0 = 0.0
    r0 = 1.0
    v_theta0 = 1.0 + delta_v_theta
    v_r0 = 0.0 + delta_v_r
    State0 = [theta0, r0, v_theta0, v_r0]
    tspan = [0, 5]

    ## Solve IVP
    sol = solve_ivp(dynEqn, tspan, State0, max_step = 0.1)
    thetas = sol.y[0]
    rs = sol.y[1]
    print('Final Radius (sim)', rs[-1], ' [DU]')
    print("")
    ## Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    ax.plot(thetas, rs)
    ax.set_rmax(3)
    plt.title('Optimal Trajectory')
    plt.show()
    return

# build the model
prob = om.Problem()
prob.model.add_subsystem('Orbit', Orbit(), promotes = ['*'])


# Inital Guess
prob.model.set_input_defaults('delta_v_theta', [0.5])
prob.model.set_input_defaults('delta_v_r', [0.0])

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('delta_v_theta', lower=[0], upper=[2])
prob.model.add_design_var('delta_v_r', lower=[0], upper=[2])
prob.model.add_objective('delta_v')

# to add the constraint to the model
prob.model.add_constraint('r_final',lower= 0.0)


prob.setup()
om.n2
prob.run_driver();

print('Optimized')
print('Delta V Theta : ', prob.get_val('delta_v_theta'), ' [DU/TU]')
print('Delta V R     : ', prob.get_val('delta_v_r'), ' [DU/TU]')
print('Delta V Total : ', prob.get_val('delta_v'), ' [DU/TU]')
print('Final Radius  :', prob.get_val('r_final') + [2], ' [DU]')

# Re-run and plot IVP using optimized solution
opt_dv_theta = float(prob.get_val('delta_v_theta'))
opt_dv_r = float(prob.get_val('delta_v_r'))
prop_orbit(opt_dv_theta, opt_dv_r)



