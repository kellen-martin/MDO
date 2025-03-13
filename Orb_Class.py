import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import openmdao.api as om


class Orbit(om.ExplicitComponent):

    def setup(self):
        self.add_input('delta_v_theta', val=0.0)
        self.add_input('delta_v_r', val=0.0)

        self.add_output('delta_v', val= 0.0)
        self.add_output('r_final', val = 0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def dynEqn(self, t, State):
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
    
    def compute(self, inputs, outputs):

        delta_v_theta = inputs['delta_v_theta']
        delta_v_r = inputs['delta_v_r']
        
        ## Initial Conditions
        theta0 = 0.0
        r0 = 1.0
        v_theta0 = 1.0 + float(delta_v_theta)
        v_r0 = 0.0 + float(delta_v_r)
        State0 = np.array([theta0, r0, v_theta0, v_r0])
        tspan = [0, 5]

        ## Solve IVP
        sol = solve_ivp(self.dynEqn, tspan, State0, max_step = 0.1)
        thetas = sol.y[0]
        rs = sol.y[1]
        rf = rs[-1]

        ## Outputs
        outputs['delta_v'] = np.abs(delta_v_theta) + np.abs(delta_v_r)
        outputs['r_final'] = rf - 2

if __name__ == "__main__":

    model = om.Group()
    model.add_subsystem('orb_comp', Orbit())

    prob = om.Problem(model)
    prob.setup()

    prob.set_val('orb_comp.delta_v_theta', 0.5)
    prob.set_val('orb_comp.delta_v_r', 0.0)

    prob.run_model()
    print('Initial Guess : ')
    print('Delta V Theta : ', prob.get_val('orb_comp.delta_v_theta'), ' [DU/TU]')
    print('Delta V R     : ', prob.get_val('orb_comp.delta_v_r'), ' [DU/TU]')
    print('Delta V Total : ', prob['orb_comp.delta_v'], '[DU/TU]')
    print('Final Radius  :', prob['orb_comp.r_final'] + [2], '[DU]')
    print("")

