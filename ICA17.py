import numpy as np
import openmdao.api as om

class Sur(om.ExplicitComponent):
    """
    Component containing Discipline 1 -- no derivatives version.
    """

    def setup(self):

        # Global Design Variable
        self.add_input('w', val = np.zeros(3))
        self.add_input('x', val = np.zeros(5))
        self.add_input('y', val = np.zeros(5))

        # Coupling output
        self.add_output('e', val = 0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method ='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        """
        w = inputs['w']
        x = inputs['x']
        y = inputs['y']
        
        e0 = w[0] + w[1]*x[0] + w[2]*x[0]**2 - y[0]
        e1 = w[0] + w[1]*x[1] + w[2]*x[1]**2 - y[1]
        e2 = w[0] + w[1]*x[2] + w[2]*x[2]**2 - y[2]
        e3 = w[0] + w[1]*x[3] + w[2]*x[3]**2 - y[3]
        e4 = w[0] + w[1]*x[4] + w[2]*x[4]**2 - y[4]

        e = np.sqrt(e0**2 + e1**2 + e2**2 + e3**2 + e4**2)

        outputs['e'] = e



prob = om.Problem()
prob.model.add_subsystem('sur', Sur(), promotes=['*'])

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['maxiter'] = 100
prob.driver.options['tol'] = 1e-8

prob.model.set_input_defaults("w", np.ones(3))
prob.model.set_input_defaults("x", np.array([0.0, 0.25, 0.5, 0.75, 1.0]))
prob.model.set_input_defaults("y", np.array([0.0, 1.0, 1.5, 0.9, 1.0]))
prob.model.add_design_var('w', lower=0, upper=10)
prob.model.add_objective('e')

prob.setup()
om.n2
prob.run_driver();

print('Optimal Weights: ', prob.get_val('w'))
