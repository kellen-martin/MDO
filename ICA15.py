import numpy as np
import openmdao.api as om


class Aero(om.ExplicitComponent):
    """
    Component containing Discipline 1 -- no derivatives version.
    """

    def setup(self):

        # Global Design Variable
        self.add_input('theta', val = np.zeros(2))

        # Local Design Variable
        self.add_input('d', val = np.zeros(2))

        # Coupling output
        self.add_output('gamma', val = np.zeros(2))

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method ='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        """
        theta = inputs['theta']
        d = inputs['d']

        A = np.zeros((2,2))
        b= np.zeros((2,1))

        A[0, 0] = (theta[0] + d[0])**2 + 3
        A[0, 1] = 1.0
        A[1,0] = 1.0
        A[1,1] = (theta[1] + d[1])**2 + 5

        b[0, 0] = theta[0] + d[0]
        b[1, 0] = theta[1] + d[1]
        outputs['gamma'] = np.linalg.solve(A,b)

class Struct(om.ExplicitComponent):
    """
    Component containing Discipline 2 -- no derivatives version.
    """

    def setup(self):
        # Global Design Variable
        self.add_input('t', val=np.zeros(2))

        # Coupling parameter
        self.add_input('theta', val=np.zeros(2))

        self.add_input('gamma', val = np.zeros(2))

        # Coupling output
        self.add_output('d', val=np.zeros(2))

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """
        t = inputs['t']
        theta = inputs['theta']
        gamma = inputs['gamma']

        A = np.zeros((2,2))
        b= np.zeros((2,1))

        A[0, 0] = 10*t[0] - theta[0]
        A[0, 1] = 1.0
        A[1,0] = 1.0
        A[1,1] = 10*t[1] - theta[1]

        b[0, 0] = gamma[0]**2
        b[1, 0] = gamma[1]**2
        outputs['d'] = np.linalg.solve(A,b)

class Force(om.ExplicitComponent):
    """
    Component containing Discipline 2 -- no derivatives version.
    """

    def setup(self):
        # Global Design Variable
        self.add_input('gamma', val=np.zeros(2))

        # Coupling parameter
        self.add_input('d', val=np.zeros(2))

        self.add_input('theta', val = np.zeros(2))

        # Coupling output
        self.add_output('D', 0.0)

        self.add_output('L', 0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials('*', '*', method='fd')

    def compute(self, inputs, outputs):
        """
        Evaluates the equation
        y2 = y1**(.5) + z1 + z2
        """
        gamma = inputs['gamma']
        theta = inputs['theta']

        outputs['D'] = gamma[0]*np.sin(theta[0]) + gamma[1]*np.sin(theta[1])
        outputs['L'] = 10*(gamma[0] + gamma[1])







class SellarMDA(om.Group):
    """
    Group containing the Sellar MDA.
    """

    def setup(self):
        cycle = self.add_subsystem('cycle', om.Group(), promotes=['*'])
        cycle.add_subsystem('Aero', Aero(), promotes=['*'])
        cycle.add_subsystem('Struct', Struct(), promotes=['*'])

        #cycle.set_input_defaults('x', 1.0)
        #cycle.set_input_defaults('z', np.array([5.0, 2.0]))

        # Nonlinear Block Gauss Seidel is a gradient free solver
        cycle.nonlinear_solver = om.NonlinearBlockGS()

        self.add_subsystem('force_comp', Force(), promotes=['*'])
        self.add_subsystem('stress_comp', om.ExecComp('sigma = 1000*(d[0] + d[1])'))


prob = om.Problem()
prob.model = SellarMDA()

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'
# prob.driver.options['maxiter'] = 100
prob.driver.options['tol'] = 1e-8

prob.model.set_input_defaults("theta", np.ones(2)*.1)
prob.model.set_input_defaults("t", np.ones(2))
prob.model.add_design_var('theta', lower=0, upper=10)
prob.model.add_design_var('t', lower=0, upper=10)
prob.model.add_objective('D')
prob.model.add_constraint('L', equals=0)
prob.model.add_constraint('sigma', upper=1)

# Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer
prob.model.approx_totals()

prob.setup()
prob.set_solver_print(level=0)

prob.run_driver()

print('minimum found at')
print(prob.get_val('theta'))
print(prob.get_val('t'))
