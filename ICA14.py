from Classes import Paraboloid

import openmdao.api as om

# build the model
prob = om.Problem()
prob.model.add_subsystem('parab', Paraboloid(), promotes_inputs=['x', 'y'])


# Design variables 'x' and 'y' span components, so we need to provide a common initial
# value for them.
prob.model.set_input_defaults('x', [1.0,])
prob.model.set_input_defaults('y', [1.0,])

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('x', lower=[-50], upper=[50])
prob.model.add_design_var('y', lower=[-50], upper=[50])
prob.model.add_objective('parab.f_xy')

# to add the constraint to the model
prob.model.add_constraint('parab.h', equals = 0.0, linear=True)

prob.setup()
om.n2
prob.run_driver();

print()
