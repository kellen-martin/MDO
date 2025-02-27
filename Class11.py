import openmdao.api as om

# build the model
prob = om.Problem()

prob.model.add_subsystem('paraboloid', om.ExecComp('f = x0**3 + 2*x1**2 - 4*x0 - 2*x0*x1**2'))

# setup the optimization
prob.driver = om.ScipyOptimizeDriver()
prob.driver.options['optimizer'] = 'SLSQP'

prob.model.add_design_var('paraboloid.x0', lower=-50, upper=50)
prob.model.add_design_var('paraboloid.x1', lower=-50, upper=50)
prob.model.add_objective('paraboloid.f')

prob.setup()

# Set initial values.
prob.set_val('paraboloid.x0', 3.0)
prob.set_val('paraboloid.x1', -4.0)

# run the optimization
prob.run_driver();

# Get values of design variables
x0 = prob.get_val('paraboloid.x0')
x1 = prob.get_val('paraboloid.x1')
print('x0 = ', x0)
print('x1 = ', x1)