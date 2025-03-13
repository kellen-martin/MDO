
import numpy as np
import openmdao as om
from ICA15 import Aero

model = om.Group()
model.add_susystem("Aero", Aero(), promotes=["*"])

prob = om.Problem(model)
prob.setup()

prob.set_val("theta", np.ones(2))
prob.set_val("d", np.zeros(2))

prob.run_model()
gamma = prob.get_val("gamma")
print("Gamma", gamma)