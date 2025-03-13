import numpy as np
import openmdao.api as om


class Aerodynamic(om.ExplicitComponent):

    def setup(self):

        # Global Design Variable
        self.add_input("theta", val=np.zeros(2))

        # Local Design Variable
        self.add_input("d_t", val=np.ones(2))

        # Coupling parameter
        self.add_output("gamma", val=np.ones(2))

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        theta = inputs["theta"]
        d_t = inputs["d_t"]

        A = np.zeros((2, 2))
        b = np.zeros((2, 1))
        A[0, 0] = (theta[0] + d_t[0]) ** 2 + 3
        A[0, 1] = 1.0
        A[1, 0] = 1.0
        A[1, 1] = (theta[1] + d_t[1]) ** 2 + 5
        b[0, 0] = theta[0] + d_t[0]
        b[1, 0] = theta[1] + d_t[1]

        gamma = np.dot(np.linalg.inv(A), b)
        outputs["gamma"] = gamma.flatten()


class Structure(om.ExplicitComponent):

    def setup(self):
        # Global Design Variable
        self.add_input("theta", val=np.zeros(2))

        self.add_input("t", val=np.ones(2))

        self.add_input("gamma_t", val=np.ones(2))

        self.add_output("d", val=np.ones(2))

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        theta = inputs["theta"]
        t = inputs["t"]
        gamma_t = inputs["gamma_t"]

        K = np.zeros((2, 2))
        f = np.zeros((2, 1))
        K[0, 0] = 10 * t[0] - theta[0]
        K[0, 1] = 1.0
        K[1, 0] = 1.0
        K[1, 1] = 10 * t[1] - theta[1]
        f[0, 0] = gamma_t[0] ** 2
        f[1, 0] = gamma_t[1] ** 2

        d = np.dot(np.linalg.inv(K), f)

        outputs["d"] = d.flatten()


class Force(om.ExplicitComponent):

    def setup(self):
        # Global Design Variable
        self.add_input("theta", val=np.zeros(2))

        self.add_input("gamma", val=np.zeros(2))

        self.add_output("D", val=0.0)

        self.add_output("L", val=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):

        theta = inputs["theta"]
        gamma = inputs["gamma"]

        D = gamma[0] * np.sin(theta[0]) + gamma[1] * np.sin(theta[1])
        L = 10.0 * (gamma[0] + gamma[1])

        outputs["D"] = D
        outputs["L"] = L


class Cons(om.ExplicitComponent):
    def setup(self):
        self.add_input("gamma", val = np.zeros(2))

        self.add_input("gamma_t", val = np.zeros(2))

        self.add_input("d", val=np.ones(2))

        self.add_input("d_t", val = np.ones(2))

        self.add_output("h1", val=np.ones(2))
        self.add_output("h2", val = np.ones(2))

    def setup_partials(self):
        self.declare_partials("*", "*", method='fd')
    
    def compute(self, inputs, outputs):

        gamma = inputs["gamma"]
        gamma_t = inputs["gamma_t"]
        d = inputs["d"]
        d_t = inputs["d_t"]

        h1 = gamma - gamma_t
        h2 = d - d_t

        outputs["h1"] = h1
        outputs["h2"] = h2


class ModelGroup(om.Group):

    def setup(self):
        self.add_subsystem(
            "aero",
            Aerodynamic(),
            promotes=['*'],
        )

        self.add_subsystem(
            "struct",
            Structure(),
            promotes=['*'],
        )

        self.add_subsystem(
            "force",
            Force(),
            promotes=["*"],
        )

        self.add_subsystem(
            "stress_comp",
            om.ExecComp("stress = 10000*(d[0]+d[1])", d=np.array([0.0, 0.0])),
            promotes=["*"],
        )

        self.add_subsystem(
            "cons", 
            Cons(), 
            promotes=['*'],
            )


prob = om.Problem()
prob.model = ModelGroup()

prob.driver = om.ScipyOptimizeDriver()
prob.driver.options["optimizer"] = "SLSQP"
prob.driver.options["maxiter"] = 100
prob.driver.options["tol"] = 1e-6
prob.driver.options["debug_print"] = ["nl_cons", "objs", "desvars"]
# prob.driver.options["print_opt_prob"] = True

prob.model.set_input_defaults("theta", np.ones(2) * 0.1)
prob.model.set_input_defaults("t", np.ones(2) * 1.0)
prob.model.set_input_defaults("gamma_t", np.ones(2) * 1.0)
prob.model.set_input_defaults("d_t", np.ones(2) * 1.0)

prob.model.add_design_var("theta", lower=-1, upper=1)
prob.model.add_design_var("t", lower=0, upper=10)
prob.model.add_design_var("gamma_t", lower=-10, upper=10)
prob.model.add_design_var("d_t", lower=-10, upper=10)
prob.model.add_objective("D")
prob.model.add_constraint("L", equals=1)
prob.model.add_constraint("stress", upper=1)
prob.model.add_constraint("h1", equals=0)
prob.model.add_constraint("h2", equals=0)

# Ask OpenMDAO to finite-difference across the model to compute the gradients for the optimizer

prob.setup()
prob.set_solver_print(level=0)


prob.run_driver()

print("minimum found at")
print(prob.get_val("theta"))
print(prob.get_val("t"))

print("minumum objective")
print(prob.get_val("D"))
