from pyxdsm.XDSM import XDSM, OPT, SOLVER, FUNC, LEFT

# Change `use_sfmath` to False to use computer modern
x = XDSM(use_sfmath=True)

x.add_system("opt", OPT, r"\text{Optimizer}")
x.add_system("solver", SOLVER, r"\text{RK45}")
x.add_system('D1', FUNC, "\Delta v = \Delta v_{\\theta} + \Delta v_r")


x.connect("opt", "solver", "\Delta v_{\\theta}, \Delta v_r")
x.connect("opt", "D1", "\Delta v_{\\theta}, \Delta v_r")
x.connect("solver", "opt", "r_{final}")
x.connect("D1", "opt", "\Delta v")


x.add_output("opt", "\Delta v_{\\theta}^*, \Delta v_r^*", side=LEFT)
x.add_output("solver", "r_{final}^* ", side=LEFT)
x.add_output("D1", "\Delta v^*", side=LEFT)

x.write("mdf")