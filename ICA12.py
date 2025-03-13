import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

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


## Initial Conditions
theta0 = 0.0
r0 = 1.0
v_theta0 = 1.0
v_r0 = 0.0
State0 = [theta0, r0, v_theta0, v_r0]
tspan = [0, 5]
print(r0)

## Solve IVP
sol = solve_ivp(dynEqn, tspan, State0, max_step = 0.1)
print("Test")
thetas = sol.y[0]
rs = sol.y[1]
print(rs[-1])
## Plot
fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
ax.plot(thetas, rs)
ax.set_rmax(3)
plt.show()



