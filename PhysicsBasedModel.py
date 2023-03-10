import numpy as np
import matplotlib.pyplot as plt

'''Simple double mass spring damper model'''

import numpy as np
import matplotlib.pyplot as plt

# Define system parameters
m1 = 1.0  # mass of first block
k1 = 10.0  # spring constant of first spring
c1 = 0.5  # damping coefficient of first damper

# Define simulation parameters
t0 = 0.0  # initial time
tf = 10.0  # final time
dt = 0.01  # time step
n = int((tf - t0) / dt)  # number of time steps


# Define initial conditions
x1_0 = 0.0  # initial position of first block
v1_0 = 0.0  # initial velocity of first block

# Define state variables
x = np.zeros((3, n))  # position and velocity of each block
x[0, 0] = x1_0
x[1, 0] = v1_0
x[2, 0] = 0
# Define system dynamics function


def system_dynamics(x, t):
    # Extract states
    x1 = x[0]
    v1 = x[1]
    F_i = 1 + np.sin(t) + np.random.randn()*0.1

    # Compute accelerations
    a1 = (F_i - k1 * x1 - c1 * v1) / m1

    # Return state derivatives
    return np.array([v1, a1])


# Simulate system using fourth-order Runge-Kutta method
for i in range(1, n):
    t = t0 + i * dt

    k_1 = dt * system_dynamics(x[0:2, i-1], t)
    k_2 = dt * system_dynamics(x[0:2, i-1] + 0.5 * k_1, t + 0.5 * dt)
    k_3 = dt * system_dynamics(x[0:2, i-1] + 0.5 * k_2, t + 0.5 * dt)
    k_4 = dt * system_dynamics(x[0:2, i-1] + k_3, t + dt)

    a = 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4) / dt
    x[2, i] = a[1]
    x[0:2, i] = x[0:2, i-1] + 1/6 * (k_1 + 2*k_2 + 2*k_3 + k_4)

# Plot results
t = np.linspace(t0, tf, n)
F = 1 + np.sin(t) + np.random.randn(n)*0.1
F = F.reshape(-1, 1)


plt.plot(t, x[0, :], label='Block 1')
plt.xlabel('Time (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.show()

plt.plot(t, F, label='Block 1')
plt.xlabel('Time (s)')
plt.ylabel('Force (N)')
plt.legend()
plt.show()

np.save('input_train.npy', x)
np.save('output_train.npy', F)
np.save('time_train.npy', t)
