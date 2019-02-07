import numpy as np
import scipy
import sys
from scipy import integrate
from functools import partial
# -----------------------------------------
class Tether:
    def __init__(self, length, linear_density, spring_constant, damping_constant, number_of_elements, mass_0, mass_L,
                 positions, velocities, forces):
        self.length = length
        self.linear_density = linear_density
        self.spring_constant = spring_constant
        self.damping_constant = damping_constant
        self.number_of_elements = number_of_elements
        self.positions = positions
        self.velocities = velocities
        self.accelerations = np.zeros((number_of_elements, 3))
        self.delta_x = self.length / self.number_of_elements
        self.masses = np.zeros(number_of_elements) + linear_density * self.delta_x
        self.masses[0] = mass_0
        self.masses[-1] = mass_L
        self.delta_t = 0
        self.tensions = np.zeros(number_of_elements - 1)
        self.fab = np.zeros(number_of_elements - 1)
        self.fba = np.zeros(number_of_elements - 1)
        self.positions_history = []
        self.tension_history = []
        self.forces = forces

    def internal_accelerations(self, x, v):
        # damp
        AxBx = x[1:self.number_of_elements, :] - x[:self.number_of_elements - 1, :]
        AxBxnorm = np.linalg.norm(AxBx, axis=1)[np.newaxis].T
        AvBv = v[1:self.number_of_elements, :] - v[:self.number_of_elements - 1, :]
        AxBx_hat = AxBx / AxBxnorm

        AvBv_dot_AxBx_hat = (AvBv * AxBx_hat).sum(axis=1)[np.newaxis].T
        vprojection = AvBv_dot_AxBx_hat * AxBx_hat

        self.fba = -self.damping_constant * vprojection

        # spring
        indices = np.arange(self.number_of_elements - 1)[np.newaxis].T
        mask = indices[AxBxnorm - self.delta_x > 0]
        F = self.spring_constant * (AxBxnorm - self.delta_x)
        self.fab = F * AxBx_hat



        Aacceleration = -self.fba / self.masses[:self.number_of_elements - 1][np.newaxis].T
        Bacceleration = self.fba / self.masses[1:self.number_of_elements][np.newaxis].T

        Aacceleration_ = self.fab / self.masses[:self.number_of_elements - 1][np.newaxis].T
        Bacceleration_ = -self.fab / self.masses[1:self.number_of_elements][np.newaxis].T

        a = np.zeros((self.number_of_elements, 3))
        a[:self.number_of_elements - 1, :] += Aacceleration
        a[1:self.number_of_elements, :] += Bacceleration

        a[:self.number_of_elements - 1, :][mask] += Aacceleration_[mask]
        a[1:self.number_of_elements, :][mask] += Bacceleration_[mask]

        return a

    def tension_check(self):
        t = self.fba + self.fab
        t = np.linalg.norm(t)
        return t

    def external_accelerations(self):
        a = self.forces.reshape(self.number_of_elements,3)*self.delta_x / self.masses
        a[:3] = a[:3] / 2
        a[-3:] = a[-3:] / 2
        return a

    def diff_eq(self, t, y):
        """Takes input of Y, a 1D array of all position and velocity components appended,
        and outputs the derivatives of these components"""
        x_flat = y[:3 * self.number_of_elements]
        x = x_flat.reshape(self.number_of_elements, 3)
        v_flat = y[3 * self.number_of_elements:]
        for i in range(3):
            v_flat[i] = 0

        v = v_flat.reshape(self.number_of_elements, 3)
        a = (self.internal_accelerations(x, v) + self.external_accelerations())
        a_flat = a.reshape(3 * self.number_of_elements)
        derv = np.append(v_flat, a_flat)

        return [derv]

    def update_step(self):
       try:
            Y = np.append(self.positions, self.velocities)
            a_t = (0, self.delta_t)
            asol = scipy.integrate.solve_ivp(self.diff_eq, a_t, Y, vectorized=True, method='RK45')
            finsol = asol.y[:, -1]
            self.positions = finsol[0:(3 * self.number_of_elements)]
            self.velocities = finsol[3 * self.number_of_elements:]
       except as err:
           stepped_ok = 0
           self.velocities = 0
           self.tensions = 0
           self.tension = ['Unexpected Error:', sys.exc_info()[0] ]
       else:
            stepped_ok = 1
            self.tensions = self.tension_check()

        return stepped_ok, self.positions, self.velocities, self.tensions

num_input_lines = 2

for i, line in enumerate(sys.stdin):
    if i < 9:
        if i == 0:
            length = float(line)
        elif i == 1:
            linear_density = float(line)
        elif i == 2:
            spring_constant = float(line)
        elif i == 3:
            damping_constant = float(line)
        elif i == 4:
            num_elements = int(line)
        elif i == 5:
            mass_0 = float(line)
        elif i == 6:
            mass_L = float(line)
        elif i == 7:
            positions = np.array([float(x) for x in line.split(',')])
        elif i == 8:
            velocities = np.array([float(x) for x in line.split(',')])

    else:
        if i % 2 == 1:
            time_step = int(line)
            if time_step == -1:
                break
        else:
            forces = np.array([float(x) for x in line.split(',')])
            tether_object = Tether(length, linear_density, spring_constant*num_elements, damping_constant*num_elements, num_elements, mass_0,
                                   mass_L, positions, velocities, forces)
            stepped_ok, positions, velocities, tensions = tether_object.update_step()
            if stepped_ok == 1:
                print(stepped_ok)
                print(",".join([str(x) for x in positions]))
                print(",".join([str(x) for x in velocities]))
                print(",".join([str(x) for x in tensions]))





