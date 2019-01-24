import matplotlib
import matplotlib.pyplot as plt

import scipy
from scipy import integrate
import numpy as np

# # parameters
WIRE_RADIUS = 0.6 * 10 ** -3  # WIRE RADIUS (M)
L = 100  # WIRE LENGTH (M)
E = 128 * 10 ** 9  # YOUNGS MODULUS (PA)
RHO = 8960  # DENSITY KG/M3
MU = RHO * np.pi * WIRE_RADIUS ** 2  # LINEAR DENSITY (KG/M)
N = 10  # NUM OF ELEMENTS
DELTAX = L / N  # CONNECTION LENGTH
ELEMENT_MASS = np.ones(N) * MU * DELTAX  # ARRAY OF MASSES OF EACH POINT OF SYSTEM
ELEMENT_MASS[0] = MU * DELTAX  # MASS OF SATELLITE
ELEMENT_MASS[-1] = MU * DELTAX  # MASS OF ENDMASS
DELTA_T = 0.1  # TIMESTEP
K = (E * np.pi * (WIRE_RADIUS ** 2)) / DELTAX  # SPRING CONSTANT OF EACH CONNECTION
C = 3 * N  # DAMPING OF EACH CONNECTION
NUM_TIMESTEPS = 10  # DURATION OF SIMULATION
TENSIONS = []  # A list of tensions on all points on the tehter at checked times


# -----------------------------------------


# physics equations

def int_acc(Ax, Bx, Av, Bv, C, ELEMENT_MASS, i):
    AxBx = Bx - Ax
    AxBxnorm = np.linalg.norm(Ax - Bx)
    AvBv = Bv - Av
    AxBx_hat = Ax-Bx / AxBxnorm
    vprojection = np.dot(AvBv, AxBx_hat) * AxBx_hat
    FBA = -C * vprojection
    Aacceleration = -FBA / ELEMENT_MASS[i]
    Bacceleration = FBA / ELEMENT_MASS[i + 1]
    if AxBxnorm - DELTAX > 0:  # Modelling the tether as a string, in which case it's never in compression
        F = K * (AxBx - DELTAX)
        FBA = F * AxBx_hat
        Aacceleration += -FBA / ELEMENT_MASS[i]
        Bacceleration += FBA / ELEMENT_MASS[i + 1]

    return [Aacceleration, Bacceleration]

def ext_acc(A):
    return [0, -9.81, 0]


def check_energy(ELEMENT_MASS, ELEMENT_VELOCITIES, ELEMENT_POSITIONS, DELTAX, K):
    """Finds energy of the whole system at a given time, useful debug tool"""
    energy = 0
    for i in range(N):  # Kinetic
        energy = energy + 0.5 * ELEMENT_MASS[i] * (np.linalg.norm(element_velocities[i, :])) ** 2
    for i in range(N - 1):  # spring potential
        sep = np.linalg.norm(ELEMENT_POSITIONS[i, :] - ELEMENT_POSITIONS[i + 1, :]) - DELTAX
        if sep > 0:  # Energy only stored in compression
            energy = energy + 0.5 * K * (
                    np.linalg.norm(ELEMENT_POSITIONS[i, :] - ELEMENT_POSITIONS[i + 1, :]) - DELTAX) ** 2
    for i in range(N):  # G potential
        energy = energy + ELEMENT_MASS[i] * 9.81 * ELEMENT_POSITIONS[i, 1]

    return energy


# ----------------------------------------


# SIMULATION EQUATIONS
# Fuctions to apply the physics along the length of the tether
def internal_accelerations(x, v, C, ELEMENT_MASS):
    a = np.zeros((N, 3))
    for i in range(N - 1):
        newa = int_acc(x[i, :], x[i + 1, :], v[i, :], v[i + 1, :], C, ELEMENT_MASS, i)

        a[i] = a[i] + newa[0]
        a[i + 1] = a[i + 1] + newa[1]

    return a



def external_accelerations(x):  # At the moment this is just gravity for testing purposes
    # Will be Handeled by James' Program
    a = np.ones((N, 3)) * np.array([0, -9.81, 0])
    return a


def tension_check(x, DELTAX, ELEMENT_MASS):
    inst_tensions = np.zeros(N)

    for i in range(N - 1):
        inst_tensions[i] = (np.linalg.norm(spring_accel(x[i, :], x[i + 1, :], DELTAX, ELEMENT_MASS, i)[0])
                            / ELEMENT_MASS[i])

    return inst_tensions


def diff_eq(t, y):
    """Takes input of Y, a 1D array of all position and velocity components appended,
    and outputs the derivatives of these components"""
    x = y[:3 * N]
    x = x.reshape(N, 3)
    vflat = y[3 * N:]
    for i in range(3):
        vflat[i] = 0

    v = vflat.reshape(N, 3)
    VDot = (internal_accelerations(x, v, C, ELEMENT_MASS) +
            external_accelerations(x))
    derv = np.append(vflat, VDot.reshape(3 * N))

    return [derv]


# ----------------------------------------
# Initial Conditions

# initialise arrays to store

# energy_history = []
ELEMENT_POSITIONS = np.zeros((N, 3))
ELEMENT_VELOCITIES = np.zeros((N, 3))

# initial conditions
for i in range(N):
    position_vector = [DELTAX * i, 0, 0]  # where each point sits
    ELEMENT_POSITIONS[i, :] = position_vector  # an array stores the entire position of the system

# ELEMENT_POSITIONS[-1,:] = ELEMENT_POSITIONS[-1,:] - np.array([2,0,0])    #initial condition parameter

# initial energy
# energy = checkEnergy(element_mass, element_velocities, element_positions, deltaX) #stores energy(debug tool)
# energy_history = np.append(energy_history, energy)

# ----------------------------------------
# Simulation
# initial plot
# print(0, "/", num_timesteps, " timesteps")
# print(0, "s")

plt.plot(ELEMENT_POSITIONS[:, 0], ELEMENT_POSITIONS[:, 1])
plt.show()

# creating 1 dimensional position and velocity arrays
FLAT_ELEMENT_POSITIONS = ELEMENT_POSITIONS.reshape(3 * N)
FLAT_ELEMENT_VELOCITIES = ELEMENT_VELOCITIES.reshape(3 * N)

for i in range(NUM_TIMESTEPS + 1):
    Y = np.append(FLAT_ELEMENT_POSITIONS, FLAT_ELEMENT_VELOCITIES)
    a_t = (0, DELTA_T)
    asol = scipy.integrate.solve_ivp(diff_eq, a_t, Y, vectorized=True, method='RK45')

    finsol = asol.y[:, -1]
    FLAT_ELEMENT_POSITIONS = finsol[0:(3 * N)]
    ELEMENT_POSITIONS = FLAT_ELEMENT_POSITIONS.reshape(N, 3)
    FLAT_ELEMENT_VELOCITIES = finsol[3 * N:]
    #    #check total energy of system
    #    if t_step%100 == 0:
    #    energy = checkEnergy(element_mass, element_velocities, element_positions, deltaX)
    #    energy_history = np.append(energy_history, energy)
    #

    ##    #monitor progress

    #    print(i, "/", num_timesteps, " timesteps")
    #    print(i, "s")

    plt.plot(ELEMENT_POSITIONS[:, 0], ELEMENT_POSITIONS[:, 1])
    axes = plt.gca()
    axes.set_ylim([-300, 100])
    axes.set_xlim([-100, 100])
    plt.show()

# print("Energy over time")
# plt.plot(energy_history[:])
# plt.show()
