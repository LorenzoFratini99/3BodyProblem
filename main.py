import numpy as np
import matplotlib.pyplot as plt
#from   matplotlib.animation import FuncAnimation
import time

def calculate_accelerations(positions, masses, G = 6.67430e-11):
    """
    Calculate the acceleration experienced by each body due to the gravitational forces
    of all other bodies. Newtonian Gravitational law.

    Parameters:
        positions   (list of numpy arrays): Array of position vectors of each body.
        masses      (list):                 Array of masses of each body.
        G           (float, optional):      Gravitational constant (default is 6.67430e-11 m^3 kg^-1 s^-2).

    Returns:
        (list of numpy arrays): Array of acceleration vectors for each body.
    """
    
    num_bodies = len(positions)
    
    accelerations = []

    for i in range(num_bodies):
        acceleration = np.zeros_like(positions[i])
        for j in range(num_bodies):
            # The acceleration of body i will depend on the interaction with all bodies, except itself
            if i != j:
                # Computing the distance vector
                r_ij = positions[j] - positions[i]
                distance = np.linalg.norm(r_ij)

                # Computing the acceleration Magnitude, Newtonian Gravity
                acceleration_magnitude = G * masses[j] / distance**2
                 
                # Summing up the components of the acceleration vector
                acceleration += acceleration_magnitude * (r_ij / distance)
        
        accelerations.append(acceleration)

    return accelerations

def rk4_step(positions, velocities, masses, dt, num_bodies):
    """
    Perform a single step of the Rk4_v integration method.

    Parameters:
        positions (list of numpy arrays): List of position vectors of each body.
        velocities (list of numpy arrays): List of velocity vectors of each body.
        masses (list of floats): List of masses of each body.
        dt (float): Time step.

    Returns:
        (list of numpy arrays): Updated positions after one time step.
        (list of numpy arrays): Updated velocities after one time step.
    """
    new_positions = []
    new_velocities = []

    # K1
    k1_v = velocities
    k1_a = calculate_accelerations(positions, masses)

    # K2
    k2_v = []
    k2_a_compute = []
    for i in range(num_bodies):
        k2_v.append(velocities[i] + dt/2 * k1_v[i])
        k2_a_compute.append(positions[i] + dt/2 * k1_a[i])
    k2_a = calculate_accelerations(k2_a_compute, masses)

    k3_v = []
    k3_a_compute = []
    for i in range(num_bodies): 
        k3_v.append(velocities[i] + dt/2 * k2_v[i])
        k3_a_compute.append(positions[i] + dt/2 * k2_a[i])
    k3_a = calculate_accelerations(k3_a_compute, masses)
    
    k4_v = []
    k4_a_compute = []
    for i in range(num_bodies): 
        k4_v.append(velocities[i] + dt * k3_v[i])
        k4_a_compute.append(positions[i] + dt * k3_a[i])
    k4_a = calculate_accelerations(k4_a_compute, masses)

    for i in range(num_bodies):
        new_position = positions[i] + dt/6.0 * (k1_v[i] + 2*k2_v[i] + 2*k3_v[i] + k4_v[i])
        new_velocity = velocities[i] + dt/6.0 * (k2_a[i] + 2*k2_a[i] + 2*k3_a[i] + k4_a[i])

        new_positions.append(new_position)
        new_velocities.append(new_velocity)

    return new_positions, new_velocities

def fw_euler(positions, velocities, masses, dt, num_bodies):
    """
    Perform a single step of the Forward Euler integration method.

    Parameters:
        positions (list of numpy arrays): List of position vectors of each body.
        velocities (list of numpy arrays): List of velocity vectors of each body.
        masses (list of floats): List of masses of each body.
        dt (float): Time step.

    Returns:
        (list of numpy arrays): Updated positions after one time step.
        (list of numpy arrays): Updated velocities after one time step.
    """

    new_positions = []
    new_velocities = []

    accelerations = calculate_accelerations(positions, masses)

    for i in range(num_bodies):
        new_position = positions[i]  + dt * velocities[i]
        new_velocity = velocities[i] + dt * accelerations[i]
        
        new_positions.append(new_position)
        new_velocities.append(new_velocity)

    return new_positions, new_velocities


def simulate(initialPos, initialVel, masses, time_factor = 10, Rk4_v=False):
    dt = 1e-1  # Time step in seconds
    num_bodies = len(initialPos)

    # computing the rate at which the simulation must run (time_factor 1 => "real time")
    sleep_time = dt / time_factor 

    # creating the necessary data for the simulation
    positions = initialPos
    velocities = initialVel

    # initial plot

    if Rk4_v:
        while True:
            new_positions, new_velocities = rk4_step(positions, velocities, masses, dt, num_bodies)
            positions = new_positions
            velocities = new_velocities

            print(new_velocities)


            # update plot

            time.sleep(sleep_time)
    else: 
        while True:
            new_positions, new_velocities = fw_euler(positions, velocities, masses, dt, num_bodies)
            
            # update plot
            print(new_positions)

            positions = new_positions
            velocities = new_velocities


            time.sleep(sleep_time)

    # fig, ax = plt.subplots()
    # lines = [ax.plot([], [], label=f'Body {i+1}')[0] for i in range(num_bodies)]
    # scatter_points = ax.scatter(initialPos[:, 0], initialPos[:, 1], s=200, color='black')

    # def update(frame):
    #     nonlocal positions, velocities
    #     positions, velocities = rk4_step(positions, velocities, masses, dt)
    #     for i, line in enumerate(lines):
    #         line.set_data(positions[i, 0], positions[i, 1])
    #     return lines

    # #ani = FuncAnimation(fig, update, frames=None, blit=True, interval=50, repeat=False)
    # ax.legend()
    # ax.set_xlim(-2e11, 2e11)  # Adjust these limits based on your simulation scale
    # ax.set_ylim(-2e11, 2e11)  # Adjust these limits based on your simulation scale
    # plt.show()

def main(isSolarSystem = True):

    if isSolarSystem:
        print("Sun, Earth, Moon")
        # Relevant Distances
        distanceMoonEarth = 384400000.0    #[m]
        distanceSunEarth  = 149597870700.0 #[m]

        # CI: Positions
        initPosSun   = np.zeros(2) # x, y
        initPosEarth = np.array([0.0, distanceSunEarth])
        initPosMoon  = np.array([0.0, distanceSunEarth + distanceMoonEarth])
        initialPos   = [initPosSun, initPosEarth, initPosMoon]

        # CI: Velocities
        initVelSun   = np.zeros(2) # vx, vy
        initVelEarth = np.array([29783.0, 0.0]) # vx, vy
        initVelMoon  = np.array([29783.0 + 1022.0, 0.0]) # vx, vy
        initialVel   = [initVelSun, initVelEarth, initVelMoon]

        # Masses
        masses = [1.9891e30, 5,972e24, 7.34767309e22] # sun, earth, moon
        
        # Simulation Start
        simulate(initialPos, initialVel, masses)

    else:
        print("Fake execution")
        initialPos = [np.zeros(2), np.array([10.0, 0.0])]
        initialVel = [np.zeros(2), np.array([0.0, 0.01])]
        masses = [100000000.0, 1.0]
        simulate(initialPos, initialVel, masses, 100, True)

if __name__=="__main__":
    main(False)