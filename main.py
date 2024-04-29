import numpy as np
import matplotlib.pyplot as plt
#from   matplotlib.animation import FuncAnimation
import time

# TODO: implement a stable solution of the 3 body problem
# TODO: increase the simulation speed (with respect to real time)
# TODO: improve the visual quality of the simulation
# TODO: rewrite in OOP (each body has mass, position, velocity, ...)

# Newtonian Gravitation
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

#####################################################
#################### Integrators ####################
#####################################################

# Runge-Kutta 4
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

    # K3
    k3_v = []
    k3_a_compute = []
    for i in range(num_bodies): 
        k3_v.append(velocities[i] + dt/2 * k2_v[i])
        k3_a_compute.append(positions[i] + dt/2 * k2_a[i])
    k3_a = calculate_accelerations(k3_a_compute, masses)
    
    # K4
    k4_v = []
    k4_a_compute = []
    for i in range(num_bodies): 
        k4_v.append(velocities[i] + dt * k3_v[i])
        k4_a_compute.append(positions[i] + dt * k3_a[i])
    k4_a = calculate_accelerations(k4_a_compute, masses)

    # Integration
    for i in range(num_bodies):
        new_position = positions[i] + dt/6.0 * (k1_v[i] + 2*k2_v[i] + 2*k3_v[i] + k4_v[i])
        new_velocity = velocities[i] + dt/6.0 * (k2_a[i] + 2*k2_a[i] + 2*k3_a[i] + k4_a[i])

        new_positions.append(new_position)
        new_velocities.append(new_velocity)

    return new_positions, new_velocities

# Forward Euler
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

    # Integration
    for i in range(num_bodies):
        new_position = positions[i]  + dt * velocities[i]
        new_velocity = velocities[i] + dt * accelerations[i]
        
        new_positions.append(new_position)
        new_velocities.append(new_velocity)

    return new_positions, new_velocities

#####################################################
#################### Aesthetics #####################
#####################################################

def get_planet_sizes(masses, scale_factor=1):
    sizes = []
    # radius will be proportional to the cube root of mass
    for mass in masses:
        radius = mass**(1/3)

        if(scale_factor != 1):
            radius *= scale_factor

        sizes.append(radius)

    return sizes

def get_axes_limits(initPos):
    lim = 0
    for pos in initPos:
        for coordinate in pos:
            if abs(coordinate) > lim:
                lim = coordinate
    
    return lim * 2

#####################################################
#################### SIMULATION #####################
#####################################################

def simulate(initialPos, initialVel, masses, graphic_radii, real_radii, time_factor = 10, RK4 = False):
    """
    Perform the simulation of an n-body system, under newtonian gravity

    Parameters:
        initialPos (list of numpy arrays): List of position vectors of each body.
        initialVel (list of numpy arrays): List of velocity vectors of each body.
        masses (list of floats): List of masses of each body.
        graphic_radii (list of floats): Figure radius of each body.
        real_radii (list of floats): Real radius of each body (for collisions)
        time_factor (float): ratio between real time and sim time (10: sim runs 10x faster than real time)
        RK4 (bool): set the integrator: true -> RK4 is selected
    """

    # Defining Simulation Parameters
    dt = 1  # Sim Time step in seconds
    num_bodies = len(initialPos)

    # Rate at which the simulation must run (time_factor 1 => "real time")
    sleep_time = dt / time_factor 

    # Initial Conditions
    positions = initialPos
    velocities = initialVel

    # Creating the plot
    fig, ax = plt.subplots()
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    ax.grid(True, color='gray')
    ax_lim = get_axes_limits(initialPos)
    ax.set_xlim(-ax_lim, ax_lim)
    ax.set_ylim(-ax_lim, ax_lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    circles = []

    for i in range(num_bodies):
        circle = plt.Circle((initialPos[i][0], initialPos[i][1]), graphic_radii[i])
        ax.add_artist(circle)
        circles.append(circle)

    plt.show(block=False)

    if RK4:
        while True:
            new_positions, new_velocities = rk4_step(positions, velocities, masses, dt, num_bodies)
            positions = new_positions
            velocities = new_velocities

            for i in range(num_bodies):
                circles[i].center = (positions[i][0], positions[i][1])
                for j in range(num_bodies):
                    if i != j and np.linalg.norm(positions[i] - positions[j]) < (real_radii[i] + real_radii[j]):
                        print(f"COLLISION DETECTED between body {i} and body {j}")
                        print("Exiting program...")
                        return

            plt.draw()
            plt.pause(sleep_time)
            time.sleep(sleep_time)
    else: 
        while True:
            new_positions, new_velocities = fw_euler(positions, velocities, masses, dt, num_bodies)
            
            positions = new_positions
            velocities = new_velocities

            print(positions)

            for i in range(num_bodies):
                circles[i].center = (positions[i][0], positions[i][1])

            plt.draw()
            plt.pause(sleep_time)
            time.sleep(sleep_time)

def main(isSolarSystem = 1):

    if isSolarSystem == 1:
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
        graphic_radii = [696340000, 6371000, 1737400]
        
        real_radii = []
        for i in range(graphic_radii):
            real_radii.append(graphic_radii[i]*1e2)
        
        # Simulation Start
        simulate(initialPos, initialVel, masses, graphic_radii)

    elif isSolarSystem == 3:
        print("Generic 3Body System")
        initialPos = [np.array([-10.0, -5.0]), np.array([10.0, 5.0]), np.array([0.0, 0.0])]
        initialVel = [np.array([0.01, -0.01]), np.array([0.00, -0.00]), np.array([-0.01, 0.0])]
        masses = [1e5, 1e5, 1e5]
        graphic_radii = [1.0, 1.0, 1.0]
        simulate(initialPos, initialVel, masses, graphic_radii, graphic_radii, 1000000, False)

    elif isSolarSystem == 5:
        print("Stable Solution")

    else:
        initialPos = [np.zeros(2), np.array([10.0, 0.0])]
        initialVel = [np.zeros(2), np.array([0.0, 0.01])]
        masses = [100000.0, 1.0]
        graphic_radii = [5.0, 0.5]
        simulate(initialPos, initialVel, masses, graphic_radii, 10, True)

if __name__=="__main__":
    main(3)