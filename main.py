# 2015-Nov-16
# Script for the main program that will drive multiscale simulation

from functions import *

### Macroscale model (PDE model) ###

# Integrate the conservation equations

# Calculate boundary conditions


### Microscale model (solid-on-solid) ###

# Variables - N of lattice sites, KMC time, coupling time

# Calculate Wa, Wd, Wm

# Use Wa, Wd and Wm to choose between adsorption, desorption or migration (a/d/m)

# If migration was chosen, choose one of the 5 classes (atoms with 1 neighbour, 2, 3, 4 or 5)

# Perform the a/d/m event - 3 separate functions for each event

# Check the time, integrate PDE equations if coupling time has been reached



