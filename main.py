# 2015-Nov-16
# Script for the main program that will drive multiscale simulation


from functions import *
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint

# Number of sites along an edge of the square simple cubic lattice
# 	N^2 is the total number of atoms
N = 20

# Array with the number of atoms at every site
surfacemat = np.zeros( (N,N), 'int' )

# Array with the number of neighbours of each atom at every site
neighsmat = np.zeros( (N,N), 'int' )

# Array that tells how many atoms in total there are with 1, 2, 3, 4 and 5 neighbours
neighstally = np.zeros( 5, 'int' )

# Coupling time for the PDE gas phase and KMC solid-on-solid models
dtcouple = 0.1

# initialize KMC timestep (updated by the runSolidOnSolidKMC() function)
dtkmc = 0.0

# Total simulation time
Ttot = 1.0

# Current simulation time
Tcurr = 0.0

# Find the value of the 2nd derivative of the dependent variable in the Fluid Flow conservation equation at the first boundary
correct2ndderivative = fsolve( residualFluidFlowSS, 1.2 )

# Solve the Fluid Flow conservation equation at steady state (equation 3-1 of Shabnam's PhD thesis)
# get the values of the dependent variable and its first and second derivatives 
fvalues = calcFluidFlowSS( correct2ndderivative )

# The difference between adsorption and desorption rates (equation 3-20 of Shabnam's PhD thesis)
Ra_Rd = (Na-Nd)*np.power( 2.*a*np.power(N,2.)*dtcouple, -1. )

# Boundary condition (di_x/di_eta value) at eta = 0: equation 3-6 of Shabnam's PhD thesis
bc0 = Sc*(Ra_Rd)*np.power( 2.*a*mu_b*rho_b, -0.5 ) 

# Pack fvalues and the boundary condition value to find the values of mass transfer function
params = [ fvalues[:,0], bc0 ]

xinit = np.zeros( max(fvalues.shape), 'float' )

# Boundary condition for the mole fraction of the precursor (x) at infinite eta
xinit[-1] = 1.

# Obtain the eta values
_, _, _, _, eta, _, _, _ = FFSSparams()

# Find the mole fraction profile vs dimensionless distance for the next time step
xvalues = odeint( MassTransfMoL, xinit, np.array( [0.,dtcouple] ), args=(params,)  )

# Get the mole fraction of the precursor on the surface
xgrow = xvalues[0]

# Calculate Pa (equation 3-8 of Shabnam's thesis)
Pa = S0 * P * xgrow * np.power( 2.*np.pi*m*R*T, -0.5 ) * np.power( Ctot, -1.0 )

# Calculate Wa, Wd, Wm
Wa = Pa * np.power( N, 2. )
Wd = calcWd( neighstally )
Wm = calcWm( neighstally )

# Carry on with the simulation until the final time is reached
while Tcurr < Ttot:

	# Check the time, integrate PDE equations if coupling time has been reached
	if dtkmc < dtcouple:
		
		### microscale model (solid-on-solid) ###
		
		# calculate Wa, Wd, Wm
		Wa = Pa * np.power( N, 2. )
		Wd = calcWd( neighstally )
		Wm = calcWm( neighstally )
		
		# the update of dtkmc is done within the function
		dtkmc = runSolidOnSolidKMC( N, surfacemat, neighsmat, neighstally, Wa, Wd, Wm, dtkmc )
		
	else:
		
		# update the total simulation time
		Tcurr += dtkmc
		
		
		# run the PDE model to get the new boundary condition value and update xgrow for the KMC model
		runGasPhasePDE()



### Macroscale model (PDE model) ###


# Integrate the gas phase conservation equations


# Calculate boundary conditions

