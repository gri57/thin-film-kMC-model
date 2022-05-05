# 2015-Nov-17
# File with the functions called by main.py


#from contracts import contract
import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint


#@contract( N='int,>0', x='int,>=0,<N', y='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray,ndarray' )
def adsorption_event( N, x, y, surfacemat, neighsmat, neighstally ):
	
	""" 
	Perform an adsorption event and call the function that does a 
	neighbour update. 
	"""
	
	# Add an atom to the surface at the specified site.
	surfacemat[x,y] += 1
	
	# Update the number of neighbours at each site affected by adsorption, use periodic boundaries.
	neighsmat, neighstally = update_neighs_pbc( N, 'ads', x, y, surfacemat, neighsmat, neighstally )
	
	return surfacemat, neighsmat, neighstally


#@contract( N='int,>0', x='int,>=0,<N', y='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray,ndarray' )
def desorption_event( N, x, y, surfacemat, neighsmat, neighstally ):
	
	""" 
	Perform a desorption event and call the function that does a 
	neighbour update. 
	"""
	
	# Remove an atom from the surface at the specified site.
	surfacemat[x,y] -= 1
	
	if surfacemat[x,y] <= 0:
		print 'The number of atoms at a surface site is zero or less.'

	# Update the number of neighbours at each site affected by desorption, use periodic boundaries.
	neighsmat, neighstally = update_neighs_pbc( N, 'des', x, y, surfacemat, neighsmat, neighstally )
	
	return surfacemat, neighsmat, neighstally


#@contract( N='int,>0', xi='int,>=0,<N', yi='int,>=0,<N', xf='int,>=0,<N', yf='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray,ndarray' )
def migration_event( N, xi, yi, xf, yf, surfacemat, neighsmat, neighstally ):
	
	"""
	Perform a migration event (which is a desorption event followed by 
	adsorption). The functions for desorption and adsorption each call 
	the neighbour update function (with periodic boundary conditions).
	"""
	
	# Desorption event.
	surfacemat, neighsmat, neighstally = desorption_event( N, xi, yi, surfacemat, neighsmat, neighstally )
	
	# Adsorption event.
	surfacemat, neighsmat, neighstally = adsorption_event( N, xf, yf, surfacemat, neighsmat, neighstally )
	
	return surfacemat, neighsmat, neighstally


#@contract( N='int,>0', callerid='str', x='int,>=0,<N', y='int,>=0,<N', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', returns='ndarray,ndarray' )
def update_neighs_pbc( N, callerid, x, y, surfacemat, neighsmat, neighstally ):
	
	""" 
	Do a local update of neighbours for atoms affected by adsorption, 
	desorption, and migration events. Use periodic boundary conditions.
	"""
	
	# Retrieve the current height at the site.
	currh = surfacemat[x,y] 
	
	# Using the callerid flag, find out the height before the action of 
	# the function that called update_neighs_pbc.
	if callerid == 'ads':
		prevh = currh - 1
	elif callerid == 'des':
		prevh = currh + 1
	
	# Find out the coordinates of the four sites that are neighbours to
	# the site of interest (x,y). Periodic boundary conditions apply.
	neighcoords = 0*np.ndarray( (4,2), 'int' )
	neighcoords[0,...] = [x-1, y]
	neighcoords[1,...] = [x+1, y]
	neighcoords[2,...] = [x, y-1]
	neighcoords[3,...] = [x, y+1]
	
	# Implement periodic boundary conditions.
	if x-1 < 0:
		neighcoords[0,0] += N
	if x+1 >= N: 
		# this coordinate should never be greater than N, could be equal to it at most
		neighcoords[1,0] -= N
	if y-1 < 0:
		neighcoords[2,1] += N
	if y+1 >= N:
		# this coordinate should never be greater than N, could be equal to it at most
		neighcoords[3,1] -= N
	
	# Update neighbour count at (x,y) site.
	siteneighs = 1
	for i in range( 4 ):
		# 4 is the number of nearest neighbours that may be surrounding the atom of interest.
		
		if currh <= surfacemat[ neighcoords[i,0], neighcoords[i,1] ]:
			# For any site, a neighbouring site contributes to neighbour count if 
			# height of the site is <= height of the neighbouring site.
			siteneighs += 1
			
		# Update neighbour count at those sites that were affected by the adsorption/desorption event.
		if surfacemat[ neighcoords[i,0], neighcoords[i,1] ] <= currh and surfacemat[ neighcoords[i,0], neighcoords[i,1] ] > prevh:
			# current site now is but was not a neighbour of the other site 
			neighsmat[ neighcoords[i,0], neighcoords[i,1] ] += 1
		elif surfacemat[ neighcoords[i,0], neighcoords[i,1] ] > currh and surfacemat[ neighcoords[i,0], neighcoords[i,1] ] <= prevh:
			# current site is not but was a neighbour of the other site 
			neighsmat[ neighcoords[i,0], neighcoords[i,1] ] -= 1
			
	# Update the number of neighbours at the site of interest (x,y).
	neighsmat[x,y] = siteneighs
	
	# Update the tally of neighbours
	neighstally[0] = np.sum( neighsmat == 1 )
	neighstally[1] = np.sum( neighsmat == 2 )
	neighstally[2] = np.sum( neighsmat == 3 )
	neighstally[3] = np.sum( neighsmat == 4 )
	neighstally[4] = np.sum( neighsmat == 5 )
	
	
	return neighsmat, neighstally


#@contract( neighstally='ndarray', returns='float' )
def calcWd( neighstally ):
	
	"""
	Calculate the total desorption rate. 
	"""
	
	# preallocate the array for storing the values of desorption event rates
	Pd = np.ndarray( 5, 'float' )
	
	# populate the preallocated array with the calculated rates of desorption events
	for i in range(5):
		Pd[i] = calcPd( i+1 )
		
	# Calculate the total desorption rate
	Wd = np.sum( neighstally*Pd )
	
	return Wd


#@contract( neighstally='ndarray', f='int,>=1,<=5', returns='float' )
def calcWm( neighstally, f = 5 ):
	
	"""
	Calculate the total migration rate. 
	
	If default value for f is used (f=5), then this function is equation 3-15 of Shabnam's PhD thesis.
	If another value for f is provided (f>=1,f<5), then this function is being used
	to select one of five classes for performing a migration event (see page 27 of 
	Shabnam's PhD thesis, the sentences immediately following equation 3-12).
	
	For efficiency purposes, it is better to use this function with the default f value.
	"""
	
	# preallocate the array for the storage of the values of the rates of migration events
	Pm = np.ndarray( f, 'float' )
	
	# populate the preallocated array with calculated rates of migration events
	for i in range(f):
		Pm[i] = calcPm( i+1 )
		
	# Calculate the total migration rate
	Wm = np.sum( neighstally[0:f]*Pm )
	
	return Wm


#@contract( n='int,>=1,<=5', returns='float' )
def calcPd( n ):
	
	"""
	Calculate the individual rates of desorption events for each number 
	of neighbours for an atom (each atom can have from 1 to 5 neighbours).
	This function is equation 3-9 of Shabnam's PhD thesis.
	"""
	
	# some of the parameters are repeated in calcPm
	kd0 = 1e9 # 1/s
	Ed = 17e3 # cal/mol
	R = 1.987 # cal/K.mol
	T = 800. # Kelvin
	E = 17e3 # cal/mol
	
	nu0 = kd0 * np.exp( -Ed / (R*T) )
	
	Pd = nu0 * np.exp( -n * E / (R*T) )
	
	return Pd


#@contract( n='int,>=1,<=5', returns='float' )
def calcPm( n ):
	
	"""
	Calculate the individual rates of migration events for each number 
	of neighbours for an atom (each atom can have from 1 to 5 neighbours).
	This function is equation 3-11 of Shabnam's PhD thesis.
	"""
	
	# some of these parameters are repeated in calcPd
	Ed = 17e3 # cal/mol
	Em = 10.2e3 # cal/mol
	R = 1.987 # cal/K.mol
	T = 800. # Kelvin
	
	A = (Ed - Em)/(R*T) # equation 3-12 of Shabnam's PhD thesis
	
	Pm = A * calcPd( n )
	
	return Pm


#@contract( N='int,>0', surfacemat='ndarray', neighsmat='ndarray', neighstally='ndarray', dtkmc='float', Wa='float', Wd='float', Wm='float', Na='float', Nd='float', returns='float,float,float' )
def runSolidOnSolidKMC( N, surfacemat, neighsmat, neighstally, dtkmc, Wa, Wd, Wm, Na, Nd ):
	
	"""
	Microscale model - solid-on-solid with adsorption, migration and desorption.
	"""
	
	# Use Wa, Wd and Wm to choose between adsorption, desorption or migration (a/d/m)
	Wtotal_inv = np.power( Wa+Wd+Wm, -1. )

	# Get a random number on the half-open interval [1e-53,1.0) to decide which event to perform
	# 1e-53 is practically zero
	zeta = np.random.uniform( low=1e-53, high=1.0 )

	# In an if/elif/else structure, as soon as one of the conditions is true, the code under the 
	# condition is executed and all other conditions are skipped.

	if zeta < Wa*Wtotal_inv:
		
		# perform adsorption
		# choose a site randomly (pick a random index for the arrays, Python indexing starts at zero)
		x = np.random.random_integers(0,N-1)
		y = np.random.random_integers(0,N-1)
		surfacemat, neighsmat, neighstally = adsorption_event( N, x, y, surfacemat, neighsmat, neighstally )
		
		# update the count of adsorbed atoms
		Na += 1.
		
	elif zeta < (Wa+Wd)*Wtotal_inv:
		
		# perform desorption
		# choose a site randomly (pick a random index for the arrays, Python indexing starts at zero)
		x = np.random.random_integers(0,N-1)
		y = np.random.random_integers(0,N-1)
		surfacemat, neighsmat, neighstally = desorption_event( N, x, y, surfacemat, neighsmat, neighstally )
		
		# update the count of desorbed atoms
		Nd += 1.
		
	else:
		# perform migration
		# choose one of the 5 "classes" (atoms with 1 neighbour, 2, 3, 4 or 5)
		
		# it is already known that the value of zeta is between (Wa+Wd)/(Wa+Wd+Wm) and 1
		# Wm is calculated by summing Mi*Pm(i) products for i from 1 to 5 (as per equation 3-15 of Shabnam's PhD thesis)
		# zeta can be anywhere between (Wa+Wd)/(Wa+Wd+Wm(i=5)) and (Wa+Wd+Wm(i))/(Wa+Wd+Wm(i=5)), where 1<=i<=5 ("i" is an integer)
		
		# modify zeta to save on CPU time (less calculations); the expression below is: zeta*(Wa+Wd+Wm)-Wa-Wd
		zetamod = zeta * np.power( Wtotal_inv, -1. ) - Wa - Wd
		
		# find "i" value that matches to where zeta is on this discretized region
		Wm_vari = 0.0
		neighclass = 0 
		while zetamod > Wm_vari:
			neighclass += 1
			Wm_vari += calcPm( neighclass )
			# this "while" loop will stop as soon as zetamod <= Wm_vari, leaving us with the correct "neighclass" value
		del Wm_vari # housekeeping
		
		# find out the x and y indeces of atoms that have the specified number of neighbours
		indecesofatoms = np.where( neighsmat == neighclass )
		
		# find out how many atoms have the specified number of neighbours
		numofatoms = len( indecesofatoms[0] )
		
		# choose one of those atoms at random
		chosenatom = np.random.random_integers( 0, numofatoms-1 )
		
		# get the atom's x and y coordinates
		xi = indecesofatoms[0][chosenatom]
		yi = indecesofatoms[1][chosenatom]
		
		# randomly find the location xf,yf where the atom selected above will migrate to by adding/subtracting 1 to xi/yi
		chosenneighbour = np.random.random_integers( 0, 3 )
		if chosenneighbour == 0:
			xf = xi - 1
			yf = yi
			# use periodic boundary conditions
			if xf < 0:
				xf += N
		elif chosenneighbour == 1:
			xf = xi + 1
			yf = yi
			# use periodic boundary conditions
			if xf >= N:
				xf -= N
		elif chosenneighbour == 2:
			xf = xi
			yf = yi - 1
			# use periodic boundary conditions
			if yf < 0:
				yf += N
		elif chosenneighbour == 3:
			xf = xi
			yf = yi + 1
			# use periodic boundary conditions
			if yf >= N:
				yf -= N
		
		# Perform the migration event
		surfacemat, neighsmat, neighstally = migration_event( N, xi, yi, xf, yf, surfacemat, neighsmat, neighstally )


	# Increment the KMC timestep (equation 3-16 of Shabnam's PhD thesis)
	# 1e-53 is practically zero, but prevents np.log(sigma) from giving "-inf" for an answer
	sigma = np.random.uniform( low=1e-53, high=1.0 )
	dtkmc += -np.log( sigma ) * Wtotal_inv    # equation 3-16
	
	return surfacemat, neighsmat, neighstally, dtkmc, Na, Nd


#@contract( returns='float,float,float,float,ndarray,float,float,float' )
def FFSSparams():
	
	# Parameter values
	rho_b = 1.0
	rho = 1.0
	
	# Initial values of the dependent variable and its derivatives
	f0 = 0.0 # known value of the dependent variable at the first boundary 
	f_eta_0 = 0.0 # known value of the first derivative at the first boundary
	
	# Make an array for the independent variable (dimensionless distance)
	eta_0 = 0.
	eta_inf = 6.0
	d_eta = 0.5
	eta = np.arange( eta_0, eta_inf, d_eta )
	
	return rho_b, rho, f0, f_eta_0, eta, eta_0, d_eta, eta_inf


#@contract( f_eta_2_0='float', returns='ndarray' )
def calcFluidFlowSS(f_eta_2_0):
	
	"""
	Solve the fluid flow conservation equation at steady state (equation 3-1 in Shabnam's PhD thesis).
	f_eta_2_0  -  the initial value of the second derivative at the first boundary.
	
	This page has been helpful: http://www.physics.nyu.edu/pine/pymanual/html/chap9/chap9_scipy.html
	"""
	
	# Get the parameter values and intial values for the Fluid Flow conservation equation at steady state
	# skip the variables returned by FFSSparams() that are not necessary for these purposes
	rho_b, rho, f0, f_eta_0, eta, _, _, _ = FFSSparams()
	
	# Pack the parameter values for the ODE solver
	params = [rho_b, rho] 
	
	# Pack the initial values of the dependent variable and its derivatives
	fvars = [ f0, f_eta_0, f_eta_2_0 ] 
	
	# Use the IVP ODE solver imported from scipy.integrate
	fvalues = odeint( FluidFlowSS, fvars, eta, args=(params,)  )
	
	# Return the calculated values of the dependent variable and its first two derivatives
	return fvalues


#@contract( f_eta_2_0='float', returns='float' )
def residualFluidFlowSS(f_eta_2_0):
	
	"""
	Solve the fluid flow conservation equation at steady state (equation 3-1 in Shabnam's PhD thesis).
	
	This is a BVP ODE problem (not PDE, since the equation is solved at steady state).
	There are three known boundary conditions (dependent variable and its first derivative are 
	known at the first boundary, and the first derivative is known at the second boundary).
	Since the second derivative at the first boundary is not known, it has to be solved for 
	numerically by casting this problem as a roots finding problem.
	
	f_eta_2_0  -  the guess at the initial value of the second derivative at the first boundary
	This page has been helpful: http://www.physics.nyu.edu/pine/pymanual/html/chap9/chap9_scipy.html
	"""
	
	# Get the parameter values and intial values for the Fluid Flow conservation equation at steady state
	# skip the variables returned by FFSSparams() that are not necessary for these purposes
	rho_b, rho, f0, f_eta_0, eta, _, _, _ = FFSSparams()
	
	# Pack the parameter values for the ODE solver
	params = [rho_b, rho] 
	
	# Pack the initial values of the dependent variable and its derivatives
	fvars = [ f0, f_eta_0, f_eta_2_0 ] 
	
	# Use the IVP ODE solver imported from scipy.integrate
	fvalues = odeint( FluidFlowSS, fvars, eta, args=(params,)  )
	
	# Return the difference between the known value of the first derivative at the second boundary and its estimated value
	return 1.0 - fvalues[-1,1]


#@contract( fvars='list', eta='ndarray', params='list', returns='list' )
def FluidFlowSS( fvars, eta, params ):
	
	"""
	The fluid flow conservation equation at steady state - equation 3-1 
	of Shabnam's PhD thesis at steady state.
	This page has been helpful: http://www.physics.nyu.edu/pine/pymanual/html/chap9/chap9_scipy.html
	"""
	
	f, f_eta, f_eta_2 = fvars # unpack the dependent variable, and its first and second derivatives with respect to eta
	
	rho_b, rho = params # unpack the parameter values
	
	derivs = [ f_eta,
			   f_eta_2,
			   -f * f_eta_2 - 0.5 * ( rho_b/rho - f_eta**2. ) ]
	
	return derivs


#@contract( x='ndarray', tao='ndarray', params='list', returns='ndarray' )
def MassTransfMoL( x, tao, params ):
	
	"""
	The mass transfer equation - equation 3-3 of Shabnam's PhD thesis.
	
	x is the value of the dependent variable at all nodes. 
	Dimensions of x are the same as the derivs variable that is returned 
	by this function, and the f variable that holds the values of the 
	stream function.
	"""
	
	# Unpack the parameter values: the values of the stream function at steady 
	# state (f) and the boundary condition at eta = 0 (value of di_x/d_eta at eta = 0)
	f, bc0, Sc = params 
	
	# 1/Sc, where Sc is the Schmidt number of the precursor
	Sc_inv = 1./Sc
	
	# Get the value of the step size in dimensionless distance
	_, _, _, _, _, _, d_eta, _ = FFSSparams()
	
	# Calculate these values to be able to use multiplication instead of division (multiplication is faster)
	d_eta_inv = np.power( d_eta, -1. )
	d_eta2_inv = np.power( d_eta, -2. )
	
	# Preallocate the variable for storage of derivatives at all internal nodes
	derivs = np.ndarray( max(f.shape), 'float' )
	
	# Calculate x at eta = 0 using the reverse of forward difference approximation of the derivative
	x[0] = x[1] - d_eta * bc0

	# Calculate the time derivative of eta at the eta = 0 node.
	# forward difference approximation
	derivs[0] = Sc_inv * d_eta2_inv * ( x[2] - 2.*x[1] + x[0] ) + f[0] * d_eta_inv * ( x[1] - x[0] )
	
	# Calculate the time derivative of eta at each internal node.
	# central difference approximation
	# Using array math instead of a for loop for faster calculations. NumPy does not include the last value in the 
	# indeces used below: [1:-1] will include values from index 1 to index -2 (second last), not -1 (last).
	derivs[1:-1] = Sc_inv * d_eta2_inv * ( x[0:-2] - 2.*x[1:-1] + x[2:] ) + f[1:-1] * 2. * d_eta_inv * ( x[2:] - x[0:-2] )
	
	# The time derivative of eta at the eta = inf node is always zero because the boundary condition is x( eta=inf ) = X.
	derivs[-1] = 0.0
	
	return derivs


def runGasPhasePDE( N, dtcouple, Na, Nd, fvalues, xvalues, neighstally ):
	
	# Select parameter values from Table 3-1 of Shabnam's PhD thesis
	a = 5.0 # 1/s
	Ctot = 1.6611e-5 # sites.mol/m^2
	m = 0.028 # kg/mol
	P = 1e5 # Pa
	S0 = 0.1
	Sc = 0.75
	mu_b_rho_b = 9e11 # kg^2/(m^4.s)
	
	T = 800. # Kelvin
	R = 1.987 # cal/K.mol
	
	# The difference between adsorption and desorption rates (equation 3-20 of Shabnam's PhD thesis)
	Ra_Rd = ( Na - Nd ) * np.power( 2.*a*np.power(N,2.)*dtcouple, -1. )

	# Boundary condition (di_x/di_eta value) at eta = 0: equation 3-6 of Shabnam's PhD thesis
	bc0 = Sc*Ra_Rd*np.power( 2.*a*mu_b_rho_b, -0.5 ) 

	# Find the mole fraction profile vs dimensionless distance for the next time step
	# pass the stream function solution and the boundary condition value, and the Schmidt number, 
	# to solve the mass transfer function for the precursor mole fraction profile.
	# The solution contains the x profile and the derivatives at each node. 
	# Only the x profile is of interest.
	xvalues = odeint( MassTransfMoL, xvalues, np.array( [0.,dtcouple] ), args=([ fvalues[:,0], bc0, Sc ],)  )

	if np.sum( xvalues < 0.0 ) > 0.0:
		raise RuntimeError('At least one negative value is present in the precursor mole fraction profile.')

	# Get the mole fraction of the precursor on the surface
	xgrow = xvalues[0][0]

	# Calculate Pa (equation 3-8 of Shabnam's thesis)
	Pa = S0 * P * xgrow * np.power( 2.*np.pi*m*R*T, -0.5 ) * np.power( Ctot, -1.0 )

	# Calculate Wa, Wd, Wm
	Wa = Pa * np.power( N, 2. )
	Wd = calcWd( neighstally )
	Wm = calcWm( neighstally )
	
	return Wa, Wd, Wm, xvalues[0,:] 

