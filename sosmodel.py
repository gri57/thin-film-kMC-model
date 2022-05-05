import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint
from matplotlib import pyplot as plt



class ThinFilm(object):

	""" The class contains the attributes of and methods that can be performed on
	the thin film surface during the Kinetic Monte Carlo solid-on-solid simulation.
	
	Attributes:
		N: 				(int) an integer number of the sites along the edge of the square film 
						(total number of atoms on surface is N*N)
						
		surfacemat: 	(ndarray 'int') array with the number of atoms at every site (start 
						with a perfectly flat surface)
						
		neighsmat: 		(ndarray 'int') array with the number of neighbours of each atom at every  
						site (start with a perfect surface - each atom has 4 closest neighbours in 
						the layer and 1 underneath)
						
		neighstally: 	(ndarray 'int') array that tells how many atoms in total there are with 1, 
						2, 3, 4 and 5 neighbours (perfectly flat surface - initially all atoms have 
						5 neighbours)
						
		Na: 			(int) integer to keep track of adsorbed atoms
		Nd: 			(int) integer to keep track of desorbed atoms
		Nm:				(int) integer to keep track of migrated atoms
		dtkmc:			(float) KMC timestep 
		Wa:				(float) total adsorption rate
		Wd: 			(float) total desorption rate
		Wm:				(float) total migration rate
		
		OTHER parameters are explained when they are defined (below).
		
	"""


	def __init__( self, N ):
		
		""" Return a new ThinFilm object """
		
		self.N = N
		
		self.surfacemat = np.ones( (self.N,self.N), 'int' )
		self.neighsmat = 5*np.ones( (self.N,self.N), 'int' )
		self.neighstally = np.zeros( 5, 'int' )
		self.neighstally[4] = self.N*self.N  # perfectly flat surface - initially all atoms have 5 neighbours
		self.Na = 0.
		self.Nd = 0.
		self.Nm = 0.
		self.dtkmc = 0.
		self.Wa = 0.
		self.Wd = 0.
		self.Wm = 0.
		
		self.R = 1.987 # cal/K.mol - gas constant (used in nu0, Wa requires 8.314)
		self.T = 800. # Kelvin

		# parameters from Table 3-1 of Shabnam's PhD thesis
		self.kd0 = 1e9 # 1/s
		self.E = 17e3 # cal/mol
		self.Ed = 17e3 # cal/mol
		self.Em = 10.2e3 # cal/mol
		self.Ctot = 1.6611e-5 # sites.mol/m^2
		self.m = 0.028 # kg/mol
		self.P = 1e5 # Pa
		self.S0 = 0.1 # the sticking coefficient
		
		self.nu0 = self.kd0 * np.exp(-self.Ed/(self.R * self.T)) # equation 3-10 of Shabnam's PhD thesis
		
		self.A = np.exp((self.Ed - self.Em)/(self.R * self.T)) # equation 3-12 
		
		# The rates of individual desorption events. The number of nearest 
		# neighbours (from 1 to 5) determines the individual desorption event rate.
		# ndarray of length 5
		self.Pd = self.nu0 * np.exp( -np.array( [1.,2.,3.,4.,5.] ) * self.E/(self.R * self.T) ) # equation 3-9
		
		# The rates of individual migration events
		# ndarray of length 5
		self.Pm = self.A * self.Pd # equation 3-11
		
		# The combination of equations 3-8 and 3-13, without xgrow (which will come from an instance of GasLayer)
		self.Wa_prefactor = self.S0 * self.P * np.power(2. * np.pi * self.m * 8.314 * self.T, -0.5) * np.power(self.Ctot, -1.) * np.power(self.N, 2.)
		
		

	def adsorption_event(self, x, y):
		
		""" Perform an adsorption event and call the function that does a neighbour update. 
		
		x:	(int) x coordinate of the atom on the thin film surface
		y:	(int) y coordinate of the atom on the thin film surface
		
		"""
		
		# Add an atom to the surface at the specified site.
		# surfacemat will be updated and stored
		self.surfacemat[x,y] += 1
		
		# Update the number of neighbours at each site affected by adsorption, use periodic boundaries.
		self.update_neighs_pbc( 'ads', x, y )
		
		return None

	
	def desorption_event( self, x, y ):
		
		""" Perform a desorption event and call the function that does a neighbour update. 
		
		x:	(int) x coordinate of the atom on the thin film surface
		y:	(int) y coordinate of the atom on the thin film surface
		
		"""
		
		# Remove an atom from the surface at the specified site.
		# surfacemat will be updated and stored
		self.surfacemat[x,y] -= 1
		
		# Update the number of neighbours at each site affected by desorption, use periodic boundaries.
		self.update_neighs_pbc( 'des', x, y )
		
		return None


	def migration_event( self, xi, yi, xf, yf ):
		
		""" Perform a migration event (which is a desorption event followed by adsorption). 
		The functions for desorption and adsorption each call the neighbour update function 
		(which uses periodic boundary conditions).
		
		xi:	(int) initial x coordinate of the atom
		yi:	(int) initial y coordinate of the atom
		xf:	(int) final x coordinate of the atom
		yf:	(int) final y coordinate of the atom
		
		"""
		
		# Desorption event.
		self.desorption_event( xi, yi )
		
		# Adsorption event.
		self.adsorption_event( xf, yf )
		
		return None


	def update_neighs_pbc( self, callerid, x, y ):
		
		""" Do a local update of neighbours for atoms affected by adsorption, desorption, and 
		migration events. Use periodic boundary conditions.
		
		callerid:	(str) character string that describes the function that called this function
		x:			(int) x coordinate of the atom on the thin film surface
		y:			(int) y coordinate of the atom on the thin film surface
		
		"""
		
		# Retrieve the current height at the site.
		currh = self.surfacemat[x,y] 
		
		# Using the callerid flag, find out the height before the action of 
		# the function that called update_neighs_pbc.
		if callerid == 'ads':
			prevh = currh - 1
		elif callerid == 'des':
			prevh = currh + 1
		
		# Find out the coordinates of the four sites that are neighbours to
		# the site of interest (x,y). Periodic boundary conditions apply.
		neighcoords = np.zeros( (4,2), 'int' )
		neighcoords[0,...] = [x-1, y]
		neighcoords[1,...] = [x+1, y]
		neighcoords[2,...] = [x, y-1]
		neighcoords[3,...] = [x, y+1]
		
		''' Implement periodic boundary conditions. '''
		
		if neighcoords[0,0] < 0:
			# this coordinate must not be less than zero
			neighcoords[0,0] += self.N
			
		if neighcoords[1,0] >= self.N: 
			# this coordinate should never be greater than N, could be equal to it at most
			neighcoords[1,0] -= self.N
			
		if neighcoords[2,1] < 0:
			# this coordinate must not be less than zero
			neighcoords[2,1] += self.N
			
		if neighcoords[3,1] >= self.N:
			# this coordinate should never be greater than N, could be equal to it at most
			neighcoords[3,1] -= self.N
		
		''' Update neighbour count at (x,y) site. '''
		
		# minimum possible number of nearest neighbours an atom may have (atom directly below is its only nearest neighbour)
		siteneighs = 1 
		
		for i in range( 4 ):
			# 4 is the max number of nearest neighbours that may be surrounding the atom of 
			# 	interest in its layer.
			# 5 is the maximum number of nearest neighbours an atom can have because the atom 
			# 	directly below is also its nearest neighbour.
			
			if currh <= self.surfacemat[ neighcoords[i,0], neighcoords[i,1] ]:
				# For any current site, a neighbouring site contributes to neighbour count if 
				# height of the current site is <= height of the neighbouring site.
				siteneighs += 1
				
			''' Update neighbour count at those sites that were affected by the adsorption/desorption event. '''
			
			if self.surfacemat[ neighcoords[i,0], neighcoords[i,1] ] <= currh and self.surfacemat[ neighcoords[i,0], neighcoords[i,1] ] > prevh:
				# current site now is but was not a neighbour of the other site 
				self.neighsmat[ neighcoords[i,0], neighcoords[i,1] ] += 1
				
			elif self.surfacemat[ neighcoords[i,0], neighcoords[i,1] ] > currh and self.surfacemat[ neighcoords[i,0], neighcoords[i,1] ] <= prevh:
				# current site is not but was a neighbour of the other site 
				self.neighsmat[ neighcoords[i,0], neighcoords[i,1] ] -= 1
				
		# Update the number of neighbours at the site of interest (x,y)
		# neighsmat will be updated and stored, accessible to other methods and functions
		self.neighsmat[x,y] = siteneighs
		
		# Update the tally of neighbours
		# neighstally will be updated and stored, accessible to other methods and functions
		self.neighstally[0] = np.sum( self.neighsmat == 1 )
		self.neighstally[1] = np.sum( self.neighsmat == 2 )
		self.neighstally[2] = np.sum( self.neighsmat == 3 )
		self.neighstally[3] = np.sum( self.neighsmat == 4 )
		self.neighstally[4] = np.sum( self.neighsmat == 5 )
		
		return None


	def calcWa(self, x_grow):
		
		""" Update and store the total adsorption rate. """
		
		# x_grow:	(float) the mole fraction of precursor on the surface, comes from calc_xgrow_PDE function
		self.Wa = self.Wa_prefactor * x_grow
		
		return None


	def calcWd(self):
		
		""" Update and store the total desorption rate. """
		
		# neighstally is constantly being changed by the KMC simulation, Pd stays the same
		self.Wd = np.sum(self.neighstally * self.Pd)
		
		return None


	def calcWm(self):
		
		""" Update and store the total migration rate. """
		
		# neighstally is constantly being changed by the KMC simulation, Pm stays the same
		self.Wm = np.sum(self.neighstally * self.Pm)
		
		return None

	def update_Wd_Wm_Wtot_inv( self ):	

		""" Update and store the total desorption and migration rates. """

		self.calcWd()
		self.calcWm()
		Wtotal = self.Wa + self.Wd + self.Wm
		
		return Wtotal, np.power(Wtotal, -1.)

class GasLayer(object):
	
	""" The class contains the attributes of and methods that can be applied 
	to the boundary gas layer above the thin film.
	
	Attributes:
		X: 			(float) bulk mole fraction
		eta_inf:	(float) maximum value of dimensionless distance away from the thin film (at eta_inf 
					the mole fraction is X).
		d_eta:		(float) discretization step in dimensionless distance 
		
		xprofile: 	(ndarray 'float') values of the mole fraction of the precursor at all values of 
					dimensionless distance.
		xgrow:		(float) mole fraction of the precursor on the surface of the thin film.
					Obtaining a value of xgrow is the whole purpose of the PDE model of the Gas Layer.
					The value of xgrow is what the KMC model needs from the PDE model.
		eta:		(ndarray 'float') values of the independent variable (dimensioness distance), minimum 
					is 0.0, max is eta_inf, step is d_eta.
		f: 			(ndarray 'float') stream function values at all dimensionless distance (eta) values
	
		OTHER attributes are explained when defined (below).
	
	"""

	def __init__(self):
		
		''' Return a new GasLayer object '''
		
		self.X = 2e-6
		self.eta_inf = 6.0
		self.d_eta = 0.1
		
		# Initial precursor mole fraction profile is uniform, but during the simulation 
		# the values should decrease the closer we get to the surface.
		self.xprofile = self.X * np.ones( int( self.eta_inf/self.d_eta )+1, 'float' ) 

		self.xgrow = self.xprofile[0] # @grigoriy - xgrow will not be updated automatically when xprofile is updated		

		self.eta = np.arange( 0., self.eta_inf+self.d_eta, self.d_eta ) # the length of xprofile, eta and f arrays must be the same
		
		self.f = np.zeros( int( self.eta_inf/self.d_eta )+1, 'float' ) 

		# Select parameter values from Table 3-1 of Shabnam's PhD thesis
		self.a = 5.0 # 1/s
		self.Sc = 0.75 # Schmidt number of the precursor
		self.mu_b_rho_b = 9e11 # kg^2/(m^4.s)
		self.rho_b = 1. 
		self.rho = 1.
		
		self.Sc_inv = 1./self.Sc
		
		self.RaRd_prefactor = self.Sc*np.power( 2. * self.a * self.mu_b_rho_b, -0.5 ) # for equation 3-6 in calc_xgrow_PDE() function
		
		# Boundary conditions for the stream function (equation 3-1) 
		self.f0 = 0. # value of the stream function at the eta = 0 boundary
		self.df_deta_0 = 0. # value of the first derivative of the stream function w.r.t. eta at the eta = 0 boundary
		self.df_deta_inf = 1.0 # value of the first derivative of the stream function w.r.t. eta at the eta = inf boundary

		# Calculate these to be able to use multiplication instead of division (the former is faster) when solving 
		# the mass transfer equation using Method of Lines (MassTransfMoL function).
		self.d_eta_inv = np.power( self.d_eta, -1. )
		self.d_eta2_inv = np.power( self.d_eta, -2. )
		self.Sc_inv_d_eta2_inv = self.Sc_inv * self.d_eta2_inv
		
		# optimization for MassTransfMoL function - the value is calculated in sosmain.py
		self.f_2_d_eta_inv = self.f[1:-1].copy() 

		# optimization for calc_xgrow_PDE function - the value is be calculated in sosmain.py
		self.eqn_3_20_denominator = 0.0 


	def calcFluidFlowSS(self, df_deta_2_0):
		
		"""
		Solve the fluid flow conservation equation at steady state (equation 3-1 in Shabnam's PhD thesis).
		
		df_deta_2_0: (float) the initial value of the second derivative at the first boundary.
		
		This is a BVP ODE problem (not PDE, since the equation is solved at steady state).
		There are three known boundary conditions (dependent variable and its first derivative are 
		known at the first boundary, and the first derivative is known at the second boundary).
		Since the second derivative at the first boundary is not known, it has to be solved for 
		numerically (solution is achieved by casting this problem as a roots finding problem).
		
		This page has been helpful: http://www.physics.nyu.edu/pine/pymanual/html/chap9/chap9_scipy.html
		
		"""
	
		# Use the IVP ODE solver imported from scipy.integrate
		fvalues = odeint( self.FluidFlowSS, [ self.f0, self.df_deta_0, df_deta_2_0 ] , self.eta, args=( [ self.rho_b, self.rho ] ,)  )
		# indexing fvalues as [:,0] will return values of f at all eta values (this is what's needed)
		# indexing fvalues as [:,1] will return values of df/deta at all eta values
		# indexing fvalues as [:,2] will return values of d2f/deta2 at all eta values
		
		# Obtain and store the stream function values
		self.f = fvalues[:,0]

		# Return the difference between the known value of the first derivative at the 
		# second boundary (eta = inf) and its estimated value. 
		# (float)
		return self.df_deta_inf - fvalues[-1,1]


	@staticmethod
	def FluidFlowSS(fvars, eta, params):
		
		"""
		('list', 'ndarray', 'list') -> 'list'
		
		The fluid flow conservation equation at steady state - equation 3-1 of Shabnam's PhD thesis, 
		written at steady state and as a system of first order ODEs.
		
		This page has been helpful: http://www.physics.nyu.edu/pine/pymanual/html/chap9/chap9_scipy.html
		
		"""
		
		f, f_eta, f_eta_2 = fvars # unpack the dependent variable, and its first and second derivatives with respect to eta
		
		rho_b, rho = params # unpack the parameter values
		
		derivs = [ f_eta,
				   f_eta_2,
				   -f * f_eta_2 - 0.5 * ( rho_b/rho - f_eta**2. ) ]
		
		return derivs



class Observables( object ):
	
	""" This class contains simulation results. 
	
	Attributes:
		N
		coupling_time
		total_time
		current_time
		roughness
		thickness
		growthrate
		surfacemat_previous
	
	"""
	
	def __init__(self, N, coupling_time, total_time, output_time):
		
		''' Return a new Observables object '''
		
		self.N = N # @grigoriy - this is a bit redundant with ThinFilm
		self.coupling_time = coupling_time
		self.total_time = total_time
		self.output_time = output_time
		
		self.Nsq_inv = np.power( self.N, -2. )
		
		# optimization for calculate_observables function
		self.Nsq_inv_coupling_time_inv = self.Nsq_inv / self.coupling_time 
		
		self.current_time = 0.0 # initial value, updated during simulation
		
		self.timepoints = np.zeros(int(round(self.total_time / self.coupling_time)) + 1, 'float') 
		
		self.roughness = np.zeros(int(round(self.total_time / self.coupling_time)) + 1, 'float') 
		
		self.thickness = np.zeros(int(round(self.total_time / self.coupling_time)) + 1, 'float')  # initial thickness is 1 (only one layer is deposited initially)
		
		self.growthrate = np.zeros(int(round(self.total_time / self.coupling_time)) + 1, 'float')  # initial growth rate is zero (at time zero)
		
		self.surfacemat_previous = np.ones( (self.N,self.N), 'int' ) # @grigoriy - this is a bit redundant with ThinFilm
		
		self.total_time_minus_1 = self.total_time - self.coupling_time # for the while loop in sosmain.py
		
	def calculate_observables(self, surfacemat, counter):
		
		''' Record the simulation time point '''
		self.timepoints[counter] = self.current_time
		
		''' Calculate roughness, growth rate and thickness '''
		
		# roughness - equation 3-17 of Shabnam's PhD thesis
		# find out the difference between current location and row immediately below
		# current row and row immediately above yields the same result, hence multiplication by 2
		# use "round" in conjunction with "int" (instead of only using int) to ensure that integer indeces are not skipped by "int"
		self.roughness[counter] += 2.*np.sum(np.abs(surfacemat[1:,:] - surfacemat[0:-1,:])) + 2.*np.sum(np.abs(surfacemat[0,:] - surfacemat[-1,:]))
		# the difference between the current column and column immediately before is the same as between current and immediately after
		self.roughness[counter] += 2.*np.sum(np.abs(surfacemat[:,1:] - surfacemat[:,0:-1])) + 2.*np.sum(np.abs(surfacemat[:,-1] - surfacemat[:,0]))
		self.roughness[counter] *= self.Nsq_inv
		# @grigoriy - the addition of 1 in equation 3-17 is handled at the end of sosmain.py (avoid redundant calculations)
		
		# thickness - equation 3-18
		self.thickness[counter] = np.sum(surfacemat) * self.Nsq_inv
		
		# growth rate - equation 3-19
		self.growthrate[counter] = np.sum(surfacemat - self.surfacemat_previous) * self.Nsq_inv_coupling_time_inv
		
		# store the matrix with heights at each site for the next growth rate calculation
		self.surfacemat_previous = surfacemat.copy()

		return None


	def __eq__(self, other):
		return self.current_time == other.output_time


def MassTransfMoL( x, timepoint, params ):
	
	"""
	('ndarray', 'float', 'list') -> 'ndarray'
	
	Numerical integration of the mass transfer equation - equation 3-3 of Shabnam's PhD 
	thesis - is done using the Method of Lines.
	
	The mass transfer equation  has been transformed from a PDE to ODEs at multiple eta 
	nodes. Each ODE is the time derivative of the mole fraction at a particular spatial node.
	
	x: 			mole fraction values at all spatial nodes (all eta values)
	timepoint:	time point at which derivatives are computed (not used in calculations explicitly)
	derivs: 	temporal derivatives of the mole fraction profile
	
	Dimensions of x are the same as the derivs variable that is returned by this
	function, and the gaslayer.f variable that holds the values of the stream function.
	
	"""
	
	# Unpack params
	bc0, gaslayer = params
	# bc0: 			updated boundary condition for di_x/d_eta (spatial derivative) at eta = 0
	# gaslayer: 	a GasLayer object (an instance of the GasLayer class)
	
	# Preallocate the variable for the storage of time derivatives at all nodes
	derivs = x.copy() * 0.0
	#	Multiplying by 0.0 is important since derivs[-1] must always be 0.0 (see 
	# 	the note at the end of this function).

	# Calculate x at eta = 0 using the reverse of forward difference approximation of the derivative
	# Use the boundary condition (spatial derivative at eta = 0) to arrive at the value of the dependent variable (x) at eta = 0
	x[0] = x[1] - gaslayer.d_eta * bc0

	# Calculate the time derivative of eta at the eta = 0 node.
	# forward difference approximation
	derivs[0] = gaslayer.Sc_inv_d_eta2_inv * ( x[2] - 2.*x[1] + x[0] ) + gaslayer.f[0] * bc0
	
	# Calculate the time derivative of eta at each internal node.
	# central difference approximation
	# Using array math instead of a for loop for faster calculations. NumPy does not include the last value in the 
	# indeces used below: [1:-1] will include values from index 1 to index -2 (second last), not -1 (last).
	derivs[1:-1] = gaslayer.Sc_inv_d_eta2_inv * ( x[0:-2] - 2.*x[1:-1] + x[2:] ) + gaslayer.f_2_d_eta_inv * ( x[2:] - x[0:-2] )
	
	''' The time derivative of eta at the eta = inf node is always zero because 
	the boundary condition is x( eta=inf ) = X. Hence, do nothing with derivs[-1] 
	as it is already 0.0 from when "derivs" variable was created. '''
		
	return derivs


def calc_xgrow_PDE( thinfilm, gaslayer, observables ):
	
	''' 
	Calculate the precursor mole fraction on the surface of the thin film, to be used by KMC. 
	
	xvalues: 	the array with mole fraction values at all time points provided to odeint; 
					the values are calculated using the Method of Lines
	
	'''
	
	# The difference between adsorption and desorption rates (equation 3-20 of Shabnam's PhD thesis)
	Ra_Rd = ( thinfilm.Na - thinfilm.Nd ) * gaslayer.eqn_3_20_denominator

	# Boundary condition (di_x/di_eta value) at eta = 0 (equation 3-6 of Shabnam's PhD thesis)
	bc0 = gaslayer.RaRd_prefactor * Ra_Rd

	# Use the IVP ODE solver imported from scipy.integrate
	xvalues = odeint( MassTransfMoL, gaslayer.xprofile, [observables.current_time, observables.current_time+observables.coupling_time], args=( [bc0, gaslayer] ,)  )
	''' Current time and the time point immediately following it were provided to odeint.
	 	Thus, xvalues[0,:] corresponds to x values at the "current time" and xvalues[1,:] 
	 	corresponds to x values at the time point immediately following (the latter is 
	 	what's needed).
	 	@grigoriy - NOTICE that the content of xvalues here is different from GasLayer.calcFluidFlowSS(...),
	 	in which the output of odeint ("fvalues") contained the values of f and its first and second derivatives. 
	 	Here xvalues contain only the values of x, not its derivatives, at different time points.'''

	# update and store the mole fraction profile within the gas boundary layer above the thin film
	gaslayer.xprofile = xvalues[1,:]
	
	# update and store the value of the mole fraction on the surface of the thin film
	gaslayer.xgrow = gaslayer.xprofile[0]
	
	return None


def run_sos_KMC(thinfilm, gaslayer):
	
	"""
	Microscale model - solid-on-solid with adsorption, migration and desorption.
	
	thinfilm:	instance of ThinFilm class
	gaslayer:	instance of GasLayer class
	
	"""
	
	# Calculate the total rates of adsorption, desorption and migration
	thinfilm.calcWa(gaslayer.xgrow)
	Wtotal, Wtotal_inv = thinfilm.update_Wd_Wm_Wtot_inv()

	''' Use Wa, Wd and Wm to select adsorption, desorption or migration (a/d/m) '''

	# Get a random number on the half-open interval [1e-53,1.0) to decide which event to perform
	# 1e-53 is practically zero
	zeta = np.random.uniform(low=1e-53, high=1.0)

	# In an if/elif/else structure, as soon as one of the conditions is true, the code under the 
	# condition is executed and all other conditions are skipped.

	if zeta < thinfilm.Wa*Wtotal_inv:
		
		''' perform adsorption '''
		
		# choose a site randomly (pick a random index for the arrays, Python indexing starts at zero)
		thinfilm.adsorption_event( np.random.random_integers(0, thinfilm.N-1), np.random.random_integers(0, thinfilm.N-1) )
		
		# update the count of adsorbed atoms
		thinfilm.Na += 1. # @grigoriy - this must be reset when thinfilm.dtkmc exceeds the coupling time
		
		
	elif zeta < (thinfilm.Wa+thinfilm.Wd)*Wtotal_inv:
		
		''' perform desorption '''
		
		# choose a site randomly (pick a random index for the arrays, Python indexing starts at zero)
		thinfilm.desorption_event( np.random.random_integers(0, thinfilm.N-1), np.random.random_integers(0, thinfilm.N-1) )
		
		# update the count of desorbed atoms
		thinfilm.Nd += 1. # @grigoriy - this must be reset when thinfilm.dtkmc exceeds the coupling time

				
	else:
		
		''' perform migration '''
		
		# choose one of the 5 "classes" (atoms with 1 neighbour, 2, 3, 4 or 5)
		
		# it is already known that the value of zeta is between (Wa+Wd)/(Wa+Wd+Wm) and 1
		# Wm is calculated by summing Mi*Pm(i) products, with i being between 1 to 5 (as per equation 3-15 of Shabnam's PhD thesis)
		# zeta can be anywhere between (Wa+Wd)/(Wa+Wd+Wm(i=5)) and (Wa+Wd+Wm(i))/(Wa+Wd+Wm(i=5)), where 1<=i<=5 ("i" is an integer)
		
		# modify zeta to save on CPU time (less calculations); the expression below is: zeta*(Wa+Wd+Wm)-Wa-Wd
		# zetamod value can be between 0 and Wm
		zetamod = zeta * Wtotal - thinfilm.Wa - thinfilm.Wd
		
		# find "i" (i.e. "neighclass") value that matches to where zeta is on this discretized region
		neighclass = 0
		Wm_varying = 0.
		while zetamod > Wm_varying:
			Wm_varying += thinfilm.neighstally[neighclass] * thinfilm.Pm[neighclass] 
			neighclass += 1
			# this "while" loop will stop as soon as zetamod <= Wm_varying, leaving us with the correct "neighclass" value

		# find out the x and y indeces of atoms that have the specified number of neighbours
		indecesofatoms = np.where( thinfilm.neighsmat == neighclass )
		
		# find out how many atoms have the specified number of neighbours
		numofatoms = len( indecesofatoms[0] )
		
		# choose one of those atoms at random
		chosenatom = np.random.random_integers( 0, numofatoms-1 )
		
		# get the atom's x and y coordinates
		xi = indecesofatoms[0][chosenatom]
		yi = indecesofatoms[1][chosenatom]
		
		''' randomly find the location xf,yf where the atom selected above 
		will migrate to by adding/subtracting 1 to xi/yi '''
		
		chosenneighbour = np.random.random_integers( 0, 3 )
		
		if chosenneighbour == 0:
			xf = xi - 1
			yf = yi
			# use periodic boundary conditions
			if xf < 0:
				xf += thinfilm.N
				
		elif chosenneighbour == 1:
			xf = xi + 1
			yf = yi
			# use periodic boundary conditions
			if xf >= thinfilm.N:
				xf -= thinfilm.N
				
		elif chosenneighbour == 2:
			xf = xi
			yf = yi - 1
			# use periodic boundary conditions
			if yf < 0:
				yf += thinfilm.N
				
		else:
			# chosenneighbour is 3
			xf = xi
			yf = yi + 1
			# use periodic boundary conditions
			if yf >= thinfilm.N:
				yf -= thinfilm.N
		
		# Perform the migration event
		thinfilm.migration_event(xi, yi, xf, yf)
		
		# update the count of migrated atoms
		thinfilm.Nm += 1. # @grigoriy - this must be reset when thinfilm.dtkmc exceeds the coupling time


	# Increment the KMC timestep (equation 3-16 of Shabnam's PhD thesis)
	# 1e-53 is practically zero, but prevents np.log(sigma) from giving "-inf" for an answer
	sigma = np.random.uniform( low=1e-53, high=1.0 )
	thinfilm.dtkmc += -np.log( sigma ) * Wtotal_inv    # equation 3-16 of Shabnam's PhD thesis
	# @grigoriy - must reset thinfilm.dtkmc when it exceeds the coupling time (done in sosmain.py)
	
	return None


def produce_output(observables):
	
	""" This function is called by sosmain.py to produce output 
	
	observables: 	instance of Observables class
	
	"""

	print 'Simulation timepoints'
	print observables.timepoints

	print 'Rougness results'
	print observables.roughness

	print 'Thickness results'
	print observables.thickness

	print 'Growth rate results'
	print observables.growthrate

	plt.figure()
	plt.xlabel( 'simulation timepoints' )
	plt.ylabel( 'surface roughness' )
	plt.title( 'Lattice size: '+str(observables.N)+'x'+str(observables.N) )
	plt.plot( observables.timepoints, observables.roughness )
	plt.savefig( 'roughness.png' )

	plt.figure()
	plt.xlabel( 'simulation timepoints' )
	plt.ylabel( 'growth rate' )
	plt.title( 'Lattice size: '+str(observables.N)+'x'+str(observables.N) )
	plt.plot( observables.timepoints, observables.growthrate )
	plt.savefig( 'growthrate.png' )

	plt.figure()
	plt.xlabel( 'simulation timepoints' )
	plt.ylabel( 'thickness' )
	plt.title( 'Lattice size: '+str(observables.N)+'x'+str(observables.N) )
	plt.plot( observables.timepoints, observables.thickness )
	plt.savefig( 'thickness.png' )

	return None

