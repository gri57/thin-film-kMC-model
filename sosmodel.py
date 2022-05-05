import numpy as np
from scipy.optimize import fsolve
from scipy.integrate import odeint


class ThinFilm( object ):

	"""
	The class contains the attributes of and methods that can be done to
	the thin film surface formed using Kinetic Monte Carlo solid on 
	solid simulation.
	
	Attributes:
		N: 				(int) an integer number of the sites along the edge of the square film 
						(total number of atoms on surface is N*N).
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
		dtkmc:			(float) KMC timestep 
		Wa:				(float) total adsorption rate
		Wd: 			(float) total desorption rate
		Wm:				(float) total migration rate
		
	"""

	def __init__( self, N ):
		''' Return a new ThinFilm object. '''
		self.N = N
		
		self.surfacemat = np.ones( (N,N), 'int' )
		self.neighsmat = 5*np.ones( (N,N), 'int' )
		neighstally = np.zeros( 5, 'int' )
		neighstally[4] = N*N  # perfectly flat surface - initially all atoms have 5 neighbours
		self.Na = 0.
		self.Nd = 0.
		self.dtkmc = 0.
		self.Wa = 0.
		self.Wd = 0.
		self.Wm = 0.
		

class GasLayer( object ):
	
	"""
	The class contains the attributes of and methods that can be applied 
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
		f: 			stream function values at all dimensionless distance (eta) values
	
		OTHER attributes are explained when defined (below).
	
	"""

	def __init__( self ):
		''' Return a new GasLayer object. '''
		self.X = 2e-6
		self.eta_inf = 6.0
		self.d_eta = 0.1
		
		self.xprofile = np.ones( int( self.eta_inf/self.d_eta )+1, 'float' ) 
		self.xprofile *= self.X # initial precursor mole fraction profile is uniform, but during the 
								# simulation the values should decrease the closer we get to the surface 
		self.xgrow = self.xprofile[0] # @grigoriy - xgrow will not be updated automatically when xprofile is updated		

		self.eta = np.arange( 0., self.eta_inf+self.d_eta, self.d_eta ) # the length of xprofile, eta and f arrays must be the same
		
		self.f = np.zeros( int( self.eta_inf/self.d_eta )+1, 'float' ) 
		

		# Select parameter values from Table 3-1 of Shabnam's PhD thesis
		self.a = 5.0 # 1/s
		self.Ctot = 1.6611e-5 # sites.mol/m^2
		self.m = 0.028 # kg/mol
		self.P = 1e5 # Pa
		self.S0 = 0.1
		self.Sc = 0.75
		self.mu_b_rho_b = 9e11 # kg^2/(m^4.s)
		self.T = 800. # Kelvin
		self.R = 1.987 # cal/K.mol
		self.rho_b = 1. 
		self.rho = 1.
		
		# Boundary conditions for the stream function (equation 3-1) 
		self.f0 = 0. # value of the stream function at the eta = 0 boundary
		self.df_deta_0 = 0. # value of the first derivative of the stream function w.r.t. eta at the eta = 0 boundary
		self.df_deta_inf = 1.0 # value of the first derivative of the stream function w.r.t. eta at the eta = inf boundary
			

	#@contract( f_eta_2_0='float', returns='ndarray' )
	def calcFluidFlowSS( self, df_deta_2_0 ):
		
		"""
		Solve the fluid flow conservation equation at steady state (equation 3-1 in Shabnam's PhD thesis).
		df_deta_2_0  -  the initial value of the second derivative at the first boundary.
		
		This is a BVP ODE problem (not PDE, since the equation is solved at steady state).
		There are three known boundary conditions (dependent variable and its first derivative are 
		known at the first boundary, and the first derivative is known at the second boundary).
		Since the second derivative at the first boundary is not known, it has to be solved for 
		numerically (solution is achieved by casting this problem as a roots finding problem).
		
		This page has been helpful: http://www.physics.nyu.edu/pine/pymanual/html/chap9/chap9_scipy.html
		"""
	
		# Use the IVP ODE solver imported from scipy.integrate
		fvalues = odeint( self.FluidFlowSS, [ self.f0, self.df_deta_0, df_deta_2_0 ] , self.eta, args=( [ self.rho_b, self.rho ] ,)  )
		
		# Obtain and store the stream function values
		self.f = fvalues[:,0]

		# Return the difference between the known value of the first derivative at the second boundary and its estimated value
		return self.df_deta_inf - fvalues[-1,1]


	#@contract( fvars='list', eta='ndarray', params='list', returns='list' )
	@staticmethod
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
	
	# Unpack parameters
	# bc0: updated boundary condition for di_x/d_eta at eta = 0
	bc0, gaslayer = params
	
	# 1/Sc, where Sc is the Schmidt number of the precursor
	Sc_inv = 1./gaslayer.Sc
	
	# Calculate these values to be able to use multiplication instead of division (multiplication is faster)
	d_eta_inv = np.power( gaslayer.d_eta, -1. )
	d_eta2_inv = np.power( gaslayer.d_eta, -2. )
	
	# Preallocate the variable for storage of derivatives at all internal nodes
	derivs = np.ndarray( int( gaslayer.eta_inf/gaslayer.d_eta )+1, 'float' ) 
	'''
	print derivs.shape
	print gaslayer.f.shape
	print x.shape
	'''
	# Calculate x at eta = 0 using the reverse of forward difference approximation of the derivative
	x[0] = x[1] - gaslayer.d_eta * bc0

	# Calculate the time derivative of eta at the eta = 0 node.
	# forward difference approximation
	derivs[0] = Sc_inv * d_eta2_inv * ( x[2] - 2.*x[1] + x[0] ) + gaslayer.f[0] * d_eta_inv * ( x[1] - x[0] )
	
	# Calculate the time derivative of eta at each internal node.
	# central difference approximation
	# Using array math instead of a for loop for faster calculations. NumPy does not include the last value in the 
	# indeces used below: [1:-1] will include values from index 1 to index -2 (second last), not -1 (last).
	derivs[1:-1] = Sc_inv * d_eta2_inv * ( x[0:-2] - 2.*x[1:-1] + x[2:] ) + gaslayer.f[1:-1] * 2. * d_eta_inv * ( x[2:] - x[0:-2] )
	
	# The time derivative of eta at the eta = inf node is always zero because the boundary condition is x( eta=inf ) = X.
	derivs[-1] = 0.0
	
	return derivs


def calc_xgrow( thinfilm, gaslayer, dtcouple ):
	
	''' Calculate the precursor mole fraction on the surface of the thin film '''
	
	# The difference between adsorption and desorption rates (equation 3-20 of Shabnam's PhD thesis)
	Ra_Rd = ( thinfilm.Na - thinfilm.Nd ) * np.power( 2.*gaslayer.a*np.power(thinfilm.N,2.)*dtcouple, -1. )

	# Boundary condition (di_x/di_eta value) at eta = 0 (equation 3-6 of Shabnam's PhD thesis)
	bc0 = gaslayer.Sc*Ra_Rd*np.power( 2.*gaslayer.a*gaslayer.mu_b_rho_b, -0.5 ) 

	xdx_values = odeint( MassTransfMoL, gaslayer.xprofile, np.array( [0.,dtcouple] ), args=( [bc0, gaslayer], )  )
	
	# store the mole fraction profile within the gas boundary layer
	gaslayer.xprofile = xdx_values[0,:]
	
	# store the value of the mole fraction on the surface of the thin film
	gaslayer.xgrow = gaslayer.xprofile[0]

