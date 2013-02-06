import numpy as np
from scipy.special import erf,erfinv
from .constants import s_in_y

def cmy2ms(v):
	return v/(100*s_in_y)

# See pg 158 Turcotte and Schubert
def T_hs(T0,T1,d,t,kappa):
	"""
	Gives the temperture at depth 'd' given the top ('T0') and bottom ('T1') thermal
	boundary conditions, and thermal diffusivity ('kappa')
	"""
	return T0+(T1-T0)*erf(d/(2*np.sqrt(kappa*t)))

def depth_hs(T0,T1,T,t,kappa):
	"""
	Gives the depth at where the temperature is T given a half-space cooling model
	with top and bottom thermal bounday conditions T0 and T1 and thermal diffusivity kappa
	"""
	return 2*np.sqrt(kappa*t)*erfinv( (T-T0)/(T1-T0) )

def age_hs(T0,T1,T,d,kappa):
	"""
	Gives the age of the plate when the temperature is T at depth d
	for a half-space cooling model with a top and bottom boundary conditions T0 and T1 and
	thermal diffusivity kappa.
	"""
	return (1/kappa)*(d/(2*erfinv( (T-T0)/(T1-T0) )))**2

def Ra(rho,g,alpha,delta_T,length,eta,kappa):
	return rho*g*alpha*delta_T*length**3/(eta*kappa)

def combinations(*args):
	c=[]
	l=args[0]
	for i in l:
		if len(args)>1:
			for i2 in combinations(*args[1:]):
				c.append( [i]+i2 )
		else:
			c.append( [i] ) 
	return c
		
