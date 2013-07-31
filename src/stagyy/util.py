import numpy as np
from scipy.special import erf,erfinv
from constants import s_in_y,R

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

def boundary_layer(ra,ra_ref):
    """Returns the relative boundary layer thickness for a system 
    with Rayleigh Number Ra given a reference Ra number"""
    return (ra_ref/ra)**(1./3.)

def find_val(a,b,f,y=[0],err=0.001,maxits=100000):
    """
Finds x where f(x)=y between a and b.
y may be a a scalar or list of values.
err is the relative error.  A value is considered 
equal if abs((f(x) - y)/y)<err.
maxits is the maximum number of itterations.
    """
    # TODO
    # This is a simple bisecting method (http://en.wikipedia.org/wiki/Bisection_method)
    # Perhaps a secant method would work better (http://en.wikipedia.org/wiki/Secant_method)
    p=[]
    if not hasattr(y,'__iter__'):
        y=[y]
    for y0 in y:
        a0=a
        b0=b
        it=0
        x = (a0+b0)/2
        v=f(x)
        while it<maxits and abs(y0-v)>err*y0:
            x = (a0+b0)/2
            v=f(x)
            if ((f(a0)-y0) * (v-y0)) > 0:
                a0 = x
            else:
                b0 = x
            it+=1
        p.append(x) 
    return p[0] if len(p)==1 else p

def arrhenius(E,V,eta_ref,T_ref,p_ref):
    return lambda T,p: eta_ref * np.exp( ((E+V*p)/(R*T)) - ((E+V*p_ref)/(R*T_ref)) ) 

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
		
