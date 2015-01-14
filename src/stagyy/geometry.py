import numpy as np
from scipy.optimize import newton

def expint(x):
#real function expint (x)
#  implicit none
#  real,intent(in)::x
#  if (x.lt.0) then
#     expint = exp(x)
#  else
#     expint = 2.0 - exp(-x)
#  end if
#end function expint
    return np.exp(x)*(x<0)+ (2.0-np.exp(-x))*(x>=0)

def g_of_z_exp(z,layers=[],dresl_topbot=1.0)
    """
    Each layer is a tuple:
    (normalized depth, dresl, wresl)
    Normalized depth is layer depth/domain deption [0:1]
    dresl is the approzimate refinement
    wresl is the width
    """
    a=2.0/(1.0+dresl_topbot)
    b=a*(dresl_topbot-1.0)
    
    d=1.0-z
    g0=a*z+0.5*b*z*z

    for depth,dresl,wresl in layers:
        e=(dresl-1)*(a+b*(1-depth))
        g0+=e*wresl*expint((depth-d)/wresl)

    return g0

def gnorm_of_z(z,g_of_z):
    return norm_of_f(z,g_of_z)

def z_of_gnorm(g,g_of_z):
    """
    Returns the values of z such that g=gnorm_of_z(z,g_of_z)
    """
    f=lambda g: gnorm_of_z(g,g_of_z)
    #return x_of_f_secant(g,f)
    return x_of_f(g,f)

def calc_zg2(nztot,g_of_z):
    iz2=np.arange(0.,nztot*2+1)
    g=iz2/(nztot*2)
    zg2=z_of_gnorm(g,g_of_z)
    return zg2

def norm_of_f(x,f,xmax=1.0,xmin=0.0):
    """
Normalizes the range of the function f to [0.0,1.0] over the domain [xmin,xmax]
    """
    if np.isscalar(x):
        bot=f(xmin)
        top=f(xmax)
    else:
        bot=f(xmin*np.ones(len(x)))
        top=f(xmax*np.ones(len(x)))
    return ( f(x)-bot) / ( top - bot )

def x_of_f_secant(y,f,x0=.5):
    return np.array([newton(lambda x: f(x)-yi,x0) for yi in y])

def x_of_f(y,f):
    return np.array([x_of_f_bisect(yi,f) for yi in y])

def x_of_f_bisect(y,f,errval=1e-6,xmax=1.0,xmin=0.0):
    """
For a function, y=f(x) and values y, find x such that  f(x)-y<errval
    """
    if(y==0.0 or y==1.0): return y
    xhi=xmax
    xlo=xmin
    x=0.5
    fx=f(x)
    while abs(fx-y)>errval:
        if y>fx: 
            xlo=x
        else : 
            xhi=x
        x=0.5*(xlo+xhi)
        fx=f(x)
    return x

class Grid(object):
    def __init__(self,N,L):
        self.L=np.array(L)
        self.N=np.array(N)
        self.cells=reduce(lambda x,y: max(1,x)*max(1,y), self.N-1)

class RegularGrid(Grid):
    def __init__(self,N,L):
        super(Grid, self).__init__(N,L)
        self.delta=np.array([ 0 if n==0 else l/n for l,n in zip(self.L,self.N-1)  ])
        # Calculate in the X direction
        self.xg=np.linspace(0,L[0],N[0]*2+1)
        self.x_center=self.xg[1::2]
        self.x_face=self.xg[::2]
        # Calculate in the Y direction
        self.yg=np.linspace(0,L[1],N[0]*2+1)
        self.y_center=self.yg[1::2]
        self.y_face=self.yg[::2]
        # Calculate in the Z direction
        self.zg=np.linspace(0,L[1],N[0]*2+1)
        self.z_center=self.zg[1::2]
        self.z_face=self.zg[::2]

class ExpGrid(object):
    def __init__(self,N,L,layers=[],dresl_topbot=1.0):
    """
    Each layer is a tuple:
    (normalized depth, dresl, wresl)
    Normalized depth is layer depth/domain deption [0:1]
    dresl is the approzimate refinement
    wresl is the width
    """
        super(Grid, self).__init__(N,L)
        # Calculate in the X direction
        self.xg=np.linspace(0,L[0],N[0]*2+1)
        self.x_center=self.xg[1::2]
        self.x_face=self.xg[::2]
        # Calculate in the Y direction
        self.yg=np.linspace(0,L[1],N[0]*2+1)
        self.y_center=self.yg[1::2]
        self.y_face=self.yg[::2]
        # Calculate in the Z
        g_of_z=lambda z: g_of_z_exp(z,layers,dresl_topbot)
        iz2=np.arange(0.,N[2]*2+1)
        g=iz2/(N[2]*2)
        self.zg=L[2]*z_of_gnorm(g,g_of_z)
        self.z_center=self.zg[1::2]
        self.z_face=self.zg[::2]



