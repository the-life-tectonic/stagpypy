import logging 
import time
import numpy as np
from scipy.optimize import newton

LOG=logging.getLogger(__name__)

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

def g_of_z_exp(z,layers=[],dresl_topbot=1.0):
    """
    Each layer is a tuple:
    (normalized depth, dresl, wresl)
    Normalized depth is layer depth/domain depth [0:1]
    dresl is the approximate refinement
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

def interpolate_h5_xz(h5,Lx,Lz,px=None,pz=None):
    from scipy import interpolate
    #from mpl_toolkits.basemap import interp
    if 'data' in h5:
        frames,nx,ny,nz=h5['data'].shape
    elif 'p' in h5:
        frames,nx,ny,nz=h5['p'].shape
    else:
        raise ValueError("H5 file has neither data nor pressure")
    LOG.debug("Grid size (nx,ny,nz): (%d,%d,%d)",nx,ny,nz)
    LOG.debug("Frames: %d",frames)
    x=h5['x'].value
    z=h5['z'].value
    LOG.debug("Size (x,z): (%d,%d)",len(x),len(z))
    if not px:
        px=nx
    if not pz:
        pz=int(nx*Lz/Lx)
    LOG.debug("Pixel Size (px,pz): (%f,%f)",px,pz)
    px=int(px)
    pz=int(pz)
    LOG.debug('Interpolated grid size size %dx%d',px,pz)
    dx=Lx/px
    dz=Lz/pz
    x_new=(np.arange(px)+.5)*dx
    z_new=(np.arange(pz)+.5)*dz
    Z_new,X_new=np.meshgrid(z_new,x_new)

    #  Create the image group if it doesn't exist
    if not 'image' in h5:
        img=h5.create_group('image')
        for dset_name in ['data','p','v','vx','vy','vz']:
            if dset_name in h5: img.create_dataset(dset_name, (0,px,pz),compression='gzip', compression_opts=4,maxshape=(None,px,pz))
        # Create the x and z datasets
        img.create_dataset('x', data=x_new,compression='gzip', compression_opts=4)
        img.create_dataset('z', data=z_new,compression='gzip', compression_opts=4)
        # link the frames, and y points
        img['y']=h5['y']
        img['frame']=h5['frame']
    else:
        img=h5['image']
    
    for dset_name in ['data','p','v','vx','vy','vz']:
        if dset_name in h5:
            data_set=h5[dset_name]
            img_set=img[dset_name]
            data_frames=data_set.shape[0]
            img_frames=img_set.shape[0]
            LOG.debug("Interoplating %s from %d to %d",dset_name,img_frames,data_frames)
            img_set.attrs['min']=data_set.attrs['min']
            img_set.attrs['max']=data_set.attrs['max']
            LOG.debug("Resizing image[%s] to %s frames",dset_name,str( (data_frames,px,pz) ))
            img_set.resize((data_frames,px,pz))
            start_time=time.time()
            for n in xrange(img_frames,data_frames):
                data=np.squeeze(data_set[n])
                #Interpolate using basemap
                #img_set[n]=interp(data,z,x,Z_new,X_new)

                f=interpolate.interp2d(z,x,data)
                img_set[n]=f(z_new,x_new)

                #f=interpolate.RectBivariateSpline(x,z,data)
                #img_set[n]=f(x_new,z_new)
                delta_t=time.time()-start_time
                fps=(n+1-img_frames)/delta_t
                eta=(data_frames-n)/fps
                eta_hour=int(eta/3600)
                eta_min=int((eta-eta_hour*3600)/60)
                eta_sec=int(eta-eta_hour*3600-eta_min*60)
                LOG.debug("frame %d, fps %0.1f, eta %02d:%02d:%02d",n,fps,eta_hour,eta_min,eta_sec)

class Grid(object):
    def __init__(self,N,L):
        self.L=np.array(L)
        self.N=np.array(N)
        # Calculate in the 'b' (ying/yang) direction
        self.b=np.linspace(0,1,N[3])
        # Calculate in the X direction
        self.xg=np.linspace(0,L[0],N[0]*2+1)
        self.x_center=self.xg[1::2]
        self.x_face=self.xg[::2]
        # Calculate in the Y direction
        self.yg=np.linspace(0,L[1],N[1]*2+1)
        self.y_center=self.yg[1::2]
        self.y_face=self.yg[::2]
        self.cells=reduce(lambda x,y: max(1,x)*max(1,y), self.N-1)

class RegularGrid(Grid):
    def __init__(self,N,L):
        super(Grid, self).__init__(N,L)
        self.delta=np.array([ 0 if n==0 else l/n for l,n in zip(self.L,self.N-1)  ])
        # Calculate in the Z direction
        self.zg=np.linspace(0,L[1],N[2]*2+1)
        self.z_center=self.zg[1::2]
        self.z_face=self.zg[::2]

class ExpGrid(Grid):
    """
    Each layer is a tuple:
    (normalized depth, dresl, wresl)
    Normalized depth is layer depth/domain deption [0:1]
    dresl is the approzimate refinement
    wresl is the width
    """
    def __init__(self,N,L,layers=[],dresl_topbot=1.0):
        super(ExpGrid, self).__init__(N,L)
        # Calculate in the Z
        g_of_z=lambda z: g_of_z_exp(z,layers,dresl_topbot)
        iz2=np.arange(0.,N[2]*2+1)
        g=iz2/(N[2]*2)
        self.zg=L[2]*z_of_gnorm(g,g_of_z)
        self.z_center=self.zg[1::2]
        self.z_face=self.zg[::2]



