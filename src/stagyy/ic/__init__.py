import numpy as np
import h5py
import logging
import os
import sys
import traceback
from stagyy import ui
from stagyy.util import T_hs

LOG=logging.getLogger(__name__)

def log(level=logging.DEBUG):
    handler=logging.StreamHandler()
    handler.setLevel(level)
    LOG.addHandler(handler)
    LOG.setLevel(level)

#
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# ! WARNING    WARNING    WARNING   WARNING    WARNING    WARNING !
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# The version of StagYY received from 12 Feb 2013 makes changes to
# the numbering of the tracer types.  You must now activly make a call
# to define the tracer types.
#
# Currently defined vintages are:
#       pre12feb2013 - the tracer types used prior to the change in Stagyy
#       12feb2013 - the tracer types used in the version from 12 Feb 2013
#

def selectTracers(when):
    """Selects the the vinatge of tracer values to use.
       The version of StagYY received from 12 Feb 2013 makes changes to
       the numbering of the tracer types.  You must now activly make a call
       to define the tracer types.

       Currently defined vintages are:
           pre12feb2013 - the tracer types used prior to the change in Stagyy
           12feb2013 - the tracer types used in the version from 12 Feb 2013
"""
    global tt_harzburgite 
    global tt_solid_harzburgite 
    global tt_solid_basalt
    global tt_molten_harzburgite 
    global tt_molten_basalt
    global tt_newly_melted_basalt 
    global tt_wants_to_freeze_basalt
    global tt_air 
    global tt_prim 
    global tt_ccrust
    global tt_erupting_basalt
    global tt_intruding_basalt
    global tracer_types

    if when=='pre12feb2013':
        tt_harzburgite=0
        tt_solid_harzburgite=0
        tt_solid_basalt=1
        tt_molten_basalt=2
        tt_air=3
        tt_prim=4
        tt_newly_melted_basalt=5
        tracer_types=range(6)
    elif when=='12feb2013':
        tt_solid_harzburgite=0
        tt_solid_basalt=1
        tt_molten_harzburgite=2
        tt_molten_basalt=3
        tt_newly_melted_basalt=4
        tt_wants_to_freeze_basalt=5
        tt_air=6
        tt_prim=7
        tt_ccrust=8
        tt_erupting_basalt=9
        tt_intruding_basalt=10
        tracer_types=range(11)
    else:
        sys.exit("Unkown tracer vintage %s."%when);


class Scene(object):
    """
    The Scene object contains information about the domain of the problem.  The size (L) and grid (shape) must
    be provided in the constructor.  The physical properties of the background material properties can also be
    passed to the constructor or set on the object after instanation.  Default material properties
    are for a modern Earth:
    alpha=3e-5      : thermal expansion coefficient (K^-1) [expan_dimensional in par]
    g=9.81          : gravity (m s^-2)  [g_dimensional in par]
    eta0=1e21        : dynamic viscosity (Pa s) [eta0 in par]
    rho=3300.0      : densiry (kg m^-3) [dens_dimensional in par]
    Cp=1200.0       : spefific heat (J kg^-1 K^-1) [Cp_dimensional in par]
    k=3.0           : thermal conductivity (W m^-1 K^-1) [tcond_dimensional]
    kappa=1e-6      : diffusivity (m^2 s^-1) {k/rho Cp}
    T_surface=300.0 : Temperature at the surface (K)
    T_mantle=1600.0 : Background mantle temperature (K)
    Amp_T           : Amplitude of randomness to add to the Temp field (K)
    """
    def __init__(self,grid,alpha=3e-5,g=9.81,eta0=1e21,rho=3300.0,Cp=1200.0,k=3.0,kappa=1e-6,
                 T_surface=300.0, T_mantle=1600.0, eta_air=1e18, air_layer=0.0,crust_depth=0.0,amp_T=0):
        self.L=grid.L
        self.N=grid.N
        self.grid=grid
        self.cells=reduce(lambda x,y: max(1,x)*max(1,y), self.N-1)
        self.alpha=alpha
        self.g=g
        self.eta0=eta0
        self.eta_air=eta_air
        self.rho=rho
        self.Cp=Cp
        self.k=k
        self.kappa=kappa
        self.T_surface=T_surface
        self.T_mantle=T_mantle
        self.air_layer=air_layer
        self.crust_depth=crust_depth
        self.objects=[]
        self.amp_T=amp_T
        self.water=False;

    def add(self,o):
        if not isinstance(o,SceneObject):
            raise TypeError("SceneObject expected, got %s"%o.__class__)
        self.objects.append(o)
        o.set_scene(self)

    def update_par(self,par):
        par['geometry']['D_dimensional']=self.L[2]
        par['geometry']['aspect_ratio(1)']=self.L[0]/self.L[2]
        par['geometry']['aspect_ratio(2)']=self.L[1]/self.L[2]
        par['geometry']['nxtot']=self.N[0]
        par['geometry']['nytot']=self.N[1]
        par['geometry']['nztot']=self.N[2]

        par['refstate']['expan_dimensional']=self.alpha
        par['refstate']['g_dimensional']=self.g
        par['refstate']['dens_dimensional']=self.rho
        par['refstate']['Cp_dimensional']=self.Cp
        par['refstate']['tcond_dimensional']=self.k
        par['refstate']['deltaT_dimensional']=self.T_mantle-self.T_surface

        par['viscosity']['eta0']=self.eta0
        par['viscosity']['eta_air']=self.eta_air

        par['boundaries']['topT_val']=self.T_surface
        par['boundaries']['air_layer']=self.air_layer>0
        par['boundaries']['air_thickness']=self.air_layer

class SceneObject(object):
    def set_scene(self,scene):
        pass;

    def T(self,x,y,z,d):
        return None

    def tracer(self,x,y,z,d):
        return None

#class Crust(SceneObject):
#   def __init__(self,left,right,depth,front=0,back=0,tracer=tt_solid_basalt):
#       self.left=left
#       self.right=right
#       self.front=front
#       self.back=back
#       self.depth=depth
#       self.tra=tracer
#
#   def tracer(self,x,y,z,d):
#       if d>0 and d<=self.depth and x>=self.left and x<=self.right and y>=self.front and y<=self.back:
#           return self.tra
#       else:
#       return None

class Sphere(SceneObject):
    """
    Creates a 2 or 3D sphere of a given temperature
    """
    def __init__(self,x,y,z,r,T):
        self.x=x
        self.y=y
        self.z=z
        self.temp=T

    def T(self,x,y,z,d):
        """Returns the temperature at x,y,z"""
        return self.temp if (x-self.x)**2+(y-self.y)**2+(z-self.z)**2 < self.r**2 else None

    def tracer(self,x,y,z,d):
        return None

class UpperPlate(SceneObject):
    """ 
    Creates an upper plate that abuts the subducting plate.
    The overriding plate has a constant age profile
    """
    def __init__(self,plate,length,age,gap,crust_depth=-1):
        self.plate=plate
        self.length=length
        self.age=age
        self.trench=plate.trench
        self.end=plate.trench+length
        self.gap=gap
        self.crust_depth=crust_depth

    def set_scene(self,scene):
        self.scene=scene
        if self.crust_depth==-1:
            self.crust_depth=scene.crust_depth

    def in_plate(self,x,y,d):
        result=False
        if d>=0 and d<=self.plate.r and y>=self.plate.front and y<=self.plate.back and ( (x-self.plate.trench)**2+(self.plate.r-d)**2 > (self.plate.r+self.gap)**2 ):
            if self.plate.trench>self.plate.ridge:
                result = x>self.plate.trench and x<self.plate.trench+self.length  
            else:
                result = x<self.plate.trench and x>self.plate.trench-self.length 
        return result

    def T(self,x,y,z,d):
        """Returns the temperature at x,y,z"""
        if self.in_plate(x,y,d):
            return T_hs(self.plate.scene.T_surface,self.plate.scene.T_mantle,d,self.age,self.plate.scene.kappa)
        return None

    def tracer(self,x,y,z,d):
        if d<self.crust_depth and self.in_plate(x,y,d):
            return (tt_solid_basalt,1.0)
        return None

class Plate(SceneObject):
    def __init__(self,ridge,trench,r,theta,v,scene,front=0,back=0,max_age=sys.maxint,crust=-1):
        # Calculate the orientation 
        # +1: left to right
        # -1: right to left
        self.ridge=float(ridge)
        self.trench=float(trench)
        self.front=float(front)
        self.back=float(back)
        self.theta=float(theta)
        self.r=float(r)
        self.v=float(v)
        self.bend_crust_depth=0;
        self.gap_crust_depth=0;
        self.crust_depth=crust if crust>=0 else scene.crust_depth
        self.max_age=max_age;
        bend_horiz_length=self.r*np.sin(self.theta)
        if trench>ridge:
            self.in_plate = lambda x: x>ridge and x<=trench
            self.in_bend = lambda x: x>=trench and x<=trench+bend_horiz_length
            #self.in_bend = lambda x: x>=trench and x<=trench+self.r*np.sin(self.theta)
        else:
            self.in_plate = lambda x: x>=trench and x<ridge
            self.in_bend = lambda x: x>=trench-bend_horiz_length and x<=trench
            #self.in_bend = lambda x: x>=trench-self.r*np.sin(self.theta) and x<=trench
        self.scene=scene
        self.age=abs(ridge-trench)/v

    def set_scene(self,scene):
        self.scene=scene

    def T(self,x,y,z,d):
        """Returns the temperature at x,y,z"""
        if d<0: return None # can't be in the air
        if d>self.r: return None # ignore deep stuff
        if y<self.front or y>self.back: return None # inside the plate

        if self.in_plate(x):
            l=abs(x-self.ridge)
        elif  self.in_bend(x):
            if self.trench>self.ridge:
                x=x-self.trench
            else:
                x=self.trench-x
            y=self.r-d
            d=self.r-np.sqrt(x**2+y**2)
            if d<0:
                return None
            phi=np.arctan2(x,y)
            if phi>self.theta:
                return None
            l=abs(self.ridge-self.trench)+phi*self.r
        else:
            return None
        t=min(self.max_age,abs(l/self.v))
        return T_hs(self.scene.T_surface,self.scene.T_mantle,d,t,self.scene.kappa)

    def tracer(self,x,y,z,d):
        if d<0 or d>self.r or y<self.front or y>self.back : return None # quick check

        if self.in_plate(x):
            crust_depth=self.crust_depth
        elif self.in_bend(x):
            if self.trench>self.ridge:
                x=x-self.trench
            else:
                x=self.trench-x
            surface_d=d
            y=self.r-d
            d=self.r-np.sqrt(x**2+y**2)
            # if the location is above the bend
            if d<0:
                # if the location is more shallow than the crust
                if surface_d<self.gap_crust_depth:
                    return (tt_solid_basalt,1.0)
                else:
                    return None
            phi=np.arctan2(x,y)
            if phi>self.theta:
                return None
            crust_depth=self.crust_depth+self.bend_crust_depth
        else :
            return None
        
        if d<crust_depth:
            return (tt_solid_basalt,1.0)
        else:
            return None

class GaussianPlume(SceneObject):
    def __init__(self,temp,thick,A,x,sigma_x,y=0,sigma_y=1):
        self.thick=thick
        self.x=x
        self.y=y
        self.A=A
        self.sigma_x=sigma_x
        self.sigma_y=sigma_y
        self.temp=temp

    def set_scene(self,scene):
        self.scene=scene
    
    def tracer(self,x,y,z,d):
        return None

    def T(self,x,y,z,d):
        if z<self.thick+self.A: # only calculate if we are below the maximum amplitude
            surface=self.A*np.exp(-((x-self.x)**2/(2*self.sigma_x**2) + (y-self.y)**2/(2*self.sigma_y**2)) )+self.thick;
            if z<=surface:
                return self.temp
            else:
                return None
        return None


class Air(SceneObject):
    def __init__(self):
        pass

    def set_scene(self,scene):
        self.scene=scene
        self._T=scene.T_surface

    def T(self,x,y,z,d):
        return self._T if d<=0 else None

    def tracer(self,x,y,z,d):
        return (tt_air,0.0) if d<=0 else None

def calc_temp(scene):
    N=scene.N.copy()
    # if N[0]=1 the data is Y,Z but the objects expect X,Z so do some swapping
    if scene.N[0]==1:
        LOG.debug('Processing Y,Z data')
        N[0]=scene.N[1]
        N[1]=scene.N[0]
#        dy,dx,dz=scene.delta
    else:
        LOG.debug('Processing X,Z or X,Y,Z data')
#        dx,dy,dz=scene.delta

    # Set the background viscosity
    temp=np.ones(N)*scene.T_mantle+(np.random.rand(*N)*scene.amp_T-scene.amp_T/2)
    # The progress bar
    total=N[0]*N[1]*N[2]*N[3]
    i=0
    pb=ui.ProgressBar(total=total-1)
    for ib in scene.grid.b:
        for iz,z in enumerate(scene.grid.z_center):
            for iy,y in enumerate(scene.grid.y_center):
                for ix,x in enumerate(scene.grid.x_center):
                    pb.progress(i)
                    i=i+1
                    depth=scene.grid.L[2]-z-scene.air_layer
                    T=None
                    LOG.debug('T(%0.2f,%0.2f,%0.2f,%0.2f)'%(x,y,z,depth))
                    LOG.debug('T(%3d,%3d,%3d,%3d)'%(ix,iy,iz,ib))
                    for o in scene.objects:
                        T=o.T(x,y,z,depth)
                        if T!=None:
                            try:
                                temp[ix,iy,iz,ib]=T
                            except:
                                LOG.error('temp[%0.2f,%0.2f,%0.2f,%0.2f]'%(x,y,z,depth))
                                LOG.error('temp[%3d,%3d,%3d,%3d]'%(ix,iy,iz,ib))
                                LOG.error('temp.shape=%s'%str(temp.shape))
                                LOG.error('T=%f'%T)
                                raise
                            break
    if scene.N[0]==1:
        temp=np.swapaxes(temp,0,1)
    return temp

def calc_tracers(scene,tracers):
    N=scene.N
    L=scene.L
    tra=np.zeros((tracers,5 if not scene.water else 6))
    # Randomly select x,y,z locations
    tra[:,:3]=np.random.random((tracers,3))
    # depth is 1-z
    tra[:,-1]=1-tra[:,2]
    # dimensionalize
    tra[:,0]*=L[0]
    tra[:,1]*=L[1]
    tra[:,2]*=L[2]
    tra[:,-1]*=L[2]
    tra[:,-1]-=scene.air_layer

    # If nx=1, then swap x and y
    if N[0]==1:
        tmp=tra[:,0].copy()
        tra[:,0]=tra[:,1]
        tra[:,1]=tmp

    # The progress bar
    i=0
    pb=ui.ProgressBar(total=tracers-1)
    for t in tra[:]:
        for o in scene.objects:
            tracer=o.tracer(t[0],t[1],t[2],t[-1])
            if tracer!=None:
                break
        if tracer==None: 
            tracer=(tt_solid_harzburgite,0.0) if t[-1]>scene.crust_depth else (tt_solid_basalt,1.0)
        assert(tracer[0] in tracer_types)
        tra[i,3]=tracer[0]
        if scene.water:
            tra[i,4]=tracer[1]
        pb.progress(i)
        i+=1
    # Swap them back    
    if scene.N[0]==1:
        tmp=tra[:,0].copy()
        tra[:,0]=tra[:,1]
        tra[:,1]=tmp
    return tra[:,:-1]

def write_temp(out_dir,scene):
    temp=calc_temp(scene)
    filename=os.path.join(out_dir,'init.h5')
    h5file=h5py.File(filename)
    h5file.create_dataset('temp', data=temp)
    h5file.close()
    return filename

def write_tracers(out_dir,scene,tracers,tracers_per_cell=True):
    try:
        if tracers_per_cell:
            tracers=tracers*scene.cells
        tracers=calc_tracers(scene,tracers)
        filename=os.path.join(out_dir,'init.h5')
        h5file=h5py.File(filename)
        #h5file.create_dataset('tracers', data=tracers,compression='gzip')
        h5file.create_dataset('tracers', data=tracers,compression='gzip')
        h5file.close()
    except NameError:
        sys.stderr.write('NameError encountered, perhaps tracer vintage was not set\n')
        sys.stderr.write('Make sure you set tracer vintage using selectTracers(when)\n')
        traceback.print_exc()
        sys.exit()
    return filename
