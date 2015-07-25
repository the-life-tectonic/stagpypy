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
#       this fearture is deprecated and now only the post 12feb2013 tracers are used
#
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
        par['geometry']['D_dimensional']=self.L[1]

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

    def calc_temp(self):
        # Set the background viscosity
        temp=np.ones(self.N)*self.T_mantle+(np.random.rand(*self.N)*self.amp_T-self.amp_T/2)
        LOG.debug('temp.shape=%s'%str(temp.shape))
        # The progress bar
        total=self.N[0]*self.N[1]
        i=0
        pb=ui.ProgressBar(total=total-1)
        for iz,z in enumerate(self.grid.z_center):
            for ix,x in enumerate(self.grid.x_center):
                pb.progress(i)
                i=i+1
                depth=self.grid.L[1]-z-self.air_layer
                T=None
                LOG.debug('temp[%0.2f,%0.2f,%0.2f]'%(x,z,depth))
                LOG.debug('temp[%3d,%3d]'%(ix,iz))
                for o in self.objects:
                    T=o.T(x,z,depth)
                    if T!=None:
                        try:
                            temp[ix,iz]=T
                        except:
                            LOG.error('temp[%0.2f,%0.2f,%0.2f]'%(x,z,depth))
                            LOG.error('temp[%3d,%3d]'%(ix,iz))
                            LOG.error('T=%f'%T)
                            raise
                        break
        return temp

    def calc_tracers(self,tracers):
        x=np.random.random(tracers)*self.L[0]
        z=np.random.random(tracers)*self.L[1]
        depth=self.grid.L[1]-z-self.air_layer
        tracer_type=np.zeros(tracers)
        if self.water:
            water=np.zeros(tracers)
        # The progress bar
        i=0
        pb=ui.ProgressBar(total=tracers-1)
        for i in xrange(tracers):
            for o in self.objects:
                tracer=o.tracer(x[i],z[i],depth[i])
                if tracer!=None:
                    break
            if tracer==None: 
                tracer=(tt_solid_harzburgite,0.0) if depth[i]>self.crust_depth else (tt_solid_basalt,1.0)
            assert(tracer[0] in tracer_types)
            tracer_type[i]=tracer[0]
            if self.water:
                water[i]=tracer[0]
            pb.progress(i)
        if self.water:
            return x,z,tracer_type,water
        else:
            return x,z,tracer_type

    def write_temp(self,out_dir):
        temp=self.calc_temp()
        filename=os.path.join(out_dir,'init.h5')
        h5file=h5py.File(filename)
        h5file.create_dataset('temp', data=temp)
        h5file.close()
        return filename

    def write_tracers(self,out_dir,tracers,tracers_per_cell=True):
        try:
            if tracers_per_cell:
                tracers=tracers*self.grid.cells
            tracers=self.calc_tracers(tracers)
            filename=os.path.join(out_dir,'init.h5')
            h5file=h5py.File(filename)
            h5file.create_dataset('tracers', data=tracers,compression='gzip')
            h5file.close()
        except NameError:
            sys.stderr.write('NameError encountered, perhaps tracer vintage was not set\n')
            sys.stderr.write('Make sure you set tracer vintage using selectTracers(when)\n')
            traceback.print_exc()
            sys.exit()
        return filename

class CartesianScene(Scene):
    def update_par(self,par):
        super(CartesianScene,self).update_par(par)
        par['geometry']['aspect_ratio(1)']=self.L[0]/self.L[1]
        par['geometry']['aspect_ratio(2)']=0
        par['geometry']['nxtot']=self.N[0]
        par['geometry']['nytot']=1
        par['geometry']['nztot']=self.N[1]
        par['geometry']['shape']='cartesian'

    def calc_temp(self):
        temp=super(CartesianScene,self).calc_temp()
        temp=temp.reshape(self.N[0],1,self.N[1],1)
        LOG.debug('temp.shape=%s'%str(temp.shape))
        return temp

    def calc_tracers(self,tracers):
        if self.water:
            x,z,tracer_type,water=super(CartesianScene,self).calc_tracers(tracers)
            return np.vstack((x,np.zeros(tracers),z,tracer_type,water)).T
        else:
            x,z,tracer_type=super(CartesianScene,self).calc_tracers(tracers)
            return np.vstack((x,np.zeros(tracers),z,tracer_type)).T

class AnnulusScene(Scene):
    def __init__(self,grid,alpha=3e-5,g=9.81,eta0=1e21,rho=3300.0,Cp=1200.0,k=3.0,kappa=1e-6,
                 T_surface=300.0, T_mantle=1600.0, eta_air=1e18, air_layer=0.0,crust_depth=0.0,amp_T=0,aspect_ratio=6.3):
        super(AnnulusScene, self).__init__(grid,alpha=alpha,g=g,eta0=eta0,rho=rho,Cp=Cp,k=k,kappa=kappa,
                                      T_surface=T_surface, T_mantle=T_mantle, eta_air=eta_air, 
                                      air_layer=air_layer,crust_depth=crust_depth,amp_T=amp_T)
        self.aspect_ratio=self.aspect_ratio
    def update_par(self,par):
        super(AnnulusScene,self).update_par(par)
        par['geometry']['aspect_ratio(1)']=0
        par['geometry']['aspect_ratio(2)']=self.aspect_ratio
        par['geometry']['nxtot']=1
        par['geometry']['nytot']=self.N[0]
        par['geometry']['nztot']=self.N[1]
        par['geometry']['shape']='cartesian'

    def calc_temp(self):
        temp=super(AnnulusScene,self).calc_temp()
        temp=temp.reshape(1,self.N[0],self.N[1],1)
        LOG.debug('temp.shape=%s'%str(temp.shape))
        return temp

    def calc_tracers(self,tracers):
        if self.water:
            x,z,tracer_type,water=super(AnnulusScene,self).calc_tracers(tracers)
            return np.vstack((np.zeros(tracers),x,z,tracer_type,water)).T
        else:
            x,z,tracer_type=super(AnnulusScene,self).calc_tracers(tracers)
            return np.vstack((np.zeros(tracers),x,z,tracer_type)).T

class SceneObject(object):
    def set_scene(self,scene):
        pass;

    def T(self,x,z,d):
        return None

    def tracer(self,x,z,d):
        return None

class Sphere(SceneObject):
    """
    Creates a 2 or 3D sphere of a given temperature
    """
    def __init__(self,x,z,r,T):
        self.x=x
        self.z=z
        self.temp=T

    def T(self,x,z,d):
        """Returns the temperature at x,y,z"""
        return self.temp if (x-self.x)**2++(z-self.z)**2 < self.r**2 else None

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

    def in_plate(self,x,d):
        result=False
        if d>=0 and d<=self.plate.r and ( (x-self.plate.trench)**2+(self.plate.r-d)**2 > (self.plate.r+self.gap)**2 ):
            if self.plate.trench>self.plate.ridge:
                result = x>self.plate.trench and x<self.plate.trench+self.length  
            else:
                result = x<self.plate.trench and x>self.plate.trench-self.length 
        return result

    def T(self,x,z,d):
        """Returns the temperature at x,y,z"""
        if self.in_plate(x,d):
            return T_hs(self.plate.scene.T_surface,self.plate.scene.T_mantle,d,self.age,self.plate.scene.kappa)
        return None

    def tracer(self,x,z,d):
        if d<self.crust_depth and self.in_plate(x,d):
            return (tt_solid_basalt,1.0)
        return None

class Plate(SceneObject):
    def __init__(self,ridge,trench,r,theta,v,scene,max_age=sys.maxint,crust=-1):
        # Calculate the orientation 
        # +1: left to right
        # -1: right to left
        self.ridge=float(ridge)
        self.trench=float(trench)
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

    def T(self,x,z,d):
        """Returns the temperature at x,y,z"""
        if d<0: return None # can't be in the air
        if d>self.r: return None # ignore deep stuff

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

    def tracer(self,x,z,d):
        if d<0 or d>self.r : return None # quick check

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
    def __init__(self,temp,thick,A,x,sigma_x,):
        self.thick=thick
        self.x=x
        self.A=A
        self.sigma_x=sigma_x
        self.temp=temp

    def set_scene(self,scene):
        self.scene=scene
    
    def tracer(self,x,z,d):
        return None

    def T(self,x,z,d):
        if z<self.thick+self.A: # only calculate if we are below the maximum amplitude
            surface=self.A*np.exp(-(x-self.x)**2/(2*self.sigma_x**2))+ self.thick;
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

    def T(self,x,z,d):
        return self._T if d<=0 else None

    def tracer(self,x,z,d):
        return (tt_air,0.0) if d<=0 else None


