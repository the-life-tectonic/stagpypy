import logging 
import numpy as np
import grid_refinement

LOG=logging.getLogger(__name__)

class Grid(object):
    def __init__(self,N,L):
        self.L=np.array(L)
        self.N=np.array(N)
        self.cells=N[0]*N[1]
        # Calculate in the X direction
        self.xg=np.linspace(0,L[0],N[0]*2+1)
        self.x_center=self.xg[1::2]
        self.x_face=self.xg[::2]

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
        g_of_z=lambda z: grid_refinement.g_of_z_exp(z,layers,dresl_topbot)
        iz2=np.arange(0.,N[1]*2+1)
        g=iz2/(N[1]*2)
        self.zg=L[1]*grid_refinement.z_of_gnorm(g,g_of_z)
        self.z_center=self.zg[1::2]
        self.z_face=self.zg[::2]



