import sys

class Field(object):
    def __init__(self,name,prefix,fields,par_flag=None):
        self.name=name
        self.prefix=prefix
        self.fields=fields
        self.par_flag=par_flag or prefix+'_write'
        self.scalar=len(fields)==1

    def index(self,v):
        return self.fields.index(v)

    def __str__(self):
        return "Field(%s)"%self.name

    def __repr__(self):
        return self.__str__()

AGE=Field('age','age',['age'])
AIR=Field('air','air',['air'])
COMPOSITION=Field('composition','c',['c'])
CRUSTAL_THICKNESS=Field('crust','cr',['cr'])
DENSITY=Field('density','rho',['rho'])
DEWATER=Field('dewater','dwtr',['dwtr'],'dewater_write')
GEOID=Field('geoid','g',['g'])
MELT_FRACTION=Field('melt fraction','f',['f'])
STRAIN_RATE=Field('strain rate','ed',['ed'])
STRESS=Field('stress','str',['str'],'stress_write')
STRESS_AXIS=Field('stress axis','sx',['sx','sy','sz','s2'],'stress_axis_write')
TEMP=Field('temp','t',['t'])
TOPOGRAPHY=Field('topography','cs',['cs'])
TOPO_SELF_GRAVITY=Field('self gravity','csg',['csg'])
VELOCITY_PRESSURE=Field('velocity and pressure','vp',['vx','vy','vz','p'])
VISCOSITY=Field('viscosity','eta',['eta'])
TRACERS=Field('tracers','tra',['tra'])

current_module = sys.modules[__name__]

ALL_FIELDS=[v for v in current_module.__dict__.values() if  isinstance(v,Field)]

by_prefix  =dict([(F.prefix,F) for F in ALL_FIELDS])
by_name    =dict([(F.name,F) for F in ALL_FIELDS])
by_par_flag=dict([(F.par_flag,F) for F in ALL_FIELDS])
by_field   =dict([(fld,F) for F in ALL_FIELDS for fld in F.fields])
fields=by_prefix

