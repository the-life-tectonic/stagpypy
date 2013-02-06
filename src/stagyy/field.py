class Field(object):
	def __init__(self,prefix,scalar,name):
		self.prefix=prefix
		self.scalar=scalar
		self.name=name


AIR=Field('air',True,'air')
AGE=Field('age',True,'age')
COMPOSITION=Field('c',True,'composition')
CRUSTAL_THICKNESS=Field('cr',True,'crust')
DENSITY=Field('rho',True,'density')
GEOID=Field('g',True,'geoid')
MELT_FRACTION=Field('f',True,'melt fraction')
STRAIN_RATE=Field('ed',True,'strain rate')
TEMP=Field('t',True,'temp')
TOPOGRAPHY=Field('cs',True,'topography')
TOPO_SELF_GRAVITY=Field('csg',True,'selfgravity')
VELOCITY_FIELD=Field('vp',False,'velocity')
VISCOSITY=Field('eta',True,'viscosity')

by_prefix={
AIR.prefix: AIR,
AGE.prefix: AGE,
COMPOSITION.prefix: COMPOSITION,
CRUSTAL_THICKNESS.prefix: CRUSTAL_THICKNESS,
DENSITY.prefix: DENSITY,
GEOID.prefix: GEOID,
MELT_FRACTION.prefix: MELT_FRACTION,
STRAIN_RATE.prefix: STRAIN_RATE,
TEMP.prefix: TEMP,
TOPOGRAPHY.prefix: TOPOGRAPHY,
TOPO_SELF_GRAVITY.prefix: TOPO_SELF_GRAVITY,
VELOCITY_FIELD.prefix: VELOCITY_FIELD,
VISCOSITY.prefix: VISCOSITY
}

fields=by_prefix

by_name={
AIR.name: AIR,
AGE.name: AGE,
COMPOSITION.name: COMPOSITION,
CRUSTAL_THICKNESS.name: CRUSTAL_THICKNESS,
DENSITY.name: DENSITY,
GEOID.name: GEOID,
MELT_FRACTION.name: MELT_FRACTION,
STRAIN_RATE.name: STRAIN_RATE,
TEMP.name: TEMP,
TOPOGRAPHY.name: TOPOGRAPHY,
TOPO_SELF_GRAVITY.name: TOPO_SELF_GRAVITY,
VELOCITY_FIELD.name: VELOCITY_FIELD,
VISCOSITY.name: VISCOSITY
}
