# vim: set fileencoding=utf-8
import glob
import logging 
import numpy as np
import os
import re
import sys
import struct
import h5py

LOG=logging.getLogger(__name__)
LOG_RN=logging.getLogger(__name__+'.read_native')
LOG_PAR=logging.getLogger(__name__+'.read_par')
LOG_TRA=logging.getLogger(__name__+'.read_tracer')
LOG_RN.setLevel(logging.WARNING)
LOG_PAR.setLevel(logging.WARNING)
LOG_TRA.setLevel(logging.WARNING)

class Suite(object):
	def __init__(self,suite_dir):
		self.dir=suite_dir
		self.name=os.path.basename(self.dir)
		self.description=''
		LOG.debug('Suite dir: %s',self.dir)
		if os.path.exists(os.path.join(self.dir,'description')):
			f=open(os.path.join(self.dir,'description'))
			self.description=''.join(f.readlines()).strip()
			f.close()
		# build the list of models
		self.model_map={}
		model_list=[ d for d in os.listdir(self.dir) if os.path.exists(os.path.join(self.dir,d,'par')) ]
		LOG.debug('Model list: %s',model_list)
		self.fields=None
		for model_dir in model_list:
			try:
				model_path=os.path.join(self.dir,model_dir)
				LOG.debug('Reading model from %s',model_path)
				m=Model(model_path)
				self.model_map[model_dir]=m
				if self.fields==None:
					self.fields=set(m.fields)
				else:
					self.fields=self.fields&set(m.fields)
			except:
				LOG.error('Unable to process model %s',model_dir,exc_info=True)
		self.models=[self.model_map[model_name] for model_name in sorted(self.model_map.keys())]
		self.max_last_timestep=max([m.last_timestep for m in self.models])
		self.min_last_timestep=min([m.last_timestep for m in self.models])
		self.fields=sorted(self.fields)
		self.par_diff=par_diff([ m.par for m in self.models ])
		
class Model(object):
	def __init__(self,model_dir):
		self.dir=model_dir
		self.name=os.path.basename(self.dir)
		self.description=''
		if os.path.exists(os.path.join(self.dir,'description')):
			f=open(os.path.join(self.dir,'description'))
			self.description=''.join(f.readlines()).strip()
			f.close()
		# The par file
		par_file=os.path.join(model_dir,'par')
		LOG.debug('Reading parameter file %s',par_file)
		self.par=Par(par_file)
		# Grid members 
		self.nx=self.par['geometry']['nxtot']
		self.ny=self.par['geometry']['nytot']
		self.nz=self.par['geometry']['nztot']
		self.grid_size=( self.par['geometry']['nxtot'], self.par['geometry']['nytot'], self.par['geometry']['nztot'] )

		self.nwrite=self.par['timein']['nwrite']

		# output directory
		self.output_file_stem=self.par['ioin']['output_file_stem']
		ndx = self.output_file_stem.rfind('/')+1
		if ndx!=0:
			out_dir=self.output_file_stem[:ndx]
			prefix=self.output_file_stem[ndx:]
		else:
			out_dir=''
			prefix=self.output_file_stem
		self.output_file_stem_path=os.path.join(model_dir,self.output_file_stem)
		self.output_file_dir=os.path.join(model_dir,out_dir)
		self.output_file_prefix=prefix

		# image directory
		self.plot_file_stem=self.par['plot']['plot_file_stem']
		ndx = self.plot_file_stem.rfind('/')+1
		if ndx!=0:
			plot_dir=self.plot_file_stem[:ndx]
			prefix=self.plot_file_stem[ndx:]
		else:
			plot_dir=''
			prefix=self.plot_file_stem
		self.plot_file_stem_path=os.path.join(model_dir,self.plot_file_stem)
		self.plot_file_dir=os.path.join(model_dir,plot_dir)
		self.plot_file_prefix=prefix

		# The timestep file
		self.timestep_filename=os.path.join(model_dir,self.output_file_stem+'_time.dat')
		self.timesteps=Timesteps(self.timestep_filename)
		self.last_timestep=int(self.timesteps.istep[-1])
		self.total_timesteps=self.last_timestep+1
		self.frames=self.last_timestep/self.nwrite+1

		# The available fields
		self.fields=sorted(set([ f[len(self.output_file_stem_path)+1:-5] for f in glob.glob(self.output_file_stem_path+'_*[0-9][0-9][0-9][0-9][0-9]') ]))


def model_to_h5(model,dest,fields):
	# Calculate the number of timesteps
	LOG.debug('The model run completed %d timesteps',model.total_timesteps)
	LOG.debug('There should be %d frames available',model.frames)
	result=[]
	for field in fields:
		LOG.debug('Processing field %s',field.name)
		h5_filename=os.path.join(dest,get_h5_filename(model,field))
		frame_count=field_to_h5( h5_filename, os.path.join(model.dir,model.output_file_stem), field, model.grid_size, model.frames)
		result.append({'field': field.name, 'filename': h5_filename, 'frames': frame_count})
	return result

def get_h5_filename(model,field):
	return get_field_filename(model,field)+'.h5'

def get_field_filename(model,field,frame=None):
	return model.output_file_prefix+'_'+field.prefix+('%05d'%frame if frame!=None else '')

def get_field_path(model,field,frame):
	return os.path.join(model.dir,model.par.output_file_stem+'_'+field.prefix+('%05d'%frame))

def get_field_pattern(model,field):
	return model.output_file_prefix+'_'+field.prefix+'%05d'

def field_to_h5(h5_filename, output_file_stem, field, shape, frames, overwrite=False):
	# Check if the 
	if os.path.exists(h5_filename) and overwrite:
		LOG.debug('Removing file %s to overwrite',h5_filename)
		os.remove(h5_filename)
	LOG.debug('Opening file %s %s','existing' if os.path.exists(h5_filename) else 'new',h5_filename)
	h5_file=h5py.File(h5_filename,'a')
	LOG.debug('H5 itmes: %s',str(h5_file.items()))
	frame_start=0

	# Existing file
	if 'frame' in h5_file:
		frame_start=len(h5_file['frame'])
		LOG.debug('There are %d existing frames',frame_start)
		if frame_start<frames:
			LOG.debug('Processing an additional %d frames',frames-frame_start);
			frameDSet=h5_file['frame']
			frameDSet.resize((frames,2))
			dataDSet=h5_file['data']
			dataDSet.resize((frames,)+shape)
			fmax=dataDSet.attrs['max']
			fmin=dataDSet.attrs['min']
			xDSet=h5_file['x']
			yDSet=h5_file['y']
			zDSet=h5_file['z']
	else: # a new file
		LOG.debug("'frame' dataset not found, assuming a new file")
		# The file attribues
		h5_file.attrs['field']=field.name
		# The data dataset
		dataDSet=h5_file.create_dataset('data', (frames,)+shape,compression='gzip', compression_opts=4,maxshape=(None,)+shape)
		dataDSet.attrs['prefix']=field.prefix
		dataDSet.attrs['scalar']=field.scalar
		dataDSet.attrs['fieldname']=field.name
		# The frame dataset
		frameDSet=h5_file.create_dataset('frame', (frames,2) ,compression='gzip', compression_opts=4,maxshape=(None,2))
		# The x,y,z grid centers
		xDSet=h5_file.create_dataset('x', (shape[0],),compression='gzip',compression_opts=4)
		yDSet=h5_file.create_dataset('y', (shape[1],),compression='gzip',compression_opts=4)
		zDSet=h5_file.create_dataset('z', (shape[2],),compression='gzip',compression_opts=4)
		fmax=sys.float_info.min
		fmin=sys.float_info.max

	LOG.debug('Starting with frame %d to frame %d',frame_start,frames-1)

	frame_pattern=output_file_stem+'_'+field.prefix+'%05d'

	if frame_start<frames:
		for frame in range(frame_start,frames):
			file=frame_pattern%frame
			LOG.debug('Reading native file %s',file)
			d,step,time,x,y,z=read_native(file,field.scalar)
			LOG.debug("Raw data has the shape: %s",str(d.shape))
			fmax=max(fmax,d.max())
			fmin=min(fmin,d.min())
			LOG.debug('(min,max)=(%f,%f)',fmin,fmax)
			# copy the data
			dataDSet[frame]=d
			frameDSet[frame,0]=step
			frameDSet[frame,1]=time
		dataDSet.attrs['max']=fmax
		dataDSet.attrs['min']=fmin
		xDSet[:]=x[:]
		yDSet[:]=y[:]
		zDSet[:]=z[:]
	h5_file.close()
	return h5_filename,frames-frame_start

def get_filename(prefix,field,frame):
	if type(frame) is int:
		return prefix+field[0]+'%05d'%frame
	else:
		return prefix+field+frame

def frame_to_h5(directory,prefix,field,frame,out=None,overwrite=True):
	return file_to_h5(get_filename(prefix,field,frame),field,out,overwrite)


def file_to_h5(file,field,out=None,overwrite=True):
	if out==None: out=os.path.dirname(file)
	base=os.path.basename(file)
	h5filename=os.path.join(out,base+'.h5')
	if not os.path.exists(h5filename) or overwrite:
		suffix,scalar,fieldname=field
		try:
			d,step,time,x,y,z=read_native(file,scalar)
			fmax=d.max()
			fmin=d.min()
			d=d.squeeze()
			h5file=h5py.File(h5filename,'w')
			h5dset=h5file.create_dataset(fieldname, data=d,compression='gzip', compression_opts=4)
			h5dset=h5file.create_dataset('x', data=x)
			h5dset=h5file.create_dataset('y', data=y)
			h5dset=h5file.create_dataset('z', data=z)
			h5dset.attrs['max']=fmax
			h5dset.attrs['min']=fmin
			h5dset.attrs['step']=step
			h5dset.attrs['time']=time
		finally:
			h5file.close()
	return h5filename


def read_native_frame(directory,prefix,frame,field):
	suffix=field[0]
	scalar=field[1]
	filename=os.path.join(directory,prefix+suffix+'%05d'%frame)
	read_native(filename,scalar)

def read_native(filename,scalar):
	LOG.debug("Reading %s data from %s",'scalar' if scalar else 'vector',filename)
	if not os.path.exists(filename):
		raise IOError('File not found: %s',filename)
	try:
		f=open(filename,'rb')
		ver=read_int32(f)
		LOG_RN.debug('Version: %d',ver)
		nval = 1 if scalar else 4
		LOG_RN.debug('Each value has %d component(s)',nval)
		if (ver<100 and nval>1) or (ver>300 and nval==1):
			raise IOError('Incorrect number of components for field: %d, scalar=%s'%(nval,scalar))
		ver=ver%100
		# extra ghost point in the x and y directions
		xyp=1 if ver>=9 and nval==4 else 0

		nxtot,nytot,nztot,nblocks=read_int32(f,4)
		LOG_RN.debug('[nxtot,nytot,nztot,nblocks] = [%d,%d,%d,%d]',nxtot,nytot,nztot,nblocks)
		aspect=read_float32(f,2)
		LOG_RN.debug('aspect = %s'%str(aspect))
		nnx,nny,nnz,nnb=read_int32(f,4)
		LOG_RN.debug('[nnx,nny,nnz,nnb] = [%d,%d,%d,%d]',nnx,nny,nnz,nnb)

		nz2 = nztot*2 + 1 # zg is the center and bottom of the cells + the top of the first cell
		zg = read_float32(f,nz2) # z-coordinates
		LOG_RN.debug('len(zg) = %d\n',len(zg))
		LOG_RN.debug('zg=%s',str(zg))

		# compute nx, ny, nz and nb PER CPU
		nx = nxtot/nnx
		ny = nytot/nny
		nz = nztot/nnz
		nb = nblocks/nnb
		npi = (nx+xyp)*(ny+xyp)*nz*nb*nval # the number of values per 'read' block
		LOG_RN.debug('[nx,ny,nz,nb] = [%d,%d,%d,%d]',nx,ny,nz,nb)
		LOG_RN.debug('npi = %d',npi)
#		nxr=np.arange(nx)
#		nyr=np.arange(ny)
#		nzr=np.arange(nz)
#		nbr=np.arange(nb)
#		LOG_RN.debug("nxr=%s",nxr)
#		LOG_RN.debug("nyr=%s",nyr)
#		LOG_RN.debug("nzr=%s",nzr)
#		LOG_RN.debug("nbr=%s",nbr)

		rcmb = read_float32(f)
		istep = read_int32(f)
		time = read_float32(f)
		erupta_total = read_float32(f)
		botT_val = read_float32(f)
		LOG_RN.debug('rcmb %f',rcmb)
		LOG_RN.debug('istep %d',istep)
		LOG_RN.debug('time %f',time)
		LOG_RN.debug('erupta_total %f',erupta_total)
		LOG_RN.debug('botT_val %f',botT_val)

		x = read_float32(f,nxtot) # x-coordinates
		if nxtot==1: x=[x]
		y = read_float32(f,nytot)	# y-coordinates
		if nytot==1: y=[y]
		z = read_float32(f,nztot) # z-coordinates
		if nztot==1: z=[z]
		LOG_RN.debug('x=%s',str(x))
		LOG_RN.debug('y=%s',str(y))
		LOG_RN.debug('z=%s',str(z))

		# read the parallel blocks
		if scalar:
			DATA_3D = np.zeros((nxtot,nytot,nztot));
		else:
			scale_fac= read_float32(f)             # scale factor
			LOG_RN.debug('scale factor=%f',scale_fac)
			VX_3D = np.zeros((nxtot,nytot,nztot));   #   Vx
			VY_3D = np.zeros((nxtot,nytot,nztot));   #   Vy
			VZ_3D = np.zeros((nxtot,nytot,nztot));   #   Vz
			P_3D  = np.zeros((nxtot,nytot,nztot));   #   Pressure

		# loop over parallel subdomains
		subdomain=0
		for ibc in range(nnb):
			for izc in range(nnz):
				for iyc in range(nny):
					for ixc in range(nnx):
						LOG_RN.debug('--------------------------------------')
						LOG_RN.debug('Reading parallel subdomin %d',subdomain)
						subdomain+=1
						LOG_RN.debug('[ixc,iyc,izc,ibc]=[%d,%d,%d,%d]',ixc,iyc,izc,ibc)
						data_CPU = read_float32(f,npi) # read the data for this CPU
						LOG_RN.debug('len(data_CPU): %d',len(data_CPU))
						LOG_RN.debug('First 4 elements : %s',data_CPU[0:5])
						# Create a 3D matrix from these data

						if scalar:
							data_CPU_3D=data_CPU.reshape( (nx,ny,nz),order='F');                   
							LOG_RN.debug('shape data_CPU_3D: %s',str(data_CPU_3D.shape))
#                    		DATA_3D((ixc-1)*nx + (1:nx), (iyc-1)*ny + (1:ny), (izc-1)*nz + (1:nz),(ibc-1)*nb + (1:nb)) = data_CPU_3D; 
							LOG_RN.debug(' %d:%d, %d:%d, %d:%d ',ixc*nx,ixc*nx+nx, iyc*ny,iyc*ny+ny, izc*nz,izc*nz+nz )
							DATA_3D[ ixc*nx:ixc*nx+nx, iyc*ny:iyc*ny+ny, izc*nz:izc*nz+nz ] = data_CPU_3D
						else:
#							raise Exception('Vector data not supported yet')
							data_CPU_3D = scale_fac*data_CPU.reshape((nval,nx+xyp,ny+xyp,nz,nb),order='F');
							LOG_RN.debug('shape data_CPU_3D: %s'%str(data_CPU_3D.shape))
							# velocity-pressure data
#							LOG_RN.debug("ixc*nx=%d",ixc*nx)
#							LOG_RN.debug("(ixc+1)*nx=%d",(ixc+1)*nx)
#							LOG_RN.debug("iyc*ny=%d",iyc*ny)
#							LOG_RN.debug("(iyc+1)*ny=%d",(iyc+1)*ny)
#							LOG_RN.debug("izc*nz=%d",izc*nz)
#							LOG_RN.debug("(izc+1)*nz=%d",(izc+1)*nz)
#							LOG_RN.debug("VX_3D.shape=%s",VX_3D.shape)
#							LOG_RN.debug('(nx,ny,nz)=%s',(nx,ny,nz))
#							LOG_RN.debug('[ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz]=%s', ((ixc+1)*nx-ixc*nx, (iyc+1)*ny-iyc*ny, (izc+1)*nz-izc*nz))
							LOG_RN.debug('[ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz]=[%d:%d, %d:%d, %d:%d]', ixc*nx,(ixc+1)*nx, iyc*ny,(iyc+1)*ny, izc*nz,(izc+1)*nz)
#							LOG_RN.debug('ixc=%d',ixc)
#							LOG_RN.debug('iyc=%d',iyc)
#							LOG_RN.debug('izc=%d',izc)
							VX_3D[ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz] = data_CPU_3D[0,:nx,:ny,:,:].reshape((nx,ny,nz))
							VY_3D[ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz] = data_CPU_3D[1,:nx,:ny,:,:].reshape((nx,ny,nz))
							VZ_3D[ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz] = data_CPU_3D[2,:nx,:ny,:,:].reshape((nx,ny,nz))
							P_3D[ ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz] = data_CPU_3D[3,:nx,:ny,:,:].reshape((nx,ny,nz))
							LOG_RN.debug('VX_3D[%d,%d,%d]=%g',ixc*nx, iyc*ny, izc*nz, VX_3D[ixc*nx, iyc*ny, izc*nz])

#							VX_3D[ixc*nx+nxr, iyc*ny+nyr, izc*nz+nzr, ibc*nb+nbr] = data_CPU_3D[0,nxr,nyr,:,:].squeeze()
#							VY_3D[ixc*nx+nxr, iyc*ny+nyr, izc*nz+nzr, ibc*nb+nbr] = data_CPU_3D[1,nxr,nyr,:,:].squeeze()
#							VZ_3D[ixc*nx+nxr, iyc*ny+nyr, izc*nz+nzr, ibc*nb+nbr] = data_CPU_3D[2,nxr,nyr,:,:].squeeze()
#							P_3D[ ixc*nx+nxr, iyc*ny+nyr, izc*nz+nzr, ibc*nb+nbr] = data_CPU_3D[3,nxr,nyr,:,:].squeeze()
							DATA_3D=(VX_3D,VY_3D,VZ_3D,P_3D)
	finally:
		f.close()
	return DATA_3D,istep,time,x,y,z

def read_tracers(filename,callback=None):
	tracers=None
	LOG_TRA.debug("\nOpening %s"%filename)
	f=open(filename);

	try:
		# Read the magic number
		magic=read_int32(f)
		LOG_TRA.debug('magic=%d'%magic)

		# The magic number contains information about the number of blocks
		nb_in=magic%100
		LOG_TRA.debug('nb_in (# of blocks)=%d',nb_in)

		# Read the aspect ratio
		asp_in=read_float32(f,2)
		LOG_TRA.debug('asp_in (aspect ratio)=%s',asp_in)

		istep_in=read_int32(f)
		LOG_TRA.debug('istep_in (timestep)=%f',istep_in)

		time_in=read_float32(f)
		LOG_TRA.debug('time_in=%g'%time_in)

		ntracervar_in=read_int32(f)
		LOG_TRA.debug('ntracervar_in (# of tracer var)=%d',ntracervar_in)

		ntrg=read_int32(f,nb_in) # number of input tracers for each block
		LOG_TRA.debug('ntrg (# of input tracers)=%d',ntrg)

		tracernorm=read_float32(f)
		LOG_TRA.debug('tracernorm=%f',tracernorm)

		if(magic>100):
			cart0_cyl1_sph2=read_int32(f)
			domain_type=['catrisian','cylindrical','spherical'][cart0_cyl1_sph2]
			LOG_TRA.debug('cart0_cyl1_sph2 (input domain type=%d [%s])',cart0_cyl1_sph2,domain_type)
			if cart0_cyl1_sph2>0:
				r_cmb=read_float32(f)
				LOG_TRA.debug('r_cmb=%f'%r_cmb)

			tracer_varname=[]

			for var_num in range(ntracervar_in):
				tracer_varname.append(read_string(f,16).strip())
				LOG_TRA.debug('tracer_varname[%d]="%s"'%(var_num,tracer_varname[var_num]))
		t=np.zeros(ntrg*ntracervar_in)
		t=t.reshape(ntrg,ntracervar_in)
		
		for ib in range(nb_in):
			for i in range(ntrg):
				t[i]=read_float32(f,ntracervar_in)
				if callback!=None: callback(100*float(i)/float(ntrg))
		tracers={}
		tracers['magic']=magic
		tracers['blocks']=nb_in
		tracers['aspect_ratio']=asp_in
		tracers['time']=time_in
		tracers['vars']=ntracervar_in
		tracers['count']=ntrg
		tracers['norm']=tracernorm
		try:
			tracers['domain_type']=cart0_cyl1_sph2
			tracers['domain_name']=domain_type
			tracers['r_cmb']=r_cmb
		except NameError:
			pass
		tracers['tracers']=t
	finally:
		f.close()
	return tracers


# 32bit read
def read_int32(f,n=-1):
	d=np.array(struct.unpack('i'*abs(n),f.read(abs(n)*4)))
	if n==-1:
		return d[0]
	else: 
		return d

# 32bit read
def read_float32(f,n=1):
	d=np.array(struct.unpack('f'*n,f.read(n*4)))
	if n==1:
		return d[0]
	else:
		return d

# character read
def read_string(f,n=1):
	return ''.join(struct.unpack('s'*16,f.read(16)))



#def read_timedat(file,has_composition=False,has_melting=False):
#	# See main.f90 in StagYY code
#	cols=['istep','time','F_top','F_bot','Tmin','Tmean','Tmax','Vmin','Vrms','Vmax','eta_min','eta_mean','eta_max','ra_eff','Nu_top','Nu_bot']
#	if has_composition:
#		cols=cols+['C_min','C_mean','C_max']
#	if has_melting:
#		cols=cols+['F_mean','F_max','erupt_rate','erupta','erupt_heatflux','entrainment','Cmass_error']
#	cols=cols+['H_int']
#	c_dict={}
#	for n,col in enumerate(cols):
#		c_dict[col]=n
#	data=np.loadtxt(file,skiprows=1)
#	return c_dict,data


#
# Système international d'unités
# SI units
#

si_prefixes = {
	'yocto': -24,
	'zepto': -21,
	'atto':  -18,
	'femto': -15,
	'pico':  -12,
	'nano':   -9,
	'micro':  -6,
	'milli':  -3,
	'centi':  -2,
	'deci':   -1,
	'deca':    1,
	'hecto':   2,
	'kilo':    3,
	'mega':    6,
	'giga':    9,
	'tera':   12,
	'peta':   15,
	'exa':    18,
	'zetta':  21,
	'yotta':  24
}

def si_format(v,prefix):
	e=si_prefixes[prefix]
	return '%se%d'%(v/10**e,e)

#
# par file io routines
#

class Par(dict):
	def __init__(self,par=None):
		self.comments={}
		self.sections=['switches','geometry','refstate','boundaries','t_init','timein','viscosity','iteration','multi','ioin','compin','melt','phase','continents','tracersin','plot']
#		self.formats=
#		{ 'viscosity': {'stressY_eta': 'mega', 'stress0_eta'}}

		if par==None:
			return
		if os.path.isdir(par):
			par=os.path.join(par,'par')
		LOG.debug('Reading par file %s',par)
		self.name=par
		self.error=False
		f=open(par)
		try:
			p=self # The par file dictionary
			c=self.comments
			for line in f:
				line=line.strip()
				if line=='&end' or len(line)==0:
					pass
				elif line.startswith('&') and not line=='%end':
					LOG_PAR.debug('Start section %s',line[1:])
					section={}
					comment_section={}
					p[line[1:]]=section
					c[line[1:]]=comment_section={}
				else:
					LOG_PAR.debug('Processing line  %s',line)
					comment=None
					ndx=line.find('=')
					if ndx!=-1:
						name=line[0:ndx].strip()
						value=line[(ndx+1):].strip()
						if not '!' in name:
							if value.startswith("'"):  # A String
								ndx = value.find("'",1)
								value=value[1:ndx]
							else:
								# Check for a string
								if value.startswith("'"):
									value=value[1,value.find("'",1)]
								else:
									# Trim comments
									if '!' in value:
										comment=value[value.find('!')+1:].strip()
										value=value[0:value.find('!')].strip()

									if value=='.true.':
										value=True
									elif value=='.false.':
										value=False
									elif '.' in value or 'e' in value:
										value=[ float(v) for v in re.split("\s+",value) ]
										if len(value)==1:
											value=value[0]
									else:
										value=[ int(v) for v in re.split("\s+",value) ]
										if len(value)==1:
											value=value[0]
							section[name]=value
							if comment!=None:
								comment_section[name]=comment
					else:
						LOG_PAR.debug('Skipping line %s',line)
		except:
			LOG.error('Unable to parse par file %s',par,exc_info=True)
			self.error=True
		f.close()

	def __getitem__(self,key):
		try:
			val=dict.__getitem__(self,key)
		except KeyError:
			val={}
			self[key]=val
		return val

	def __setitem__(self,key,val):
			dict.__setitem__(self,key,val)
			if key not in self.comments:
				self.comments[key]={}


	def to_python(self,par_obj_name,io_mod=None,out=sys.stdout,incl_comments=False):
		if io_mod==None:
			out.write('%s=Par()\n'%par_obj_name)
		else:
			out.write('%s=%s.Par()\n'%(par_obj_name,io_mod))
		for section_name in sorted(self.keys()):
			section=self[section_name]
			comments=self.comments[section_name]
			for key in sorted(section.keys()):
				value=repr(section[key])
				out.write("%s['%s']['%s']=%s"%(par_obj_name,section_name,key,value))
				if key in comments:
					out.write(' # %s\n'%comments[key])
					if incl_comments:
						out.write("%s.comments['%s']['%s']=%s\n"%(par_obj_name,section_name,key,repr(comments[key])))
				else:
					out.write('\n')

	def write(self,out=sys.stdout,incl_header=True):
		if incl_header:
			import getpass
			import time
			from socket import gethostname
			from . import VERSION
			out.write('\n!\n')
			out.write('! Created by %s@%s\n'%(getpass.getuser(),gethostname()))
			out.write('! %s\n'%time.strftime('%H:%M:%S %d-%m-%Y UTC',time.gmtime()))
			out.write('! %s version %s\n'%(__name__,VERSION))
			out.write('!\n\n')

		for section_name in self.sections:
			try:
				section=self[section_name]
				comments=self.comments[section_name]
			except KeyError:
				section={}
				comments={}
			out.write('&%s\n'%section_name)
			for key in sorted(section.keys()):
				value=section[key]
				if type(value)==list:
					value=' '.join([str(v) for v in value])
				elif type(value)==str:
					value="'"+value+"'"
				elif type(value)==bool:
					value='.'+str(value).lower()+'.'
				else:
					value=str(value)
				out.write('\t%s=%s'%(key,value))
				if key in comments:
					out.write(' ! %s\n'%comments[key])
				else:
					out.write('\n')
			out.write('&end\n\n')
			
def par_diff(pars):
#	from sets import Set
	if type(pars[0])==str:
		pars=[Par(f) for f in pars]
	differences=[]
	sections=set()
	# the pars is a list of par files
	# set.update take an iterator, the par file is an iterator for it's keys, i.e. the par file sections 
	map(sections.update,pars)
	LOG.debug('sections = %s',str(sections))
	for section in sorted(sections):
		keys=set()
		map(keys.update,[par[section].keys() for par in pars])
		for key in sorted(keys):
			values=[]
			for par in pars:
				try:
					values.append(par[section][key])
				except KeyError:
					values.append('_NA_')
			comp = [ v for v in values if v!='_NA_' ] # missing values don't count
			if comp[1:]!=comp[:-1]:
				differences.append( { 'section':section,'key':key,'values':values } )
	return differences	

#
# *_time.dat file
# StagYY Timmstep Data
#

class Timesteps(np.ndarray):
	def __new__(subtype, fname, dtype=float):
		# Read the data from the file
		d=np.loadtxt(fname,dtype=dtype,skiprows=1)
		obj = np.ndarray.__new__(subtype, d.shape, dtype=dtype, buffer=d, offset=0, strides=None,order=None)
		# Read the first line of the file
		f=open(fname)
		obj.headers=[h.lower() for h in f.readline().split()]
		f.close()
		# Create a col_name -< col_number map
		obj._cols={}
		obj._cols.update(map(None,obj.headers,range(len(obj.headers))))
		# set the new 'info' attribute to the value passed
		obj.fname = fname
		# Finally, we must return the newly created object:
		return obj

	def __dir__(self):
		d=dir(self.__class__)
		d=d+self.__dict__.keys()+self.headers
		return sorted(d)


	def __getattr__(self,name):
		if name in self._cols:
			if len(self.shape)==1:
				return self[self._cols[name]]
			else:
				return np.array(self[:,self._cols[name]])
		else:
			return object.__getattr(self,name)

	def __array_finalize__(self, obj):
		# ``self`` is a new object resulting from
		# ndarray.__new__(InfoArray, ...), therefore it only has
		# attributes that the ndarray.__new__ constructor gave it -
		# i.e. those of a standard ndarray.
		#
		# We could have got to the ndarray.__new__ call in 3 ways:
		# From an explicit constructor - e.g. InfoArray():
		#    obj is None
		#    (we're in the middle of the InfoArray.__new__
		#    constructor, and self.info will be set when we return to
		#    InfoArray.__new__)
		if obj is None: return
		# From view casting - e.g arr.view(InfoArray):
		#    obj is arr
		#    (type(obj) can be InfoArray)
		# From new-from-template - e.g infoarr[:3]
		#    type(obj) is InfoArray
		#
		# Note that it is here, rather than in the __new__ method,
		# that we set the default value for 'info', because this
		# method sees all creation of default objects - with the
		# InfoArray.__new__ constructor, but also with
		# arr.view(InfoArray).
		self.fname = getattr(obj, 'fname', None)
		self.headers = getattr(obj, 'headers', None)
		self._cols = getattr(obj, '_cols', None)
		# We do not need to return anything


