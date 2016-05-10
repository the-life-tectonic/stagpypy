# vim: set fileencoding=utf-8
import h5py
import itertools
import logging 
import numpy as np
import os
import re
import sys
import struct
from evtk.hl import pointsToVTK
from image import image2d,viz
import stagyy.signal as ssig
import field as stag_field

LOG=logging.getLogger(__name__)
LOG_RN=logging.getLogger(__name__+'.read_native')
LOG_PAR=logging.getLogger(__name__+'.read_par')
LOG_TRA=logging.getLogger(__name__+'.read_tracer')
LOG_RN.setLevel(logging.WARNING)
LOG_PAR.setLevel(logging.WARNING)
LOG_TRA.setLevel(logging.WARNING)


def tracers_to_vtu(model,dest, overwrite=False):
    from field import TRACERS
    # Calculate the number of timesteps
    LOG.debug('The model run completed %d timesteps',model.total_timesteps)
    LOG.debug('There should be %d frames available',model.frames)
    frames=model.frames
    #result=[]
    field=TRACERS
    LOG.debug('Processing field %s',field.name)
    for frame in range(frames):
        tracer_filename=os.path.join(model.dir,model.output_file_stem)+'_'+field.prefix+'%05d'%frame
        vtu_fileout=os.path.join(dest,field.prefix+'%05d'%frame)
        vtu_filename=vtu_fileout+'.vtu'
        if os.path.exists(vtu_filename):
            if overwrite:
                LOG.debug('Overwriting %s',vtu_filename)
            else:
                LOG.debug('Skipping existing file %s',vtu_filename)
                continue
        LOG.debug('Processing %s',vtu_filename)
        tra=read_tracers(tracer_filename)
        tracers=tra['tracers']
        data={tra['varnames'][n]:tracers[:,n] for n in range(3,tra['vars'])}
        LOG.debug("data: %s",data)
        pointsToVTK(vtu_fileout,tracers[:,0],tracers[:,1],tracers[:,2],data=data)

def render_field(data,img_dir,renderer_name,renderer_class=None,renderer_options={}):
    gallery=viz.Gallery()
    if renderer_class!=None:
        gallery.register_renderer(renderer_name,renderer_class,renderer_options)

    if ssig.WALLTIME:
        return

    h5_file=None
    try:
        for frame_num in range(data.shape[0]):
            if ssig.WALLTIME:
                break
            img_file=gallery.get_img(img_dir,renderer_name,frame_num,{})
            if img_file==None:
                LOG.debug('Rendering frame %d',frame_num)
                img_file=gallery.render_img(img_dir,renderer_name,frame_num,data[frame_num],{})
                LOG.debug('Rendered as %s',img_file)
            else:
                LOG.debug('Frame %d already exists as %s',frame_num,img_file)
    finally:
        if h5_file: h5_file.close()

def regularize_field(model,dest,f):
    """Interpolates a field that has grid refinement so that the grid is regular with a cell size of the smallest cell."""
    constant_spacing=model.par['geometry']['zspacing_mode']=="constant" or all([v==1 for k,v in model.par['geometry'].items() if k.startswith('dresl')])
    LOG.debug("Spacing is %s"%("constant" if constant_spacing else "refined"))
    if constant_spacing:
        LOG.error("Not implemented yet")
    else:
        if ssig.WALLTIME:
            return
        h5_filename=os.path.join(dest,f+'.h5')
        LOG.debug('H5 file: %s',h5_filename)
        img_filename=os.path.join(dest,f+'_image.h5')
        LOG.debug('Image file: %s',img_filename)
        h5_file=None
        h5_img=None
        try:
            h5_file=h5py.File(h5_filename,'a')
            LOG.debug('H5 itmes: %s',str(h5_file.items()))
            h5_img=h5py.File(img_filename,'a')
            LOG.debug('Image itmes: %s',str(h5_img.items()))
            # TODO: implement squaring 3D
            Lz=model.par['geometry']['D_dimensional']
            Lx=Lz*model.par['geometry']['aspect_ratio(1)']
            LOG.debug('Dimensional shape %dx%d',Lx,Lz)
            dL=np.diff(h5_file['z']).min()
            LOG.debug('Smallest cell size %f',dL)
            Pz=np.floor(Lz/dL)
            Px=np.floor(Lx/dL)
            LOG.debug('Interpolated grid size size (x,z) %dx%d',Px,Pz)
            image2d.square_pixels(h5_file,h5_img,Lx,Lz,Px,Pz)
        finally:
            if h5_file: h5_file.close()
            if h5_img: h5_img.close()

def field_to_h5(model,dest,f,overwrite=False):
    frames=model.frames
    shape=model.grid_size
    field=stag_field.by_field[f]
    frame_pattern=model.get_field_filename_pattern(field)
    h5_filename=os.path.join(dest,f+'.h5')
    if os.path.exists(h5_filename) and overwrite:
        LOG.debug('Removing file %s to overwrite',h5_filename)
        os.remove(h5_filename)
    LOG.debug('Opening file %s %s','existing' if os.path.exists(h5_filename) else 'new',h5_filename)
    h5_file=h5py.File(h5_filename,'a')
    LOG.debug('H5 itmes: %s',str(h5_file.items()))
    frame_start=0

    try:
        if not ssig.WALLTIME:
            if 'frame' in h5_file: # Existing file
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
                    zgDSet=h5_file['zg']
            else: # a new file
                LOG.debug("'frame' dataset not found, assuming a new file")
                # The file attribues
                h5_file.attrs['field_name']=field.name
                h5_file.attrs['field']=f
                # The data dataset
                dataDSet=h5_file.create_dataset('data', (frames,)+shape,compression='gzip', compression_opts=4,maxshape=(None,)+shape)
                fmax=sys.float_info.min
                fmin=sys.float_info.max
                # The frame dataset
                frameDSet=h5_file.create_dataset('frame', (frames,2) ,compression='gzip', compression_opts=4,maxshape=(None,2))
                # The x,y,z grid centers
                xDSet=h5_file.create_dataset('x', (shape[0],),compression='gzip',compression_opts=4)
                yDSet=h5_file.create_dataset('y', (shape[1],),compression='gzip',compression_opts=4)
                zDSet=h5_file.create_dataset('z', (shape[2],),compression='gzip',compression_opts=4)
                zgDSet=h5_file.create_dataset('zg',(2*shape[2]+1,),compression='gzip',compression_opts=4)

            LOG.debug('Starting with frame %d to frame %d',frame_start,frames-1)
            ndx=field.index(f)
            LOG.debug("Field %s is at index %d of raw data",f,ndx)

            if 'max' in dataDSet.attrs:    
                fmax=dataDSet.attrs['max']
            if 'min' in dataDSet.attrs:    
                fmin=dataDSet.attrs['min']
            dataDSet.attrs['min']=fmin
            for frame in range(frame_start,frames):
                file=frame_pattern%frame
                LOG.debug('Reading native file %s',file)
                if not os.path.exists(file):
                    LOG.warning('Native file "%s" does not exist, though %d frames expected starting at %d, skipping' % (file,frames,frame_start))
                    break;
                try:
                    d,step,time,x,y,z,zg=read_native(file,field.scalar)
                    d=d[ndx]
                    LOG.debug("Raw data has the shape: %s",str(d.shape))
                    fmax=max(fmax,d.max())
                    fmin=min(fmin,d.min())
                    LOG.debug('(min,max)=(%f,%f)',fmin,fmax)
                    dataDSet[frame]=d
                    # The timestamp info
                    frameDSet[frame,0]=step
                    frameDSet[frame,1]=time
                except:
                    LOG.exception('Exception reading native field %s, frame %d, from file %s' % (field.name,frame,file))

            if( frames>0 ):
                dataDSet.attrs['max']=fmax
                dataDSet.attrs['min']=fmin
                xDSet[:]=x[:]
                yDSet[:]=y[:]
                zDSet[:]=z[:]
                zgDSet[:]=zg[:]
    except:
        LOG.exception("Exception coverting native to h5")
    finally: 
        h5_file.close()
        LOG.debug('%s closed',h5_filename)
    return h5_filename,frames-frame_start

    
def read_native(filename,scalar=None):
    LOG_RN.debug("Reading data from %s",filename)
    if not os.path.exists(filename):
        raise IOError('File not found: %s',filename)
    try:
        f=open(filename,'rb')
        ver=read_int32(f)
        LOG_RN.debug('Version: %d',ver)
        if scalar!=None:
            nval=1 if scalar else 4
        else:
            if ver < 100:
                nval=1
            elif ver>300:
                nval=4
        LOG_RN.debug('Each value has %d component(s)',nval)
        # extra ghost point in the x and y directions
        ver=ver%100
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
        #LOG_RN.debug('zg=%s',str(zg))

        # compute nx, ny, nz and nb PER CPU
        nx = nxtot/nnx
        ny = nytot/nny
        nz = nztot/nnz
        nb = nblocks/nnb
        npi = (nx+xyp)*(ny+xyp)*nz*nb*nval # the number of values per 'read' block
        LOG_RN.debug('[nx,ny,nz,nb] = [%d,%d,%d,%d]',nx,ny,nz,nb)
        LOG_RN.debug('npi = %d',npi)

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
        y = read_float32(f,nytot)   # y-coordinates
        if nytot==1: y=[y]
        z = read_float32(f,nztot) # z-coordinates
        if nztot==1: z=[z]
        LOG_RN.debug('x=%s',str(x))
        LOG_RN.debug('y=%s',str(y))
        LOG_RN.debug('z=%s',str(z))

        # read the parallel blocks
        if nval==1:
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

                        if nval==1:
                            data_CPU_3D=data_CPU.reshape( (nx,ny,nz),order='F');                   
                            LOG_RN.debug('shape data_CPU_3D: %s',str(data_CPU_3D.shape))
#                           DATA_3D((ixc-1)*nx + (1:nx), (iyc-1)*ny + (1:ny), (izc-1)*nz + (1:nz),(ibc-1)*nb + (1:nb)) = data_CPU_3D; 
                            LOG_RN.debug(' %d:%d, %d:%d, %d:%d ',ixc*nx,ixc*nx+nx, iyc*ny,iyc*ny+ny, izc*nz,izc*nz+nz )
                            DATA_3D[ ixc*nx:ixc*nx+nx, iyc*ny:iyc*ny+ny, izc*nz:izc*nz+nz ] = data_CPU_3D
                        else:
#                           raise Exception('Vector data not supported yet')
                            data_CPU_3D = scale_fac*data_CPU.reshape((nval,nx+xyp,ny+xyp,nz,nb),order='F');
                            LOG_RN.debug('shape data_CPU_3D: %s'%str(data_CPU_3D.shape))
                            # velocity-pressure data
#                           LOG_RN.debug("ixc*nx=%d",ixc*nx)
#                           LOG_RN.debug("(ixc+1)*nx=%d",(ixc+1)*nx)
#                           LOG_RN.debug("iyc*ny=%d",iyc*ny)
#                           LOG_RN.debug("(iyc+1)*ny=%d",(iyc+1)*ny)
#                           LOG_RN.debug("izc*nz=%d",izc*nz)
#                           LOG_RN.debug("(izc+1)*nz=%d",(izc+1)*nz)
#                           LOG_RN.debug("VX_3D.shape=%s",VX_3D.shape)
#                           LOG_RN.debug('(nx,ny,nz)=%s',(nx,ny,nz))
#                           LOG_RN.debug('[ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz]=%s', ((ixc+1)*nx-ixc*nx, (iyc+1)*ny-iyc*ny, (izc+1)*nz-izc*nz))
                            LOG_RN.debug('[ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz]=[%d:%d, %d:%d, %d:%d]', ixc*nx,(ixc+1)*nx, iyc*ny,(iyc+1)*ny, izc*nz,(izc+1)*nz)
#                           LOG_RN.debug('ixc=%d',ixc)
#                           LOG_RN.debug('iyc=%d',iyc)
#                           LOG_RN.debug('izc=%d',izc)
                            VX_3D[ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz] = data_CPU_3D[0,:nx,:ny,:,:].reshape((nx,ny,nz))
                            VY_3D[ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz] = data_CPU_3D[1,:nx,:ny,:,:].reshape((nx,ny,nz))
                            VZ_3D[ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz] = data_CPU_3D[2,:nx,:ny,:,:].reshape((nx,ny,nz))
                            P_3D[ ixc*nx:(ixc+1)*nx, iyc*ny:(iyc+1)*ny, izc*nz:(izc+1)*nz] = data_CPU_3D[3,:nx,:ny,:,:].reshape((nx,ny,nz))
                            LOG_RN.debug('VX_3D[%d,%d,%d]=%g',ixc*nx, iyc*ny, izc*nz, VX_3D[ixc*nx, iyc*ny, izc*nz])

#                           VX_3D[ixc*nx+nxr, iyc*ny+nyr, izc*nz+nzr, ibc*nb+nbr] = data_CPU_3D[0,nxr,nyr,:,:].squeeze()
#                           VY_3D[ixc*nx+nxr, iyc*ny+nyr, izc*nz+nzr, ibc*nb+nbr] = data_CPU_3D[1,nxr,nyr,:,:].squeeze()
#                           VZ_3D[ixc*nx+nxr, iyc*ny+nyr, izc*nz+nzr, ibc*nb+nbr] = data_CPU_3D[2,nxr,nyr,:,:].squeeze()
#                           P_3D[ ixc*nx+nxr, iyc*ny+nyr, izc*nz+nzr, ibc*nb+nbr] = data_CPU_3D[3,nxr,nyr,:,:].squeeze()
        if nval==1:
            DATA_3D=(DATA_3D,)
        else:
            DATA_3D=(VX_3D,VY_3D,VZ_3D,P_3D)
    finally:
        f.close()
    return DATA_3D,istep,time,x,y,z,zg

def read_tracers(filename,callback=lambda x: None):
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
                callback(100*float(i)/float(ntrg))
        tracers={}
        tracers['magic']=magic
        tracers['blocks']=nb_in
        tracers['aspect_ratio']=asp_in
        tracers['time']=time_in
        tracers['varnames']=tracer_varname
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
    return ''.join(struct.unpack('s'*n,f.read(n)))

#
# par file io routines
#

re_string_literal=re.compile("('.*?')")
re_comma=re.compile("\s*,\s*")
re_equal=re.compile("(\s*=\s*)")

def parse_par_line(s):
    tokens=[]
    comment=''
    # Remove any trailiing or leading white space
    s=s.strip()

    # Split the string into string literals and non-literals
    tokens=re_string_literal.split(s)

    # Look for comments
    for n,t in enumerate(tokens):
        if len(t)>2 and t[0]!="'" and t[-1]!="'" and t.find("!")>-1 :
            a,b=t.split("!",1)
            t=a.strip()
            comment = b+ ''.join(tokens[(n+1):])
            tokens=tokens[:n]
            tokens.append(t)

    # compact lists remove whitespace from commas and equal signs
    # split tokens along white space
    n=0
    while n<len(tokens):
        t=tokens[n]
        if len(t)>2 and t[0]!="'" and t[-1]!="'":
            # Remove space around any commas
            if t.find(",")>-1 :
                t=re_comma.sub(",",t)
            # Remove space around any equals
            if t.find("=")>-1 :
                t=re_equal.sub("=",t)
            tokenized=list(itertools.chain(*[t2.split() for t2 in re_equal.split(t)]))
            tokens=tokens[:n]+tokenized+tokens[(n+1):]
            n=n+len(tokenized)  
        else:
            n=n+1

    # concatinate comma seperated values
    n=0
    while n<len(tokens):
        if tokens[n]==",":
            tokens=tokens[:(n-1)]+[tokens[n-1]+","+tokens[n+1]]+tokens[(n+2):]
        n=n+1

    name_value=[]   
    try:
        ndx=tokens.index("=")
        while True:
            name=tokens[ndx-1]
            v=tokens[ndx+1]
            # Convert the value to the approprate type
            if len(v)>1 and v[0]=="'" and v[-1]=="'":
                v=v[1:-1].split("','")
            elif v.startswith(".true.") or v.startswith(".false."):
                v=[v2==".true." for v2 in v.split(",")]
            elif v.find(".")>-1 or v.find('e')>-1:
                v=[float(v2) for v2 in v.split(",")]
            else:
                v=[int(v2) for v2 in v.split(",")]
            if len(v)==1: v=v[0]
            name_value.append( (name,v) )
            ndx=tokens.index("=",ndx+1)
    except ValueError:
        pass
    return name_value,comment

class Par(dict):
    def __init__(self,par=None):
        self.comments={}
        self.errors=[]
        self.sections=['switches','geometry','refstate','boundaries','t_init','timein','viscosity','iteration','multi','ioin','compin','melt','phase','continents','tracersin','plot']
#       self.formats=
#       { 'viscosity': {'stressY_eta': 'mega', 'stress0_eta'}}

        if par==None:
            return
        if os.path.isdir(par):
            par=os.path.join(par,'par')
        LOG_PAR.debug('Reading par file %s',par)
        self.name=par
        self.error=False
        f=open(par,'r')
        p=self # The par file dictionary
        c=self.comments
        c['__file__']=comment_section={}
        for line_no,line in enumerate(f):
            try:
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
                    tokens,comment=parse_par_line(line)
                    if comment!='':
                        try:
                            comment_section[tokens[0][0]]=comment
                        except IndexError:
                            comment_section["line %d" % line_no] = comment
                    for name,value in tokens:
                        section[name]=value
            except:
                LOG_PAR.warning('Unable to parse line: "%s"',line)
                LOG_PAR.warning('Check the error array')
                self.errors.append(sys.exc_info())
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

    def to_python(self,par_obj_name,out=sys.stdout,incl_comments=False,incl_import=False):
        if incl_import:
            out.write('import stagyy.model\n')
            out.write('%s=stagyy.model.Par()\n'%par_obj_name)
        else:
            out.write('%s=Par()\n'%par_obj_name)

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
            
def par_diff(pars,showdefaults=False):
#   from sets import Set
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
                    values.append('_DEFAULT_')
            if showdefaults:
                comp = values
            else:
                comp = [ v for v in values if v!='_DEFAULT_' ] # missing values don't count
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


