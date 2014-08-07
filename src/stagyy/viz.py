import logging 
import os
import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import h5py
import png
import model
from . import field
from .constants import s_in_y

LOG=logging.getLogger(__name__)

def litho_colormap(min,max,boundary=1600.0,width=20):
	if boundary<min or boundary>max:
		bndry=max-50
		b=(bndry-min)/(max-min)
	else:
		b=(boundary-min)/(max-min)

	cdict  = {'red':  ((                  0.0, 0.0 , 0.0),
					   (    (width-1)*b/width, 0.0 , 0.0),
					   (                    b, 0.8 , 1.0),
					   (((width-1)*b+1)/width, 1.0 , 1.0),
					   (                  1.0, 0.4 , 1.0)),

			 'green': ((                  0.0, 0.0 , 0.0),
					   (    (width-1)*b/width, 0.0 , 0.0),
					   (                    b, 0.9 , 0.9),
					   (((width-1)*b+1)/width, 0.0 , 0.0),
					   (                  1.0, 0.0 , 0.0)),

			 'blue':  ((                  0.0, 0.0 , 0.4),
					   (    (width-1)*b/width, 1.0 , 1.0),
					   (                    b, 1.0 , 0.8),
					   (((width-1)*b+1)/width, 0.0 , 0.0),
					   (                  1.0, 0.0 , 0.0))}

	return LinearSegmentedColormap('litho', cdict)

def alpha_colormap(red,green,blue,name,alpha_min=0.0,alpha_max=1.0):
    cdict = {'red':    ((0.0,red,red),
                        (1.0,red,red)),
             'green':  ((0.0,green,green),
                        (1.0,green,green)),
             'blue':   ((0.0,blue,blue),
                        (1.0,blue,blue)),
             'alpha':  ((0.0,alpha_min,alpha_min),
                        (1.0,alpha_max,alpha_max))}
    return LinearSegmentedColormap(name, cdict)

def plot_modeltime(timesteps,prefix,out,title='model time per timestep',ts_max=None,overwrite=False,unit=s_in_y):
	plot_out=os.path.join(out,prefix+'_modeltime.png')
	if not os.path.exists(plot_out):
		LOG.debug("Plotting modeltime to %s",plot_out)
		if ts_max==None: ts_max=len(timesteps.time)
		delta_ts=np.diff(timesteps.time)/unit
		fig=plt.figure(frameon=False)
		fig.clf()
		plt.plot(delta_ts)
		plt.xlim(0,ts_max)
		plt.title(title)
		plt.savefig(plot_out,bbox_inches='tight')
		plt.close()
	else:
		LOG.debug("Skipping existing frame %s",plot_out)


def plot_native_frame(directory,prefix,frame,field):
	d=model.read_native(directory,prefix,frame,field)
	plt.clf()
	plt.imshow(d[:,0,:].T,origin='bottom')
	plt.colorbar()


def plot_frames_2D(frames,dataset,out,mm=None,title='',filter=lambda d: d,cmap=plt.get_cmap('jet'),extent=None,overwrite=False):
	for n in range(len(frames)):
		try:
			plot_out=out%n
		except TypeError:
			plot_out=out
		if not os.path.exists(plot_out) or overwrite:
			LOG.debug("Plotting frame %s",plot_out)
			step,time=frames[n]
			data=dataset[n]
			LOG.debug("data shape : %s",data.shape)
			fig=plt.figure(frameon=False)
			fig.clf()
			plt.imshow(filter(data.T),origin='bottom',cmap=cmap,extent=extent)
			plt.colorbar(shrink=.3,aspect=40)
			if mm!=None:
				plt.clim(filter(mm[0]),filter(mm[1]))
			ma=time/s_in_y/1e6
			plt.title('%s (%0.3f Ma @ timestep %d)'%(title,ma,step))
			plt.savefig(plot_out,bbox_inches='tight')
			plt.close()
		else:
			LOG.debug("Skipping existing frame %s",plot_out)

def plot_field_2D(file,out=None,filter=lambda d:d,cmap_func=lambda min,max: plt.get_cmap('jet'),opts={}):
	if out==None: out=os.path.dirname(file)
	LOG.debug('Plotting file %s to %s',file,out)
	# Open the hdf5 data
	h5file=h5py.File(file,'r')
	try:
		field=h5file.attrs['field']
		# Get the 2D data, timesteps, x and z
		dset=h5file['/data']
		dmax=dset.attrs['max']
		dmin=dset.attrs['min']
		cmap=cmap_func(dmin,dmax)
		frames=h5file['/frame']
		path, ext = os.path.splitext(file) # get the prefix_fieldname
		# extent=(left,right,bottom,top)
		x=h5file['x'].value
		z=h5file['z'].value
		extent=(0,(x[0]+x[-1])/1e3,0,(z[0]+z[-1])/1e3)
		figname=os.path.join(out,os.path.basename(path)+'%05d.png')
		plot_frames_2D(frames,dset.value[:,:,0,:],figname,mm=(dmin,dmax),title=field,filter=filter,cmap=cmap,extent=extent)
	finally:
		h5file.close()

def plot_field_log_2D(file,out=None,opts={}):
	plot_field_2D(file,out=out,filter=lambda d: np.log10(d),opts=opts)

def plot_viscosity_2D(file,out=None,opts={}):
	plot_field_2D(file,out=out,filter=lambda d: np.log10(d),opts=opts)

def plot_temp_2D(file,out=None,opts={}):
	if 'Tlitho' in opts:
		boundary=float(opts['Tlitho'])
	else:
		boundary=1600.0
	plot_field_2D(file,out=out,cmap_func=lambda min,max: litho_colormap(min,max,boundary))

PLOTTERS_2D = {
field.TEMP.prefix: plot_temp_2D,
field.STRAIN_RATE.prefix: plot_field_log_2D,
field.VISCOSITY.prefix: plot_viscosity_2D,
field.DENSITY.prefix: plot_field_2D,
}

def plot_2D(file,out=None,opts={}):
	h5file=h5py.File(file,'r')
	plotter=None
	try:
		field_name=h5file.attrs['field']
		plotter=PLOTTERS_2D[field.by_name[field_name].prefix]
	finally:
		h5file.close()
	if plotter:
		plotter(file,out=out,opts=opts)

def write_frames(frames,filename,renderer,overwrite=True):
    n=0
    for d in frames:
        fname=filename%n
        if overwrite or not os.path.exists(fname):
            out=open(fname,'wb')
            frame=d.squeeze().T[::-1]
            renderer.write(frame,fname)
            out.close()
        yield fname
        n+=1

class Renderer(object):
    def __init__(self,vmin=0.0,vmax=1.0,colormap=plt.get_cmap('jet'),filter=lambda x:x, depth=8,compression=None):
        self.filter=filter
        self.set_minmax(vmin,vmax)
        self.set_depth(depth)
        self.colormap=colormap
        self.compression=compression

    def set_depth(self,depth):
        self._depth=depth
        self._max_val=2**depth-1

    def get_depth(self):
        return self._depth;

    def set_minmax(self,vmin,vmax):
        self.vmin=min(vmin,vmax)
        self.vmax=max(vmin,vmax)
        self.set_normalize(plt.Normalize(self.filter(vmin),self.filter(vmax)))

    def get_minmax(self):
        return (self.vmin,self.vmax)

    def set_normalize(self,norm):
        self._norm=norm

    def get_normalize(self):
        return self._norm

    def as_array(self,a):
        img=(self._max_val*self.colormap(self._norm(self.filter(a)))).astype(int)
        return img.reshape(img.shape[0],img.shape[1]*img.shape[2])

    def render(self,a):
        return png.from_array(self.as_array(a),"RGBA;%d"%self._depth)

    def write(self,a,out,compression=None):
        comp = self.compression if compression==None else compression
        writer=png.Writer(size=a.shape[::-1],alpha=True,bitdepth=self._depth,compression=comp)
        writer.write(out,self.as_array(a))

    def colorbar(self,colors=None,width=20):
        vmin=self.vmin
        vmax=self.vmax
        ncolors=2**self._depth if colors==None else colors
        a=np.vstack([np.linspace(vmax,vmin,ncolors)]*width).T
        return self.render(a);
        
    def write_colorbar(self,filename,colors=None,width=20):
        p=self.colorbar(colors,width)
        p.save(filename)

class Log10Renderer(Renderer):
    def __init__(self,vmin=0.0,vmax=1.0,colormap=plt.get_cmap('jet'), depth=8,compression=None):
        super(Log10Renderer, self).__init__(vmin=vmin,vmax=vmax,colormap=colormap,filter=np.log10,depth=depth,compression=compression)

    def colorbar(self,colors=None,width=20):
        print("log10 colorbar");
        vmin=np.log10(self.vmin)
        vmax=np.log10(self.vmax)
        ncolors=2**self._depth if colors==None else colors
        a=10**np.vstack([np.linspace(vmax,vmin,ncolors)]*width).T
        return self.render(a);
