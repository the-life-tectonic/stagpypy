import logging 
import os
import inspect
import numpy as np
import Queue
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
from threading import Thread
import png

# The logger for this module
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

    cm=LinearSegmentedColormap('lithosphere_%d'%boundary, cdict)
    plt.register_cmap(cmap=cm)
    return cm

def alpha_colormap(name,red,green,blue,alpha_min=0.0,alpha_max=1.0):
    cdict = {'red':    ((0.0,red,red),
                        (1.0,red,red)),
             'green':  ((0.0,green,green),
                        (1.0,green,green)),
             'blue':   ((0.0,blue,blue),
                        (1.0,blue,blue)),
             'alpha':  ((0.0,alpha_min,alpha_min),
                        (1.0,alpha_max,alpha_max))}
    cm=LinearSegmentedColormap(name, cdict)
    plt.register_cmap(cmap=cm)
    return cm


def write_frames(frames,filename,renderer,filter=lambda x:x,overwrite=True):
    n=0
    for d in frames:
        fname=filename%n
        if overwrite or not os.path.exists(fname):
            frame=filter(d.squeeze().T[::-1])
            renderer.write(frame,fname)
        yield fname
        n+=1

class ColormapManager(object):
    def __init__(self,dir):
        self.dir=dir
        if not os.path.exists(self.dir):
            os.makedirs(self.dir)
        for cm in plt.cm.datad.keys():
            self._plot_colormap(cm);

    def get_colormaps(self):
        return plt.cm.datad.keys()

    def add_colormap(self,cm):
        self._plot_colormap(cm.name);

    def _plot_colormap(self,cm):
        r=Renderer(colormap=plt.get_cmap(cm))
        name=os.path.join(self.dir,cm+"_h.png")
        if not os.path.exists(name):
            f=open(name,'wb')
            r.write_h_colorbar(f)
            f.close()
        name=os.path.join(self.dir,cm+"_v.png")
        if not os.path.exists(name):
            f=open(name,'wb')
            r.write_v_colorbar(f)
            f.close()
    
    def get_hcolormap(self,cm):
        return os.path.join(self.dir,cm+"_h.png")

    def get_vcolormap(self,cm):
        return os.path.join(self.dir,cm+"_v.png")

class Renderer(object):
    def __init__(self,vmin=None,vmax=None,
                 filter=lambda x:x, 
                 depth=8,compression=None,
                 colormap=plt.get_cmap('jet'),
                 over_color=None, over_alpha=1.0,
                 under_color=None, under_alpha=1.0,
                 bad_color=None, bad_alpha=1.0,
                 gamma=1.0):
        self.filter=filter

        if vmin!=None and vmax!=None:
            self.set_minmax(float(vmin),float(vmax))
        else:
            self.vmin=None
            self.vmax=None
            self._norm=None
        if compression!=None:
            compression=int(compression)
        self.compression=compression

        self.set_depth(int(depth))

        if isinstance(colormap,str):
            self.colormap=plt.get_cmap(colormap)
        else:
            self.colormap=colormap
        if over_color!=None:
            try:
                int(over_color,16)
                over_color='#'+over_color
            except ValueError:
                pass
            self.colormap.set_over(over_color,alpha=float(over_alpha))
        if under_color!=None:
            try:
                int(under_color,16)
                under_color='#'+under_color
            except ValueError:
                pass
            self.colormap.set_under(under_color,alpha=float(under_alpha))
        if bad_color!=None:
            self.colormap.set_bad(bad_color,alpha=float(bad_alpha))
        self.colormap.set_gamma(float(gamma))
                
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

    def as_rgba(self,a):
        norm=self._norm if self._norm!=None else plt.Normalize(self.filter(a.min()),self.filter(a.max()))
        img=(self._max_val*self.colormap(norm(self.filter(a)))).astype(int)
        return img.reshape(img.shape[0],img.shape[1]*img.shape[2])

    def write(self,a,out,compression=None):
        comp = self.compression if compression==None else compression
        writer=png.Writer(size=(a.shape[1],a.shape[0]),alpha=True,bitdepth=self._depth,compression=comp)
        writer.write(out,self.as_rgba(a))

    def _colorbar_array(self,colors,size):
        ncolors=2**self._depth if colors==None else colors
        return np.vstack([np.linspace(ncolors,0,ncolors)]*size)

    def write_h_colorbar(self,out,colors=None,height=20):
        a = self._colorbar_array(colors,height)[:,::-1]
        writer=png.Writer(size=a.shape[::-1],alpha=True,bitdepth=self._depth)
        writer.write(out,self.as_rgba(a))

    def write_v_colorbar(self,out,colors=None,width=20):
        a = self._colorbar_array(colors,width).T
        writer=png.Writer(size=a.shape[::-1],alpha=True,bitdepth=self._depth)
        writer.write(out,self.as_rgba(a))

class Log10Renderer(Renderer):
    def __init__(self,**kwargs):
        kwargs['filter']=np.log10
        super(Log10Renderer, self).__init__(**kwargs)

    def colorbar(self,colors=None,width=20):
        vmin=np.log10(self.vmin)
        vmax=np.log10(self.vmax)
        ncolors=2**self._depth if colors==None else colors
        a=10**np.vstack([np.linspace(vmax,vmin,ncolors)]*width).T
        return self.render(a);

def symlog10(a,threshold=1,fillvalue=0):
    masked=np.ma.masked_inside(a,-threshold,threshold)
    sign = np.sign(a)
    result=np.ma.log10(abs(masked))*sign
    result.data[result.mask]=fillvalue
    return result.data

class SymLog10Renderer(Renderer):
    def __init__(self,**kwargs):
        kwargs['filter']=symlog10
        super(SymLog10Renderer, self).__init__(**kwargs)

    def colorbar(self,colors=None,width=20):
        vmin=symlog10(self.vmin)
        vmax=symlog10(self.vmax)
        ncolors=2**self._depth if colors==None else colors
        a=10**np.vstack([np.linspace(vmax,vmin,ncolors)]*width).T
        return self.render(a);

def get_init_args(klass,args):
    try:
        argspec=inspect.getargspec(klass.__init__)
        args.update(argspec.args)
        if argspec.keywords!=None:
            for b in klass.__bases__:
                get_init_args(b,args)
    except TypeError:
        pass

class Gallery(Thread):
    def __init__(self):
        super(Gallery, self).__init__()
        self.img_queue=Queue.Queue()
        self._renderers={}
        self._renderer_classes={}
        self._progress={}
        self.running=False

    def register_renderer(self,name,klass,defaults):
        init_args=set()
        get_init_args(klass,init_args)
        self._renderer_classes[name]=(klass,init_args,defaults)

    def get_renderer(self,renderer_name):
        try:
            r_class,init_args,defaults=self._renderer_classes[renderer_name]
        except KeyError:
            r_class=Renderer
            init_args=set()
            get_init_args(r_class,init_args)
            defaults={}
        return r_class,init_args,defaults

    def kwargs_to_str(self,name,kwargs):
        return name+'-'+','.join([str(k)+'='+str(v) for k,v in sorted(kwargs)])

    def get_img(self,dir,renderer_name,frame_num,kwargs):
        r_class,init_args,defaults=self.get_renderer(renderer_name)
        custom_args=[i for i in kwargs.items() if i[0] in init_args]
        args_str=self.kwargs_to_str(renderer_name,custom_args)
        img_file=os.path.join(dir,"%s-%05d.png"%(args_str,frame_num))
        if os.path.exists(img_file):
            return img_file
        else:
            return None

    def render_img(self,dir,renderer_name,frame_num,frame,kwargs):
        r_class,init_args,defaults=self.get_renderer(renderer_name)
        custom_args=[i for i in kwargs.items() if i[0] in init_args]
        args_str=self.kwargs_to_str(renderer_name,custom_args)
        args=defaults.copy()
        args.update(custom_args)
        img_file=os.path.join(dir,"%s-%05d.png"%(args_str,frame_num))
        if os.path.exists(img_file):
            return img_file
        try:
            renderer=self._renderers[args_str]
        except KeyError:
            renderer=r_class(**args)
            self._renderers[args_str]=renderer
        out=open(img_file,'wb')
        renderer.write(frame,out)
        out.close()
        return img_file

    def progress(self,model,field_name,renderer_name,kwargs):
        r_class,init_args,defaults=self.get_renderer(renderer_name)
        custom_args=[i for i in kwargs.items() if i[0] in init_args]
        args_str=self.kwargs_to_str(renderer_name,custom_args)
        return self._progress[model.name+field_name+args_str]

    def enqueue(self,model,field_name,renderer_name,kwargs):
        r_class,init_args,defaults=self.get_renderer(renderer_name)
        custom_args=[i for i in kwargs.items() if i[0] in init_args]
        args_str=self.kwargs_to_str(renderer_name,custom_args)
        args=defaults.copy()
        args.update(custom_args)
        try:
            renderer=self._renderers[args_str]
        except KeyError:
            renderer=r_class(**args)
            self._renderers[args_str]=renderer
        self.img_queue.put( (model,field_name,renderer,args_str) )
        
    def run(self):
        LOG.info("Gallery thread started")
        self.running=True;
        while self.running:
            try:
                model,field_name,renderer,args_str=self.img_queue.get(True,100)
                LOG.debug("rendering %s begun",args_str)
                img_pattern=os.path.join(model.img_dir,"%s-%%05d.png"%args_str)
                model[field_name].checkout()
                data=model[field_name].frames
                frame_count=data.shape[0]
                for frame_num in xrange(frame_count):
                    LOG.debug("rendering %s frame %d/%d",field_name,frame_num,frame_count)
                    img_file=img_pattern%frame_num
                    if not os.path.exists(img_file):
                        out=open(img_file,'wb')
                        renderer.write(data[frame_num],out)
                        out.close()
                    self._progress[model.name+field_name+args_str]=(frame_num+1,frame_count)
                LOG.debug("rendering %s compelete",args_str)
                model[field_name].checkin(close=True)

            except Queue.Empty:
                pass
