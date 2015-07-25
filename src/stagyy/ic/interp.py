import logging 
import time
import numpy as np

LOG=logging.getLogger(__name__)

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
