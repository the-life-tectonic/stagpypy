import logging 
import time
import numpy as np
import stagyy.signal as ssig

LOG=logging.getLogger(__name__)

def h5_img(h5):
    """
    Converts a StagyYY frame of the form (n,x,y,z) to (n,rows,cols) suitable for imaging.  
    This function expects square pixels are square.  Use h5_2d_img_interpolate if the
    pixels need to be squared.
    """
    pass

def h5_img_interpolate(h5,Lx,Lz,px=None,pz=None):
    """
    Converts a StagyYY frame of the form (n,x,y,z) to (n,rows,cols) suitable for imaging.  
    This function interpolates the image so that the pixles are square.
    """
    import scipy.interpolate 
    #from mpl_toolkits.basemap import interp
    if 'data' in h5:
        frames,nx,ny,nz=h5['data'].shape
    elif 'p' in h5:
        frames,nx,ny,nz=h5['p'].shape
    elif 'strinv' in h5:
        frames,nx,ny,nz=h5['strinv'].shape
    else:
        raise ValueError("H5 file has neither data nor pressure nor stress")
    LOG.debug("Frames: %d",frames)
    LOG.debug("Grid size (nx,ny,nz): (%d,%d,%d)",nx,ny,nz)
    if ny==1:
        LOG.debug("2D frame using x and z")
        x=h5['x'].value
    elif nx==1:
        LOG.debug("2D frame using y and z")
        x=h5['y'].value
    else:
        LOG.error("Unexpected grid size (%d,%d,%d) either x or y should have size 1 for 2d interpolation",nx,ny,nz)
        raise ValueError("Unexpected grid size (%d,%d,%d) either x or y should have size 1 for 2d interpolation"%(nx,ny,nz))

    z=h5['z'].value
    LOG.debug("Size (x,z): (%d,%d)",len(x),len(z))
    if not px:
        px=nx
    if not pz:
        pz=int(nx*Lz/Lx)
    px=int(px)
    pz=int(pz)
    LOG.debug("Number of pixels in image (px,pz): (%d,%d)",px,pz)
    dx=Lx/px
    dz=Lz/pz
    LOG.debug("dx,dz: (%d,%d)",dx,dz)
    x_new=(np.arange(px)+.5)*dx
    z_new=(np.arange(pz)+.5)*dz
    Z_new,X_new=np.meshgrid(z_new,x_new)

    #  Create the image group if it doesn't exist
    if not 'image' in h5:
        img=h5.create_group('image')
        img.attrs['order']='rc'
        for dset_name in ['data','p','v','vx','vy','vz','strinv','strx','stry','strz']:
            if dset_name in h5: 
                LOG.debug('Creating dataset %s with size (r,c) %d,%d',dset_name,pz,px)
                img.create_dataset(dset_name, (0,pz,px),compression='gzip', compression_opts=4,maxshape=(None,pz,px))
        # Create the x and z datasets
        img.create_dataset('x', data=x_new,compression='gzip', compression_opts=4)
        img.create_dataset('z', data=z_new,compression='gzip', compression_opts=4)
        # link the frames, and y points
        # img['y']=h5['y']
        img['frame']=h5['frame']
    else:
        img=h5['image']
    
    for dset_name in ['data','p','v','vx','vy','vz','strinv','strx','stry','strz']:
        if ssig.WALLTIME:
            break

        if dset_name in h5:
            data_set=h5[dset_name]
            img_set=img[dset_name]
            data_frames=data_set.shape[0]
            img_frames=img_set.shape[0]
            LOG.debug("Interoplating %s from %d to %d",dset_name,img_frames,data_frames)
            img_set.attrs['order']='rc'
            img_set.attrs['min']=data_set.attrs['min']
            img_set.attrs['max']=data_set.attrs['max']
            img_set.resize((data_frames,pz,px))
            start_time=time.time()
            for n in xrange(img_frames,data_frames):
                data=np.squeeze(data_set[n])
                LOG.debug('Uninterpolated data shape: %s',data.shape)

                #Interpolate using basemap
                #img_set[n]=interp(data,z,x,Z_new,X_new)

                f=scipy.interpolate.interp2d(z,x,data)
                interp_img=f(z_new,x_new)
                LOG.debug('Interpolated data shape: %s',interp_img.shape)
                img_set[n]=interp_img.T[::-1]
                LOG.debug('Image shape: %s',img_set[n].shape)

                #f=interpolate.RectBivariateSpline(x,z,data)
                #img_set[n]=f(x_new,z_new)

                delta_t=time.time()-start_time
                fps=(n+1-img_frames)/delta_t
                eta=(data_frames-n)/fps
                eta_hour=int(eta/3600)
                eta_min=int((eta-eta_hour*3600)/60)
                eta_sec=int(eta-eta_hour*3600-eta_min*60)
                LOG.debug("frame %d, fps %0.1f, estimated completion in %02d:%02d:%02d",n,fps,eta_hour,eta_min,eta_sec)
                if ssig.WALLTIME:
                    LOG.warning("Wall time about to be excedded, cleaning up")
                    # truncate the image matrix
                    img_set.resize((n,pz,px))
                    break

def square_pixels(h5,img,Lx,Lz,px=None,pz=None):
    """
    Converts a StagyYY frame of the form (n,x,y,z) to (n,rows,cols) suitable for imaging.  
    This function interpolates the image so that the pixles are square.
    """
    import scipy.interpolate 
    #from mpl_toolkits.basemap import interp
    data_set=h5['data']
    frames,nx,ny,nz=data_set.shape
    LOG.debug("Frames: %d",frames)
    LOG.debug("Grid size (nx,ny,nz): (%d,%d,%d)",nx,ny,nz)
    if ny==1:
        LOG.debug("2D frame using x and z")
        x=h5['x'].value
    elif nx==1:
        LOG.debug("2D frame using y and z")
        x=h5['y'].value
    else:
        LOG.error("Unexpected grid size (%d,%d,%d) either x or y should have size 1 for 2d interpolation",nx,ny,nz)
        raise ValueError("Unexpected grid size (%d,%d,%d) either x or y should have size 1 for 2d interpolation"%(nx,ny,nz))

    z=h5['z'].value
    LOG.debug("Size (x,z): (%d,%d)",len(x),len(z))
    if not px:
        px=nx
    if not pz:
        pz=int(nx*Lz/Lx)
    px=int(px)
    pz=int(pz)
    LOG.debug("Image size in pixels (px,pz): (%d,%d)",px,pz)
    dx=Lx/px
    dz=Lz/pz
    LOG.debug('Dimensional pixel size (dx,dz): (%0.1f,%0.1f)',dx,dz)
    x_new=(np.arange(px)+.5)*dx
    z_new=(np.arange(pz)+.5)*dz
    Z_new,X_new=np.meshgrid(z_new,x_new)

    #  Create the data dataset in img if it doesn't exist
    img.attrs['order']='rc'
    if 'data' not in img:
        LOG.debug('Creating dataset data with size (r,c) %d,%d',pz,px)
        img.create_dataset('data', (0,pz,px),compression='gzip', compression_opts=4,maxshape=(None,pz,px))
    # Create the frames dataset if it doesn't exist
    if 'frame' not in img:
        LOG.debug('Creating dataset frame with size %s',str(data_set.shape))
        img.create_dataset('frame',h5['frame'].shape,compression='gzip',maxshape=(None,)+h5['frame'].shape[1:])
    img['frame'].resize(h5['frame'].shape)
    img['frame'][:]=h5['frame'][:]

    # Create the x and z datasets
    if 'x' not in img:
        img.create_dataset('x', data=x_new, compression='gzip', compression_opts=4)
    if 'z' not in img:
        img.create_dataset('z', data=z_new, compression='gzip', compression_opts=4)
    
    if ssig.WALLTIME:
        return

    img_set=img['data']
    data_frames=data_set.shape[0]
    img_frames=img_set.shape[0]
    LOG.debug("Interoplating data from %d to %d",img_frames,data_frames)
    img_set.attrs['order']='rc'
    img_set.attrs['min']=data_set.attrs['min']
    img_set.attrs['max']=data_set.attrs['max']
    img_set.resize((data_frames,pz,px))
    start_time=time.time()
    for n in xrange(img_frames,data_frames):
        data=np.squeeze(data_set[n])
        LOG.debug('Uninterpolated data shape: %s',data.shape)

        #Interpolate using basemap
        #img_set[n]=interp(data,z,x,Z_new,X_new)

        f=scipy.interpolate.interp2d(z,x,data)
        interp_img=f(z_new,x_new)
        LOG.debug('Interpolated data shape: %s',interp_img.shape)
        img_set[n]=interp_img.T[::-1]
        LOG.debug('Image shape: %s',img_set[n].shape)

        #f=interpolate.RectBivariateSpline(x,z,data)
        #img_set[n]=f(x_new,z_new)

        delta_t=time.time()-start_time
        fps=(n+1-img_frames)/delta_t
        eta=(data_frames-n)/fps
        eta_hour=int(eta/3600)
        eta_min=int((eta-eta_hour*3600)/60)
        eta_sec=int(eta-eta_hour*3600-eta_min*60)
        LOG.debug("frame %d, fps %0.1f, estimated completion in %02d:%02d:%02d",n,fps,eta_hour,eta_min,eta_sec)
        if ssig.WALLTIME:
            LOG.warning("Wall time about to be excedded, cleaning up")
            # truncate the image matrix
            img_set.resize((n,pz,px))
            break
