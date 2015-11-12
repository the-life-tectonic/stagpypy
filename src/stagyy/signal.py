import logging 

WALLTIME=False

LOG=logging.getLogger(__name__)

def handler(signum,frame):
        LOG.debug('Caught signal %d'%signum)
        global WALLTIME
        WALLTIME=True
        LOG.debug('WALLTIME=%s'%str(WALLTIME))
        

