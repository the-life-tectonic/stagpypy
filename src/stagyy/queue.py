import sys
import math

class Job(object):
	def __init__(self,name,walltime,cpus,alloc_cpus,account,queue,commands,cwd=True,stdout=None,stderr=None,combine_stdout_stderr=False,current_env=True,envlist=None,notifications=None,email=None):
		self.name=name
		self.walltime=walltime
		self.cpus=cpus
		self.alloc_cpus=alloc_cpus
		self.account=account
		self.queue=queue
		self.cwd=cwd
		self.stdout=stdout
		self.stderr=stderr
		self.combine_stdout_stderr=combine_stdout_stderr
		self.current_env=current_env
		self.envlist=envlist
		self.notifications=notifications
		self.email=email
		self.commands=commands

	def __str__(self,jobname=''):
		return "<%s_job name:%s cpu:%d queue:%s walltime:%d>" % (jobname,self.name,self.cpus, self.queue, self.walltime)
	
	def write_header(self,out):
		import getpass
		import time
		from socket import gethostname
		from . import VERSION
		out.write('\n#\n')
		out.write('# Created by %s@%s\n'%(getpass.getuser(),gethostname()))
		out.write('# %s\n'%time.strftime('%H:%M:%S %d-%m-%Y UTC',time.gmtime()))
		out.write('# %s version %s\n'%(__name__,VERSION))
		out.write('#\n\n')

class BASH(Job):
	"""
	Creats a batch script that runs in serial
	"""
	def __init__(self,*args,**kwargs):
		self.extension='sh'
		super(BASH,self).__init__(*args,**kwargs)

	def __str__(self):
		return super(BASH,self).__str__('bash')

	def write(self,out=sys.stdout,incl_header=True):
		# Set the shell
		out.write('#!/bin/bash\n')
		# write the header
		if incl_header: self.write_header(out)
		# set the name
		out.write("# name: %s\n"%self.name)

		# set the name of stdout
		if self.stdout!=None: 
			out.write("OUT_FILE=%s\n"%self.stdout)
		else: 
			out.write("OUT_FILE=%s.$$.out\n"%self.name)
		# write the commands
		out.write("\n\n")
		for cmd in self.commands:
			out.write(cmd)
			out.write("\n")

class PBS(Job):
	""" 
	Creates a PBS batch job
	"""
	def __init__(self,*args,**kwargs):
		self.extension='pbs'
		super(PBS,self).__init__(*args,**kwargs)

	def __str__(self):
		return super(PBS,self).__str__('pbs')

	def write(self,out=sys.stdout,incl_header=True):
		# Set the shell
		out.write('#!/bin/bash\n')
		# write the header
		if incl_header: self.write_header(out)
		# set the name
		out.write("#PBS -N %s\n"%self.name)
		# set the walltime
		if type(self.walltime)==str:
			out.write("#PBS -l walltime=%s\n"%self.walltime)
		elif type(self.walltime)==int:
			hr=self.walltime/3600
			minute=(self.walltime-hr*3600)/60
			sec=(self.walltime-hr*3600)-minute*60
			out.write("#PBS -l walltime=%02d:%02d:%02d\n"%(hr,minute,sec))
		# allocate the cpu
		out.write("#PBS -l size=%d\n"%self.alloc_cpus)
		# set the account
		out.write("#PBS -A %s\n"%self.account)
		# set the queue
		if self.queue!=None: out.write('#PBS -q %s # queue\n'%self.queue.name)
		# set the cwd
		if self.cwd: self.commands.insert(0,"cd $PBS_O_WORKDIR")
		# set the name of stdout
		if self.stdout!=None : out.write("#PBS -o %s\n"%self.stdout)
		# set the name of stderr
		if self.stderr!=None : out.write("#PBS -e %s\n"%self.stderr)
		# combine stdout and stderr
		if self.combine_stdout_stderr : out.write("#PBS -j oe\n")
		# use env when job was submitted
		if self.current_env : out.write("#PBS -V\n")
		# set the notification level
		if self.notifications!=None : out.write("#PBS -m %s\n" % ''.join(set([ c for c in self.notifications if c in 'aben' ])) )
		# set the notification email address
		if self.email!=None : out.write("#PBS -M %s\n"%self.email)
		# write the commands
		out.write("\n\n")
		for cmd in self.commands:
			out.write(cmd)
			out.write("\n")

class SGE(Job):
	"""
	Creates an SGE batch job.
	"""
	def __init__(self,*args,**kwargs):
		self.extension='sge'
		self.way=kwargs['way']
		del kwargs['way']
		super(SGE,self).__init__(*args,**kwargs)

	def __str__(self):
		return super(PBS,self).__str__('sge')

	def write(self,out=sys.stdout,incl_header=True):
		# Set the shell
		out.write('#!/bin/bash\n')
		# Write the header
		if incl_header: self.write_header(out)
		# set the name
		out.write('#$ -N %s # job name\n'%self.name)
		# set the walltime
		if type(self.walltime)==str:
			out.write('#$ -l h_rt=%s\n'%self.walltime)
		elif type(self.walltime)==int:
			hr=self.walltime/3600
			minute=(self.walltime-hr*3600)/60
			sec=(self.walltime-hr*3600)-minute*60
			out.write('#$ -l h_rt=%02d:%02d:%02d\n'%(hr,minute,sec))
		# allocate the cpu
		out.write('#$ -pe %dway %d # tasks per node and total cores\n'%(self.way,self.alloc_cpus))
		if self.alloc_cpus!=self.cpus: self.commands.insert(0,"export MY_NSLOTS=%d"%self.cpus)
		# set the account
		out.write('#$ -A %s # account\n'%self.account)
		# set the queue
		if self.queue!=None: out.write('#$ -q %s # queue\n'%self.queue.name)
		# set the cwd
		if self.cwd: out.write('#$ -cwd # start job in the current working directory\n')
		# set the environment and runtime directory
		if self.current_env: out.write('#$ -V # Use current environment setting in batch job\n')
		# set the name of stdout
		if self.stdout!=None : out.write("#$ -o %s\n"%self.stdout)
		# set the name of stderr
		if self.stderr!=None : out.write("#$ -e %s\n"%self.stderr)
		# combine stdout and stderr
		if self.combine_stdout_stderr : out.write("#$ -j\n")
		# set the notification level
		if self.notifications!=None : out.write("#$ -m %s\n" % ''.join(set([ c for c in self.notifications if c in 'abens' ])) )
		# set the notification email address
		if self.email!=None : out.write("#$ -M %s\n"%self.email)
		# write the commands
		out.write("\n\n")

class SLURM(Job):
	"""
	Creates an SLURM batch job.
	"""
	def __init__(self,*args,**kwargs):
		self.extension='sbat'
		super(SLURM,self).__init__(*args,**kwargs)

	def __str__(self):
		return super(PBS,self).__str__('slurm')

	def write(self,out=sys.stdout,incl_header=True):
		# Set the shell
		out.write('#!/bin/bash\n')
		# Write the header
		if incl_header: self.write_header(out)
		# set the name
		out.write('#SBATCH -J %s # job name\n'%self.name)
		# set the walltime
		if type(self.walltime)==str:
			out.write('#SBATCH -t %s\n'%self.walltime)
		elif type(self.walltime)==int:
			hr=self.walltime/3600
			minute=(self.walltime-hr*3600)/60
			sec=(self.walltime-hr*3600)-minute*60
			out.write('#SBATCH -t %02d:%02d:%02d\n'%(hr,minute,sec))
		# allocate the cpu
		out.write('#SBATCH -n %d # total number of mpi tasks\n'%self.cpus)
		# set the account
		out.write('#SBATCH -A %s # account\n'%self.account)
		# set the queue
		if self.queue!=None: out.write('#SBATCH -p %s # queue\n'%self.queue.name)
		# set the cwd
		#if self.cwd: out.write('#SBATCH -cwd # start job in the current working directory\n')
		# set the environment and runtime directory
		#if self.current_env: out.write('#SBATCH -V # Use current environment setting in batch job\n')
		# set the name of stdout
		if self.stdout!=None : out.write("#SBATCH -o %s\n"%self.stdout)
		# set the name of stderr
		if self.stderr!=None : out.write("#SBATCH -e %s\n"%self.stderr)
		# combine stdout and stderr
		#if self.combine_stdout_stderr : out.write("#SBATCH -j\n")
		
		# set the notification level
		if self.notifications!=None and len(self.notifications)>0: 
			if len(self.notifications)>1:
				notifications='ALL'
			else:
				notifications={'e':'END','b':'BEGIN','a':'FAIL','s':'FAIL'}[self.notifications]
			out.write("#SBATCH --mail-type %s\n" % notifications)
		# set the notification email address
		if self.email!=None : out.write("#SBATCH --mail-user %s\n"%self.email)
		# write the commands
		out.write("\n\n")
		for cmd in self.commands:
			out.write(cmd)
			out.write("\n")
		
class Queue(object):
	def __init__(self,name,max_cpus,max_walltime):
		self.name=name
		self.max_cpus=max_cpus
		self.max_walltime=max_walltime

	def __str__(self):
		return "<Queue:%s>" % self.name

	def __lt__(self,other):
		return self.max_cpus<other.max_cpus or ( self.max_cpus==other.max_cpus and self.max_walltime<other.max_walltime)

	def __le__(self,other):
		return self.max_cpus<=other.max_cpus or ( self.max_cpus==other.max_cpus and self.max_walltime<=other.max_walltime)

	def __gt__(self,other):
		return self.max_cpus>other.max_cpus or ( self.max_cpus==other.max_cpus and self.max_walltime>other.max_walltime)

	def __ge__(self,other):
		return self.max_cpus>=other.max_cpus or ( self.max_cpus==other.max_cpus and self.max_walltime>=other.max_walltime)

	def __eq__(self,other):
		return self.max_cpus==other.max_cpus and self.max_walltime==other.max_walltime

	def __ne__(self,other):
		return not (type(other)==Queue and self.max_cpus==other.max_cpus and self.max_walltime==other.max_walltime)

class System(object): 
	def __init__(self,name,queues,scheduler):
		self.name=name
		self.queues=queues
		self.scheduler=scheduler
		
	def __str__(self):
		return "<System:%s>" % self.name

	def set_options(self,kwargs):
		pass;

	def get_queue(self,cpus,walltime):
		for q in sorted(self.queues):
			if q.max_cpus>=cpus and q.max_walltime>=walltime:
				break
		return q

	def allocate_cpus(self,cpus):
		return cpus

_jms=dict()
#
#  Local system
#
LOCAL=System('Local Serial', [Queue('serial',2,1e6),Queue('mpi',2,1e6)], BASH)
def local_mpi_exec(cpus,mpi_command):
	cmd=''
	if cpus==1:
		cmd="%s 2>&1 | tee $OUT_FILE "%(mpi_command) 
	else:
		cmd="mpiexec -n %d %s 2>&1 | tee $OUT_FILE "%(cpus,mpi_command) 
	return cmd
LOCAL.mpi_exec=local_mpi_exec
_jms['__DEFAULT__']=LOCAL

#
#  Kraken
#
KRAKEN=System('Kraken', [Queue('small',512,24*3600), Queue('medium',8192,24*3600), Queue('large',49536,24*3600), Queue('capability',98352,48*3600), Queue('dedicated',112896,48*3600)], PBS)
KRAKEN.mpi_exec=lambda cpus,mpi_command: "aprun -n %d %s"%(cpus,mpi_command) 
def kraken_options(kwargs): kwargs['current_env']=False
KRAKEN.set_options=kraken_options
KRAKEN.get_queue=lambda cpus,qalltime: None
KRAKEN.allocate_cpus = lambda cpus: int(math.ceil(cpus/12.0)*12)
_jms['kraken']=KRAKEN

#
# Stampede
#
STAMPEDE=System('Stampede', [Queue('normal',4096,24*3600),Queue('development',256,4*3600),Queue('serial',16,12*3600),Queue('large',16385,24*3600)], SLURM)
def stampede_options(kwargs):
	pass
STAMPEDE.set_options=stampede_options
STAMPEDE.mpi_exec=lambda cpus,mpi_command: "ibrun %s"%mpi_command 
STAMPEDE.allocate_cpus = lambda cpus: int(math.ceil(cpus/16.0)*16)
_jms['stampede']=STAMPEDE

# On February 4, 2013 the Ranger compute cluster and the Spur visualization cluster will be decommissioned
#RANGER=System('Ranger', [Queue('normal',4096,24*3600) , Queue('long',1024,48*3600) , Queue('large',16384,24*3600) , Queue('development',256,2*3600) , Queue('serial',16,12*3600)], SGE)
#RANGER.mpi_exec=lambda cpus,mpi_command: "ibrun %s"%mpi_command 
#def ranger_options(kwargs): 
#	if 'way' not in kwargs:
#		kwargs['way']=16
#RANGER.set_options=ranger_options
#RANGER.allocate_cpus = lambda cpus: int(math.ceil(cpus/16.0)*16)

def get_job_management_system():
	import re
	import socket
	jms=None
	hostname = socket.gethostname()
	for regex in _jms.keys():
		if len(re.findall(regex,hostname))>0:
			jms=_jms[regex]
			break
	if jms==None:
		jms=_jms['__DEFAULT__']
	return jms






def get_job(system,name,walltime,cpus,account,mpi_command,**kwargs):
# Select the correct sized queue from the system
	q=system.get_queue(cpus,walltime)
	print("Got queue %s"%q);
	system.set_options(kwargs)	
	if 'commands' not in kwargs: kwargs['commands']=[]
	kwargs['commands'].append(system.mpi_exec(cpus,mpi_command))
	return system.scheduler(name,walltime,cpus,system.allocate_cpus(cpus),account,q,**kwargs)
