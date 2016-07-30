import os
import stat
from datetime import datetime
from setuptools import setup

def read(fname):
	    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def most_recent_mod(directory):
	mod=0;
	for dirpath, dirnames, filenames in os.walk(directory): 
		for filename in filenames:
			fname=os.path.join(dirpath,filename)
			stats=os.stat(fname)
			mod=max(mod,stats[stat.ST_MTIME])
	return mod

src='src/stagyy'

ver=datetime.fromtimestamp(most_recent_mod(src)).strftime('%Y.%m.%d.%H.%M')

setup(
	name='stagyy',
#	install_requires=['of_xml>=0.0.1','of_util>=0.0.1'],
	description='Python modules in support of StagYY',
	author='Robert I. Petersen',
	author_email='rpetersen@ucsd.edu', 
	version='0.5.0',
	scripts=['src/scripts/pardiff.py'],
	package_dir={'stagyy': src},
	packages=['stagyy','stagyy.ic','stagyy.image'], 
	license='GPL 2.0', 
	classifiers=[
'Development Status :: 4 - Beta',
'Intended Audience :: Developers',
'License :: OSI Approved :: GNU General Public License (GPL)',
'Programming Language :: Python'
	],
	long_description=read('README')
)
