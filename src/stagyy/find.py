import re
import os
import glob

RECLASS=re.compile('').__class__

def find(patterns,path=os.getcwd()):
    if isinstance(patterns,str):
        patterns=[patterns]
	if len(patterns)==0: return None
	return find_regex(patterns,path) if isinstance(patterns[0],RECLASS) else find_glob(patterns,path)
   
__call__=find

def find_regex(patterns,path=os.getcwd()):
    return _find(patterns,regex_check,path)

def find_glob(patterns,path=os.getcwd()):
    return _find(patterns,glob_check,path)

def _find(patterns,check,path):
    if path[-1] != '/':
        path=path+'/'
    if not os.path.isdir(path):
        raise ValueError('%s is not a directory or does not exist'%path)
    dirs=set()
    os.path.walk(path,check,{'dirs':dirs,'patterns':patterns})
    dirs=sorted(dirs)
    return dirs

def glob_check(args,dirname,names):
    dirs=args['dirs']
    patterns=args['patterns']
    if all([len(glob.glob(os.path.join(dirname,p)))>0 for p in patterns]):
        # if it is a model then truncate the 'names' list so we stop walking
        # down the tree and add the directory to the list of models
        names[:]=[]
        dirs.add(dirname)

def regex_check(args,dirname,names):
    dirs=args['dirs']
    patterns=args['patterns']
    if all([any([pattern.match(f) for f in names]) for pattern in patterns]):
        # if it is a model then truncate the 'names' list so we stop walking
        # down the tree and add the directory to the list of models
        names[:]=[]
        dirs.add(dirname)

