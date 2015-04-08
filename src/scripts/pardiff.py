#!/usr/bin/env python
import argparse
import logging
import os
import sys
import stagyy.model as model



# By default we arn't verbose
verbose=False
# Setup a logger and formatter
name = os.path.basename(sys.argv[0])
logger = logging.getLogger(name)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s (%(process)d) - %(levelname)s - %(message)s")

def output_txt(diffs):
	# Calculate the width needed for the section/key label
	label_width=max([ len(h['section'])+len(h['key']) for h in diffs])+1
	# Calculate the column widths
	col_widths=[len(f) for f in args.par]
	for d in diffs:
		for i in range(len(d['values'])):
			col_widths[i]=max(col_widths[i],len(str(d['values'][i])))

	fmt='%%-%ds : '%label_width
	line=(' '*label_width)+'   '
	header=(' '*label_width)+'   '
	for i in range(len(args.par)):
		w=col_widths[i]
		fmt=fmt+('%%%ds '%w)
		hdr='%%%ds '%w
		header=header+(hdr%args.par[i])
		line=line+('-'*w)+' '

	print(header)
	print(line)
	for d in diffs:
		v=[d['section']+'/'+d['key']]
		map(v.append,[str(val) for val in d['values']])
		try:
			print(fmt%tuple(v))
		except:
			logger.error('Unable to print pardiff',exc_info=True)
			logger.error('format = "%s"',fmt)
			logger.error('values = %s',v)
			logger.error('len(values) = %d',len(v))

OUTPUTS= { 'txt': output_txt }

if __name__ == "__main__":
#	import atexit
#	atexit.register(logExit)
	# Set up the screen handler for errors
	screen_handler=logging.StreamHandler()
	screen_handler.setFormatter(formatter)
	screen_handler.setLevel(logging.ERROR)
	logger.addHandler(screen_handler)
	model.LOG.addHandler(screen_handler)

	parser = argparse.ArgumentParser('Compares multiple par files, based on values (better than diff)')
	parser.add_argument('par',nargs='*',help='Name of the par files to compare')
	parser.add_argument('-v',dest='verbose',action='store_true',help='Verbose output')
	parser.add_argument('-f',dest='format',default='txt',help='Output type')
	args=parser.parse_args()

	if args.verbose:
		screen_handler.setLevel(logging.DEBUG)
		logger.setLevel(logging.DEBUG)
		model.LOG.setLevel(logging.DEBUG)

	if len(args.par)<2:
		parser.print_help()
		sys.exit()

	# Calculate the differences
	diffs=model.par_diff(args.par,True)
    OUTPUTS[args.format](diffs) 

