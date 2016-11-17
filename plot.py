#!/usr/bin/env python
from __future__ import print_function
import sys
from pdb import set_trace as br #For debugging I prefer the c style "break" nomenclature to "trace"
# define an error printing function for error reporting to terminal STD error IO stream
def eprint(*arg, **kwargs):
	print(*arg, file=sys.stderr, **kwargs)
#define a standard printing function that only functions if there is no silence flag on script invocation
def qprint(*arg,**kwargs):
	if not args.quiet:
		print(*arg,**kwargs)
try:
	from Xdmf import *
except:
	eprint("Fatal Error: Xdmf python bindings not found")	
	eprint("\tIf on Mac, try \"$brew install tdsmith/xdmf/xdmf --HEAD\"")
	sys.exit()
try:
	import numpy as np
except:
	eprint('Fatal Error: numpy not found!')
	sys.exit()
try:
	import matplotlib.pyplot as plt
except:
	eprint('Fatal Error: matplotlib.pyplot not found!')
	sys.exit()
reader = XdmfReader.New()
dom = XdmfReader.read(reader,'../chimera/chimera_00774_grid_1_01.xmf')
grid = dom.getRectilinearGrid(0)
time=grid.getTime().getValue()
temp=grid.getAttribute('Temperature')
temp.read()
temp2=np.array([float(piece) for piece in temp.getValuesString().split()])
br()
