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
from matplotlib.colors import LinearSegmentedColormap
reader = XdmfReader.New()
dom = XdmfReader.read(reader,'../chimera/chimera_00774_grid_1_01.xmf')
grid = dom.getRectilinearGrid(0)
time=grid.getTime().getValue()

grid.getCoordinates(0).read()
zeniths=np.array([float(piece) for piece in grid.getCoordinates(0).getValuesString().split()]) #terrible fallback to get data but nothing else works for coordinates it seems
grid.getCoordinates(1).read()
azimuths=np.array([float(piece) for piece in grid.getCoordinates(1).getValuesString().split()])
rad, theta = np.meshgrid(zeniths, azimuths)
grid.getAttribute('Entropy').read()
entropy=np.frombuffer(grid.getAttribute('Entropy').getBuffer()).reshape((rad.shape[0]-1,rad.shape[1]-1))
fig = plt.figure()
ax = fig.add_subplot(111,polar='True')
def format_coord(x, y):
	return 'Theta=%1.4f, r=%9.4g, entropy=%1.4f'%(x, y, entropy[max(0,np.where(azimuths<x)[0][-1]-1),max(0,np.where(zeniths<y)[0][-1]-1)])
plt.axis([theta.min(), theta.max(), rad.min(), rad.max()/60])
ax.format_coord = format_coord

# fig.subplots_adjust(bottom=-.9)
cdict = {'red':((.000, 0.263, 0.263),
			(0.143, 0.000, 0.000),
			(0.286, 0.000, 0.000),
			(0.429, 0.000, 0.000),
			(0.571, 1.000, 1.000),
			(0.714, 1.000, 1.000),
			(0.857, 0.420, 0.420),
			(1.000, 0.878, 0.878)),

		 'green':((.000, 0.263, 0.263),
			(0.143, 0.000, 0.000),
			(0.286, 1.000, 1.000),
			(0.429, 0.498, 0.498),
			(0.571, 1.000, 1.000),
			(0.714, 0.376, 0.376),
			(0.857, 0.000, 0.000),
			(1.000, 0.298, 0.298)),

		 'blue':((.000, 0.831, 0.831),
			(0.143, 0.357, 0.357),
			(0.286, 1.000, 1.000),
			(0.429, 0.000, 0.000),
			(0.571, 0.000, 0.000),
			(0.714, 0.000, 0.000),
			(0.857, 0.000, 0.000),
			(1.000, 0.294, 0.294)),
}
candybar=LinearSegmentedColormap('test',cdict,N=256,gamma=1.0)
test=ax.pcolormesh(theta, rad, entropy,cmap=candybar)
plt.colorbar(test,ax=ax)
plt.show()