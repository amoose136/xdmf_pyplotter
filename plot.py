#!/usr/bin/env python
from __future__ import print_function # Anticipating the PY3 apocalypse in 2020
import sys # For basic file IO stuff
from pdb import set_trace as br #For debugging I prefer the c style "break" nomenclature to "trace"

# define an error printing function for error reporting to terminal STD error IO stream
def eprint(*arg, **kwargs):
	print(*arg, file=sys.stderr, **kwargs)

#define a standard printing function that only functions if there is no silence flag on script invocation
def qprint(*arg,**kwargs):
	if not args.quiet:
		print(*arg,**kwargs)

# Carefully import xdmf
try:
	from Xdmf import *
except:
	eprint("Fatal Error: Xdmf python bindings not found")	
	eprint("\tIf on Mac, try \"$brew install tdsmith/xdmf/xdmf --HEAD\"")
	sys.exit()

# Carefully import numpy
try:
	import numpy as np
	def pol2cart(rho, phi):
		x = rho * np.cos(phi)
		y = rho * np.sin(phi)
		return(x, y)
	def cart2pol(x, y):
		rho = np.sqrt(x**2 + y**2)
		phi = np.arctan2(y, x)
		return(rho, phi)
except:
	eprint('Fatal Error: numpy not found!')
	sys.exit()

# Carefully import matplotlib
try:
	import matplotlib.pyplot as plt
except:
	eprint('Fatal Error: matplotlib.pyplot not found!')
	sys.exit()

# Library needed for custom "Candybar" colormap
from matplotlib.colors import LinearSegmentedColormap

reader = XdmfReader.New()
dom = XdmfReader.read(reader,'../chimera/chimera_00774_grid_1_01.xmf')
grid = dom.getRectilinearGrid(0)
time=grid.getTime().getValue()

grid.getCoordinates(0).read()
zeniths=np.array([float(piece) for piece in grid.getCoordinates(0).getValuesString().split()]) #terrible fallback to get data but nothing else works for coordinates it seems
grid.getCoordinates(1).read()
azimuths=np.array([float(piece) for piece in grid.getCoordinates(1).getValuesString().split()])
rad, phi = np.meshgrid(zeniths, azimuths)
x,y=pol2cart(rad,phi)
grid.getAttribute('Entropy').read()
entropy=np.frombuffer(grid.getAttribute('Entropy').getBuffer()).reshape((rad.shape[0]-1,rad.shape[1]-1))
fig = plt.figure()
ax = fig.add_subplot(111)

# # Setup output string to intergate data interactively for polar
# def format_coord(x, y):
# 	return 'Theta=%1.4f, r=%9.4g, %s=%1.4f'%(x, y, 'entropy',entropy[max(0,np.where(azimuths<x)[0][-1]-1),max(0,np.where(zeniths<y)[0][-1]-1)])

# Setup output string to intergate data interactively for cartesian coordinates
def format_coord(x, y):
	return 'Theta=%1.4f (rad), r=%9.4g, %s=%1.4f'%(cart2pol(x,y)[1], cart2pol(x,y)[0], 'entropy',entropy[max(0,np.where(azimuths<cart2pol(x,y)[0])[0][-1]-1),max(0,np.where(zeniths<cart2pol(x,y)[1])[0][-1]-1)])

# plt.axis([theta.min(), theta.max(), rad.min(), rad.max()/60])
zoomvalue=1./90
plt.axis([x.min()*zoomvalue, x.max()*zoomvalue, y.min(), y.max()*zoomvalue])
ax.format_coord = format_coord

# fig.subplots_adjust(bottom=0)
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
test=ax.pcolormesh(x, y, entropy,cmap=candybar)
plt.colorbar(test,ax=ax)
plt.axes().set_aspect('equal', 'datalim')
# plt.savefig('test',format='svg')
plt.show()