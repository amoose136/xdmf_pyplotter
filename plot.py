#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function # Anticipating the PY3 apocalypse in 2020
import sys, argparse, csv # For basic file IO stuff
from pdb import set_trace as br #For debugging I prefer the c style "break" nomenclature to "trace"
import time as time_lib
# needed for utf-encoding:
reload(sys)
sys.setdefaultencoding('utf8')
# for io diagnostics:
start_time = time_lib.time()
# construct main parser:
parser = argparse.ArgumentParser(description="Plot variables from XDMF with matplotlib")
parser.add_argument('-quiet','-q',dest='quiet',action='store_const',const=True, help='only display error messages (default full debug messages)')
parser.add_argument('-settings','-s',dest='settingsfile',required=True)
parser.add_argument('-threads','-t',dest='threads')
# create subparser for all the plot settings:
settings_parser=argparse.ArgumentParser(description="Take input settings for matplotlib",prog='./plot.py file[01..N].xmf -s plot.config where plot.config contains:')
# Create list of arguments that appear in the argparser so it knows to add a '-' in front if there isn't one:
argslist=[\
'cmap',\
'cbar_scale',\
'cbar_domain',\
'cbar_enabled',\
'cbar_orientation',\
'title',\
'title_enabled',\
'title_font',\
'smooth_zones',\
'image_format',\
'image_size',\
'x_range_km',\
'x_range_title',\
'y_range_km',\
'y_range_title',\
'time_format',\
'bounce_time_enabled',\
'elapsed_time_enabled',\
'variable',\
'zoom_value']
#
#Create type checkers:
#
def check_bool(value):
	# account for booleans using aNy CombINAtioN of cases
	if value.upper()=='FALSE' or value.upper()=='DISABLE' or value=='0': 
		return False
	elif value.upper()=='TRUE' or value.upper()=='ENABLE' or value=='1':
		return True
	else: 
		raise argparse.ArgumentTypeError("%s is an invalid boolean value" % value)
def check_int(value):
	if int(value):
		return int(value)
	elif value=='auto':
		return 'auto'
	else:
		raise argparse.ArgumentTypeError("%s is an invalid int value" % value)
def check_float(value):
	if float(value):
		return float(value)
	elif value=='auto':
		return 'auto'
	else:
		raise argparse.ArgumentTypeError("%s is an invalid float value" % value)
# create subparser arguments:
settings_parser.add_argument('-variable',required=True,type=str,metavar='AttributeName',help='The attribute to plot like \'Entropy\', or \'Density\', etc. The name must match the XDMF attribute tags')
settings_parser.add_argument('-cmap',default='hot_desaturated',help='Colormap to use for colorbar')#done
settings_parser.add_argument('-cbar_scale',type=str,default='lin',choices=['lin','log'],metavar="{{lin},log}",help='Linear or log scale colormap')
settings_parser.add_argument('-bar_domain',type=check_float,nargs=2,metavar=("{{auto},min}","{{auto},max}"),default=['auto','auto'],help='The domain of the color bar')
settings_parser.add_argument('-cbar_enabled',type=check_bool,choices=[True,False],metavar='{{True},False}',nargs=1,default=True,help='enable or disable colorbar')
settings_parser.add_argument('-cbar_orientation',type=str,choices=['vertical','horizontal'],metavar='{{\'vertical\'},\'horizontal\'}',default='vertical',help='set the colorbar to orientation')
settings_parser.add_argument('-title',type=str,metavar='{{AttributeName},str}',help='Define the plot title that goes above the plot',default='AttributeName')
settings_parser.add_argument('-title_enabled',type=check_bool,choices=[True,False],metavar='{{True},False}',default=True,help='enable or disable the title that goes above the plot')
settings_parser.add_argument('-title_font',type=str,metavar='str',help='choose the font of the plot title')
settings_parser.add_argument('-smooth_zones',type=check_bool,choices=[True,False],metavar='{True,{False}}',help='disable or enable zone smoothing')
settings_parser.add_argument('-image_format',type=str,choices=['png','svg','pdf','ps','jpeg','gif','tiff','eps'],default='png',metavar="{{'png'},'svg','pdf','ps','jpeg','gif','tiff','eps'}",help='specify graph output format')
settings_parser.add_argument('-image_size',type=check_int,nargs=2,metavar='int',default=[1280,710],help='specify the size of image')
settings_parser.add_argument('-x_range_km',type=check_float,nargs=2,metavar=("{{auto},min}","{{auto},max}"),default=['auto','auto'],help='The range of the x axis in km')
settings_parser.add_argument('-y_range_km',type=check_float,nargs=2,metavar=("{{auto},min}","{{auto},max}"),default=['auto','auto'],help='The range of the y axis in km')
settings_parser.add_argument('-x_range_title',type=str,nargs=1,metavar='str',default='X ($10^3$ km)',help='The text below the x axis')
settings_parser.add_argument('-y_range_title',type=str,nargs=1,metavar='str',default='Y ($10^3$ km)',help='The text to the left of the y axis')
settings_parser.add_argument('-time_format',type=str,metavar='{{seconds}, s, ms, milliseconds}',nargs=1,default='seconds',choices=["seconds", "s", "ms", "milliseconds"],help='Time format code to use for elapsed time and bounce time')
settings_parser.add_argument('-bounce_time_enabled',type=check_bool,choices=[True,False],metavar='{{True},False}',default=True,help='Boolean option for "time since bounce" display')
settings_parser.add_argument('-elapsed_time_enabled',type=check_bool,choices=[True,False],metavar='{{True},False}',default=True,help='Boolean option for "elapsed time" display')
settings_parser.add_argument('-zoom_value',type=check_float,help='The zoom value (percentage of total range) to use if the x or y range is set to \'auto\'')
args=parser.parse_args()
# define an error printing function for error reporting to terminal STD error IO stream
def eprint(*arg, **kwargs):
	print(*arg, file=sys.stderr, **kwargs)

#define a standard printing function that only functions if there is no silence flag on script invocation
def qprint(*arg,**kwargs):
	if not args.quiet:
		print(*arg,**kwargs)
# Define settings
if args.settingsfile and args.settingsfile!='':
	settingsargs=[]
	for supper_arg in csv.reader(open(args.settingsfile).read().split('\n'),delimiter=' ',quotechar='"',escapechar='\\'):
		for arg in supper_arg:
			# account for the required '-' needed for argparse
			if arg in argslist:
				settingsargs.append('-'+arg) 
			else: 
				settingsargs.append(arg)
	settings=settings_parser.parse_args(settingsargs)
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
import matplotlib
try:
	import matplotlib.pyplot as plt
except:
	eprint('Fatal Error: matplotlib.pyplot not found!')
	sys.exit()

# Library needed for custom "Candybar" colormap:
from matplotlib.colors import LinearSegmentedColormap

reader = XdmfReader.New()
dom = XdmfReader.read(reader,'../chimera/2d/chimera_00774_grid_1_01.xmf')
# br()
# dom = XdmfReader.read(reader,'../chimera/3d/2D_chimera_step-00400.xmf')

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
# 	return 'Theta=%1.4f, r=%9.4g, %s=%1.4f'%(x, y, 'entropy',entropy[max(0,np.where(azimuths<x)[0][-1]-1),max(0,np.where(zeniths<y)[0][-1]-1)])
fig = plt.figure()
fig.set_size_inches(12.1, 7.2)
ax = fig.add_subplot(111)

ax.set_title(settings.title,y=1.05)
ax.set_xlabel(settings.x_range_title)
ax.set_ylabel(settings.y_range_title)

# # Setup mouse-over string to interrogate data interactively when in polar coordinates
# def format_coord(x, y):

# Setup mouse-over string to interrogate data interactively when in cartesian coordinates
def format_coord(x, y):
	ia=np.where(azimuths<cart2pol(x,y)[1])[0][-1]
	ib=np.where(zeniths<cart2pol(x,y)[0])[0][-1]
	return 'Theta=%1.4f (rad), r=%9.4g, %s=%1.3f'%(cart2pol(x,y)[1], cart2pol(x,y)[0], 'entropy',entropy[ia,ib])

# plt.axis([theta.min(), theta.max(), rad.min(), rad.max()/60])
#create a function to splice in manually specified values if need be
def detect_auto(defaults,value):
	output=[]
	for a,b in zip(defaults,value):
		output.append([a,b][b!='auto'])
	return output
zoomvalue=1./90
if settings.zoom_value and settings.zoom_value!='auto':
	zoomvalue=settings.zoom_value
plt.axis(detect_auto([x.min()*zoomvalue, x.max()*zoomvalue, y.min(), y.max()*zoomvalue],settings.x_range_km+settings.y_range_km))
plt.margins(enable=False,axis='both')
ax.format_coord = format_coord

# fig.subplots_adjust(bottom=0)
# Define the colors that make up the "hot desaturated" in VisIt
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

plt.axes().set_aspect('equal', 'box')
# Create colorbar ("hot desaturated" in VisIt)
hot_desaturated=LinearSegmentedColormap('hot_desaturated',cdict,N=256,gamma=1.0)
test=ax.pcolormesh(x, y, entropy,cmap=[hot_desaturated,settings.cmap][settings.cmap!='hot_desaturated'])
plt.colorbar(test,ax=ax,shrink=[.98,.68][settings.cbar_orientation=='vertical'],pad=[.05,.01][settings.cbar_orientation=='vertical'],orientation=settings.cbar_orientation)
# plt.colorbar(test,ax=ax,extend='max')

# Comment and uncomment the next line to save the image:
plt.savefig('test.png',format='png',bbox_inches='tight') 
qprint('time elapsed:	'+str(time_lib.time()-start_time))
del start_time
from subprocess import call
call(['qlmanage -p test.png &> /dev/null'],shell=True)
# plt.show()
# Comment and uncomment the next line to show and interactive plot after optionally saving the image (above):
