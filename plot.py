#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function # Anticipating the PY3 apocalypse in 2020
import sys, argparse, csv, re # For basic file IO stuff, argument parsing, config file reading, text substitution/ regex
from pdb import set_trace as br #For debugging I prefer the c style "break" nomenclature to "trace"
import time as time_lib # for diagnostices
import six
# needed for utf-encoding on python 2:
if six.PY2:
	reload(sys)
	sys.setdefaultencoding('utf8')
# for io diagnostics:
start_time = time_lib.time()
# construct main parser:
parser = argparse.ArgumentParser(description="Plot variables from XDMF with matplotlib")
group=parser.add_mutually_exclusive_group(required=True)
parser.add_argument('--quiet','-q',dest='quiet',action='store_const',const=True, help='only display error messages (default full debug messages)')
group.add_argument('--settings','-s',dest='settingsfile',help='A settings file full of plotting options')
group.add_argument('--tree',help='Display layout of available data as found by xdmf',action='store_true',default=False)
parser.add_argument('--threads','-t',dest='threads', help='number of threads to use for parallel operations')
parser.add_argument('files',metavar='frame_###.xmf',nargs='+',help='xdmf files to plot using the settings files')

# create subparser for all the plot settings:
settings_parser=argparse.ArgumentParser(description="Take input settings for matplotlib",prog='./plot.py file[01..N].xmf -s plot.config where plot.config contains:')
# Create list of arguments that appear in the argparser so it knows to add a '-' in front if there isn't one:
argslist=[\
'cmap',\
'cbar_scale',\
'cbar_domain',\
'cbar_enabled',\
'cbar_location',\
'cbar_width',\
'title',\
'title_enabled',\
'title_font',\
'smooth_zones',\
'image_format',\
'image_size',\
'x_range_km',\
'x_range_label',\
'y_range_km',\
'y_range_label',\
'time_format',\
'bounce_time_enabled',\
'elapsed_time_enabled',\
'variable',\
'zoom_value',\
'background_color',\
'text_color',\
'label_font_size',\
'title_font_size',\
]
try:
	# Library needed for custom "Candybar" colormap:
	from matplotlib.colors import LinearSegmentedColormap,is_color_like
except ImportError():
	eprint('matplotlib not found')
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

def check_color(value):
	if is_color_like(value):
		return value
	else:
		raise argparse.ArgumentTypeError("%s is an invalid color value" % value)
# create subparser arguments:
settings_parser.add_argument('-variable',required=True,type=str,metavar='AttributeName',help='The attribute to plot like \'Entropy\', or \'Density\', etc. The name must match the XDMF attribute tags')
settings_parser.add_argument('-cmap',default='hot_desaturated',help='Colormap to use for colorbar')#done
settings_parser.add_argument('-background_color',type=check_color,default='white',help='color to use as background')#done
settings_parser.add_argument('-text_color',type=check_color,default='black',help='color to use for text and annotations')
settings_parser.add_argument('-cbar_scale',type=str,default='lin',choices=['lin','log'],metavar="{{lin},log}",help='Linear or log scale colormap')
settings_parser.add_argument('-cbar_domain',type=check_float,nargs=2,metavar=("{{auto},min}","{{auto},max}"),default=['auto','auto'],help='The domain of the color bar')
settings_parser.add_argument('-cbar_enabled',type=check_bool,choices=[True,False],metavar='{{True},False}',nargs=1,default=True,help='enable or disable colorbar')
settings_parser.add_argument('-cbar_location',type=str,choices=['left','right','top','bottom'],metavar='{\'left\',{\'right\'},\'top\',\'bottom\'}',default='right',help='set the colorbar position')
settings_parser.add_argument('-cbar_width',type=check_float,metavar='float',default='5.0',help='The width of the colorbar')
settings_parser.add_argument('-title',type=str,metavar='{{AttributeName},str}',help='Define the plot title that goes above the plot',default='AttributeName')
settings_parser.add_argument('-title_enabled',type=check_bool,choices=[True,False],metavar='{{True},False}',default=True,help='enable or disable the title that goes above the plot')
settings_parser.add_argument('-title_font',type=str,metavar='str',help='choose the font of the plot title')
settings_parser.add_argument('-title_font_size',type=check_int,metavar='int',help='font size for title')
settings_parser.add_argument('-label_font_size',type=check_int,metavar='int',help='font size for axis labels')
settings_parser.add_argument('-smooth_zones',type=check_bool,choices=[True,False],metavar='{True,{False}}',help='disable or enable zone smoothing')
settings_parser.add_argument('-image_format',type=str,choices=['png','svg','pdf','ps','jpeg','gif','tiff','eps'],default='png',metavar="{{'png'},'svg','pdf','ps','jpeg','gif','tiff','eps'}",help='specify graph output format')
settings_parser.add_argument('-image_size',type=check_int,nargs=2,metavar='int',default=[1280,710],help='specify the size of image')
settings_parser.add_argument('-x_range_km',type=check_float,nargs=2,metavar=("{{auto},min}","{{auto},max}"),default=['auto','auto'],help='The range of the x axis in km')
settings_parser.add_argument('-y_range_km',type=check_float,nargs=2,metavar=("{{auto},min}","{{auto},max}"),default=['auto','auto'],help='The range of the y axis in km')
settings_parser.add_argument('-x_range_label',type=str,nargs=1,metavar='str',default='X ($10^3$ km)',help='The text below the x axis')
settings_parser.add_argument('-y_range_label',type=str,nargs=1,metavar='str',default='Y ($10^3$ km)',help='The text to the left of the y axis')
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
	eprint("\tIf on Mac, try \"$brew install amoose136/xdmf/xdmf --HEAD\"")
	eprint("\tThis should work for python2.x bindings but I have no guarantee for python3.x yet")
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
	import matplotlib
	matplotlib.use('AGG')#change backend
	import matplotlib.pyplot as plt
	from matplotlib.colorbar import make_axes
	from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
	eprint('Fatal Error: matplotlib or parts of it not found!')
	sys.exit()

def valid_grid_names():
	grids=[]
	for i in range(0,dom.getNumberRectilinearGrids()): 
		grids.append(dom.getRectilinearGrid(i).getName())
	return grids
def valid_variables(gridname):
	variables=[]
	for i in range(0,dom.getRectilinearGrid(gridname).getNumberAttributes()):
		variables.append(dom.getRectilinearGrid(gridname).getAttribute(i).getName())
	return variables
def tree():
	qprint('Found valid scalars:')
	for grid in valid_grid_names():
		qprint('\n'+grid,end='')
		variables=valid_variables(grid)
		for i,variable in enumerate(variables):
			qprint(['\n\t',''][i!=0 or i==len(variables)]+variable+(','+'\n\t'*((i+1)%5==0))*(i!=len(variables)-1),end=' ')
	qprint()


for file in args.files:
	reader = XdmfReader.New()
	dom = XdmfReader.read(reader,file)
	if args.tree:
		tree()
		sys.exit()
	if re.search('.*\/(?!.+\/)',file):
		file_directory = re.search('.*\/(?!.+\/)',file).group()
	if re.search('(?<=abundance/)([a-z]{1,2})/?(\d+)',settings.variable.lower()):
		match=re.search('(?<=abundance/)([a-z]{1,2})/?(\d+)',settings.variable.lower())
		varname=match.group(2)
		TrueVarname=match.group(1).title()+varname
		gridname='Abundance/'+match.group(1).title()
		TrueGridname='Abundance'
	else:
		match=settings.variable.split('/')
		gridname=match[0]
		varname=match[1]
		TrueVarname=varname
		TrueGridname=gridname
	del match
	image_name = re.search('(?!.*\/).*',file).group()[:-4]+'_'+re.search('(?!.+\/.+)(?!\/).+',settings.variable).group().lower()+'.'+settings.image_format
	reader = XdmfReader.New()
	dom = XdmfReader.read(reader,file)
	grid = dom.getRectilinearGrid(gridname)
	try:
		grid.getCoordinates(0).read()
		zeniths=np.array([float(piece) for piece in grid.getCoordinates(0).getValuesString().split()]) #terrible fallback to get data but nothing else works for coordinates it seems
		grid.getCoordinates(1).read()
		azimuths=np.array([float(piece) for piece in grid.getCoordinates(1).getValuesString().split()])
	except AttributeError:
		eprint('Error: Invalid grid')
		eprint('\t'+settings.variable+' provided a grid not found in the XDMF')
		eprint('\tGrid tried was: '+gridname)
		sys.exit()
	rad, phi = np.meshgrid(zeniths, azimuths)
	x,y=pol2cart(rad,phi)
	try:
		variable=grid.getAttribute(varname)
		variable.read()
	except AttributeError:
		eprint("Error: Invalid attribute")
		eprint("\t"+settings.variable+" not found in "+file)
		eprint("\tPath looked for was :"+gridname+"/"+varname)
		sys.exit()
	variable=np.frombuffer(variable.getBuffer()).reshape((rad.shape[0]-1,rad.shape[1]-1))
	# entropy=grid.getAttribute('Entropy')
	# entropyr=entropy.getReference()
	# variable=dom.getRectilinearGrid('Abundance/He').getAttribute('3')
	# variable.read()
	# variable = np.frombuffer(dom.getRectilinearGrid('Abundance/He').getAttribute('3').getBuffer()).reshape((rad.shape[0]-1,rad.shape[1]-1))
	# variable=np.frombuffer(entropyr.read().getBuffer()).reshape((rad.shape[0]-1,rad.shape[1]-1))
	# entropy=grid.getAttribute('Entropy').getNumpyArray().reshape(180,540,order='A')
	
	# plt.pcolormesh(entropy);plt.show()
	fig = plt.figure(figsize=(12.1,7.2))
	fig.set_size_inches(12.1, 7.2,forward=True)
	sp=fig.add_subplot(111)
	# br()
	# ax.set_ymargin(.2)
	# # Setup mouse-over string to interrogate data interactively when in polar coordinates
	# def format_coord(x, y):
	# 	return 'Theta=%1.4f, r=%9.4g, %s=%1.4f'%(x, y, 'entropy',entropy[max(0,np.where(azimuths<x)[0][-1]-1),max(0,np.where(zeniths<y)[0][-1]-1)])

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
	zoomvalue=1./90 #defaults
	if settings.zoom_value and settings.zoom_value!='auto':
		zoomvalue=settings.zoom_value
	plt.axis(detect_auto([x.min()*zoomvalue, x.max()*zoomvalue, y.min(), y.max()*zoomvalue],settings.x_range_km+settings.y_range_km))
	plt.axes().set_aspect('equal')
	ax=plt.axes()
	# ax.format_coord = format_coord
	
	# fig.subplots_adjust(bottom=0)
	# Define the colors that make up the "hot desaturated" in VisIt:
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

	# Create colorbar ("hot desaturated" in VisIt)


	hot_desaturated=LinearSegmentedColormap('hot_desaturated',cdict,N=256,gamma=1.0)
	pcolor=sp.pcolormesh(x, y, variable,cmap=[hot_desaturated,settings.cmap][settings.cmap!='hot_desaturated'])
	
	for atr in ['title','x_range_label','y_range_label']:
		settings.__setattr__(atr,re.sub(r'\\var(?=[^i]|$)',TrueVarname,settings.__getattribute__(atr)))
		settings.__setattr__(atr,re.sub(r'\\variable',TrueVarname.lower(),settings.__getattribute__(atr)))
		settings.__setattr__(atr,re.sub(r'\\Variable',TrueVarname.title(),settings.__getattribute__(atr)))
		settings.__setattr__(atr,re.sub(r'\\grid',TrueGridname,settings.__getattribute__(atr)))
		settings.__setattr__(atr,re.sub(r'\\path',TrueGridname+'/'+TrueVarname,settings.__getattribute__(atr)))
	title=sp.set_title(settings.title,fontsize=settings.title_font_size)
	xlabel=sp.set_xlabel(settings.x_range_label,fontsize=settings.label_font_size)
	ylabel=sp.set_ylabel(settings.y_range_label,fontsize=settings.label_font_size)
	
	cbar_orientation=['vertical','horizontal'][settings.cbar_location in ['top','bottom']]
	divider=make_axes_locatable(ax)
	cax=divider.append_axes(settings.cbar_location,size=str([settings.cbar_width,'5'][str(settings.cbar_width)=='auto'])+"%",pad=[[.8,.4][settings.cbar_location=='top'],[.1,.8][settings.cbar_location=='left']][(cbar_orientation=='vertical')])# note that this last setting, pad, is done in a sneaky way. True + True = 2 in python. ¯\_(ツ)_/¯
	cbar=plt.colorbar(pcolor,cax=cax,orientation=cbar_orientation)
	# settings for colorbar ticks positioning:
	if settings.cbar_location in ['top','bottom']:
		cax.xaxis.set_ticks_position(settings.cbar_location)
	else:
		cax.yaxis.set_ticks_position(settings.cbar_location)

	fig.suptitle('this is the figure title', fontsize=12,)
	plt.tight_layout()
	# br()
	# plt.figtext(1,0,'Elapsed time:'+str(grid.getTime().getValue()),horizontalalignment='center',transform=ax.transAxes,bbox=dict(facecolor='red', alpha=0.5))
	# Comment and uncomment the next line to save the image:
	plt.savefig(image_name,format=settings.image_format,facecolor=settings.background_color,orientation='landscape') 
	qprint('time elapsed:	'+str(time_lib.time()-start_time))
	del start_time
	# Comment and uncomment the next 2 lines to show the image on mac (must have saved the image first):
	from subprocess import call # for on-the-fly lightning fast image viewing on mac
	call(['qlmanage -p '+image_name+' &> /dev/null'],shell=True) # for on-the-fly lightning fast image viewing on mac
	# Comment and uncomment th next line to show the image on other platforms or if you haven't saved it first:
	# plt.show() #Built in interactive viewer for non-macOS platforms. Slower.