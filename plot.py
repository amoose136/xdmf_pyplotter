#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function # Anticipating the PY3 apocalypse in 2020
import sys, argparse, csv, re, platform # For basic file IO stuff, argument parsing, config file reading, text substitution/ regex, OS identification
from pdb import set_trace as br #For debugging I prefer the c style "break" nomenclature to "trace"
import time as time_lib # for diagnostices
import six
import traceback

# needed for utf-encoding on python 2:
if six.PY2:
        reload(sys)
        sys.setdefaultencoding('utf-8')
# for io diagnostics:
start_time = time_lib.time()
# construct main parser:
parser = argparse.ArgumentParser(description="Plot variables from XDMF with matplotlib")
group=parser.add_mutually_exclusive_group(required=True)
parser.add_argument('--quiet','-q',dest='quiet',action='store_const',const=True, help='only display error messages (default full debug messages)')
group.add_argument('--settings','-s',dest='settingsfile',help='A settings file full of plotting options')
group.add_argument('--tree',help='Display layout of available data as found in XDMF and exit',action='store_true',default=False)
group.add_argument('--vars',help='Display full paths to all valid variables and exit',action='store_true',default=False)
parser.add_argument('--threads','-t',dest='threads', help='number of threads to use for parallel operations')
parser.add_argument('--directory','-d',dest='dir',help='The directory to output the graphs to.')
parser.add_argument('--debug',help='show result in window',action='store_true',default=False)
parser.add_argument('files',metavar='frame_###.xmf',nargs='+',help='xdmf files to plot using the settings files')

#define a help flag pseudo overloaded flag that also prints the help of the subparser for the settings file:
def print_help():
        if '-h' in sys.argv or '--help' in sys.argv:
                parser.print_help()
                print('\nBut also there are many settings contained in the plaintext settings file, plot.config:\n')
                settings_parser.print_help()
                sys.exit()
# create subparser for all the plot settings:
settings_parser=argparse.ArgumentParser(description="Input plot settings for matplotlib to use",prog='plot.config parser',prefix_chars=u'•')

# define an error printing function for error reporting to terminal STD error IO stream
def eprint(*arg, **kwargs):
        print(*arg, file=sys.stderr, **kwargs)

#define a standard printing function that only functions if there is no silence flag on script invocation
def qprint(*arg,**kwargs):
        if not args.quiet:
                print(*arg,**kwargs)

# import h5py with checking
try:
        import h5py
except Exception as e:
        eprint('Fatal Error: H5PY not found! (used for reading heavy data)')
        traceback.print_exception(type(e),e,sys.exc_info()[2])
        sys.exit()

# import matplotlib with checking
try:
        import matplotlib as mpl
        mpl.use('AGG')#change backend
        import matplotlib.pyplot as plt
        from matplotlib.colorbar import make_axes
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.colors import LinearSegmentedColormap,is_color_like,LogNorm
except ImportError as e:
        eprint('Fatal Error: matplotlib or parts of it not found!')
        traceback.print_exception(type(e),e,sys.exc_info()[2])
        sys.exit()

#Create type checkers:
def check_bool(value):
        # account for booleans using aNy CombINAtioN of cases
        if value.upper()=='FALSE' or value.upper()=='DISABLE' or value=='0': 
                return False
        elif value.upper()=='TRUE' or value.upper()=='ENABLE' or value=='1':
                return True
        else: 
                raise argparse.ArgumentTypeError("%s is an invalid boolean value" % value)
def check_int(value):
        try:
                return int(value)
        except ValueError:
                if value=='auto':
                        return 'auto'
                else:
                        raise argparse.ArgumentTypeError("%s is an invalid int value" % value)
def check_float(value):
        try:
                return float(value)
        except ValueError:
                if value=='auto':
                        return 'auto'
                else:
                        raise argparse.ArgumentTypeError("%s is an invalid float value" % value)
def check_color(value):
        if is_color_like(value):
                return value
        else:
                raise argparse.ArgumentTypeError("%s is an invalid color value" % value)

# create subparser for settings file arguments:
settings_parser.add_argument(u'•variable',type=str,metavar='AttributeName',help='The attribute to plot like \'Entropy\', or \'Density\', etc. The name must match the XDMF attribute tags')

# define a list of colormap names that matplotlib has, skipping over the reversed versions
colormaps=[str(m) for m in plt.cm.datad if not m.endswith("_r")]
colormaps.append('hot_desaturated') #because I add this colorbar below
colormaps.append('viridis')
colormaps=sorted(colormaps, key=lambda s: s.lower())

#continue with subparser argument creation:
settings_parser.add_argument(u'•cmap',choices=colormaps,default='hot_desaturated',help='Colormap to use for colorbar')#done
settings_parser.add_argument(u'•background_color',type=check_color,default='white',help='color to use as background')#done
settings_parser.add_argument(u'•text_color',type=check_color,default='black',help='color to use for text and annotations')
settings_parser.add_argument(u'•cbar_scale',type=str,default='lin',choices=['lin','log'],metavar="{{lin},log}",help='Linear or log scale colormap')
settings_parser.add_argument(u'•cbar_domain_min',type=check_float,metavar=("{{auto},min}"),default='auto',help='The min domain of the color bar')
settings_parser.add_argument(u'•cbar_domain_max',type=check_float,metavar=("{{auto},max}"),default='auto',help='The max domain of the color bar')
settings_parser.add_argument(u'•cbar_enabled',type=check_bool,choices=[True,False],metavar='{{True},False}',nargs=1,default=True,help='enable or disable colorbar')
settings_parser.add_argument(u'•cbar_location',type=str,choices=['left','right','top','bottom'],metavar='{\'left\',{\'right\'},\'top\',\'bottom\'}',default='right',help='set the colorbar position')
settings_parser.add_argument(u'•cbar_width',type=check_float,metavar='float',default='5.0',help='The width of the colorbar')
settings_parser.add_argument(u'•title',type=str,metavar='{{AttributeName},str}',help='Define the plot title that goes above the plot',default='AttributeName')
settings_parser.add_argument(u'•image_name',type=str,metavar='{{AttributeName},str}',help='Sets name of image',default='Image')
settings_parser.add_argument(u'•title_enabled',type=check_bool,choices=[True,False],metavar='{{True},False}',default=True,help='enable or disable the title that goes above the plot')
settings_parser.add_argument(u'•title_font',type=str,metavar='str',help='choose the font of the plot title')
settings_parser.add_argument(u'•title_font_size',type=check_int,default=18,metavar='int',help='font size for title')
settings_parser.add_argument(u'•label_font_size',type=check_int,metavar='int',help='font size for axis labels')
settings_parser.add_argument(u'•smooth_zones',type=check_bool,choices=[True,False],metavar='{True,{False}}',default=False,help='disable or enable zone smoothing')
settings_parser.add_argument(u'•image_format',type=str,choices=['png','svg','pdf','ps','jpeg','gif','tiff','eps'],default='png',metavar="{{'png'},'svg','pdf','ps','jpeg','gif','tiff','eps'}",help='specify graph output format')
settings_parser.add_argument(u'•image_size',type=check_int,nargs=2,metavar='int',default=[1280,710],help='specify the size of image')
settings_parser.add_argument(u'•x_range_km',type=check_float,nargs=2,metavar=("{{auto},min}","{{auto},max}"),default=['auto','auto'],help='The range of the x axis in km')
settings_parser.add_argument(u'•y_range_km',type=check_float,nargs=2,metavar=("{{auto},min}","{{auto},max}"),default=['auto','auto'],help='The range of the y axis in km')
settings_parser.add_argument(u'•x_range_label',type=str,metavar='str',default='X ($10^3$ km)',help='The text below the x axis')
settings_parser.add_argument(u'•y_range_label',type=str,metavar='str',default='Y ($10^3$ km)',help='The text to the left of the y axis')
settings_parser.add_argument(u'•time_format',type=str,metavar='{{seconds}, s, ms, milliseconds}',nargs=1,default='seconds',choices=["seconds", "s", "ms", "milliseconds"],help='Time format code to use for elapsed time and bounce time')
settings_parser.add_argument(u'•bounce_time_enabled',type=check_bool,choices=[True,False],metavar='{{True},False}',default=True,help='Boolean option for "time since bounce" display')
settings_parser.add_argument(u'•ctime_enabled',type=check_bool,choices=[True,False],metavar='{{True},False}',default=True,help='Boolean option for data creation time display')
settings_parser.add_argument(u'•elapsed_time_enabled',type=check_bool,choices=[True,False],metavar='{{True},False}',default=True,help='Boolean option for "elapsed time" display')
settings_parser.add_argument(u'•zoom_value',type=check_float,help='The zoom value (percentage of total range) to use if the x or y range is set to \'auto\'')
settings_parser.add_argument(u'•var_unit',type=str,default='auto',help='The unit to use for the plotted variable')
settings_parser.add_argument(u'•shock_enabled', type=check_bool,choices=[True,False],metavar='{True,{False}}',default=False,help='Displays the supernova schockwave')
settings_parser.add_argument(u'•shock_linestyle',type=str,default='solid',help='Sets the linestyle for the schock radius plot')
settings_parser.add_argument(u'•legend_enabled', type=check_bool,choices=[True,False],metavar='{True,{False}}',default=False,help='Displays the legend of the graph')
settings_parser.add_argument(u'•nse_c_contour', type=check_bool,choices=[True,False],metavar='{True,{False}}',default=False,help='Overlays the nse_c contour plot on the variable of interest plot')
settings_parser.add_argument(u'•shock_line_width',type=check_float,default=7.,metavar='float',help='Sets the line width of the shock radius plot')
settings_parser.add_argument(u'•shock_line_color',type=str,default='black',help='The line color of the shock radius plot')
settings_parser.add_argument(u'•nse_c_line_widths',type=check_float,default=4.,metavar='float',help='Sets the line width of the nse_c contour plot')
settings_parser.add_argument(u'•nse_cmap',type=str,choices=colormaps,default='binary',help='Colormap to use for nse_c contour plot')
settings_parser.add_argument(u'•nse_c_linestyles',type=str,default='solid',help='Sets the linestyle for the nse_c contour plot')
settings_parser.add_argument(u'•particle_overlay', type=check_bool,choices=[True,False],metavar='{True,{False}}',default=False,help='Overlays tracer particles on the plot')
settings_parser.add_argument(u'•particle_color',type=str,default='black',help='The dot color of the tracer particle plot')
settings_parser.add_argument(u'•particle_size',type=check_float,default=0.7,metavar='float',help='Sets the particle size of the tracer particle plot')
settings_parser.add_argument(u'•shock_contour_enabled', type=check_bool,choices=[True,False],metavar='{True,{False}}',default=False,help='Displays the supernova schockwave as a contour plot')
settings_parser.add_argument(u'•shock_contour_line_widths',type=check_float,default=4.,metavar='float',help='Sets the line width of the shock contour plot')
settings_parser.add_argument(u'•shock_contour_cmap',type=str,choices=colormaps,default='binary_r',help='Colormap to use for shock contour plot')
print_help()#print_help does a hacky help flag overload by intercepting the sys.argv before the parser in order to also print the help for the settings file
#if the help flag isn't there, continue and parse arguments as normal
args=parser.parse_args()

#display help for just the config parser if "help" or "h" appears after -s or --settings
if args.settingsfile and args.settingsfile in ['help','h']:
        settings_parser.print_help()
        sys.exit()

# Define parsed settings
argslist=[i[1:] for i in settings_parser.__dict__['_option_string_actions'].keys()] #generate list of valid arguments to prepend '•' to if it's not the first char
if args.settingsfile and args.settingsfile!='':
        settingsargs=[]
        for super_arg in csv.reader(open(args.settingsfile).read().split('\n'),delimiter=' ',quotechar='"',escapechar='\\'):
                if not filter(None,super_arg) or super_arg[0][0]+super_arg[0][1]=='//': #implement commenting and avoid empty lines
                        continue 
                for arg in filter(None,super_arg):
                        # account for the required '•' needed for argparse
                        if arg in argslist:
                                settingsargs.append(u'•'+arg) 
                        else:
                                settingsargs.append(arg)
        settings=settings_parser.parse_args(settingsargs)
del argslist

if args.tree or args.vars:
        args.quiet=True
#Robustly import an xml writer/parser for parseing the xdmf tree
try:
        from lxml import etree as et
        qprint("Running with lxml.etree")
except ImportError:
        try:
                # Python 2.5
                import xml.etree.cElementTree as et
                import xml.dom.minidom as md
                qprint("Running with cElementTree on Python 2.5+")
        except ImportError:
                try:
                # Python 2.5
                        import xml.etree.ElementTree as et
                        qprint("Running with ElementTree on Python 2.5+")
                except ImportError:
                        try:
                                # normal cElementTree install
                                import cElementTree as et
                                qprint("running with cElementTree")
                        except ImportError:
                                try:
                                        # normal ElementTree install
                                        import elementtree.ElementTree as et
                                        qprint("running with ElementTree")
                                except ImportError as e:
                                        eprint("Fatal error: Failed to import ElementTree from any known place. XML writing is impossible. ")
                                        traceback.print_exception(type(e),e,sys.exc_info()[2])
                                        sys.exit()
# Carefully import numpy for heavy number crunching
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
except ImportError as e:
        eprint('Fatal Error: numpy not found! (used to do math faster)')
        traceback.print_exception(type(e),e,sys.exc_info()[2])
        sys.exit()

#create a function to list all valid grid names in the xdmf:
def valid_grid_names():
        gridnames=[]
        for grd in domain.findall('Grid'): #grd is a grid element
                gridnames.append(grd.get('Name'))
        return gridnames

#create a function to list all valid variables for a given grid:
def valid_variables(gridname):
        variables=[]
        for attribute in domain.findall("*[@Name='"+gridname+"']/Attribute"):
                variables.append(attribute.attrib['Name'])
        return variables

#create a function to list all valid scalars from the currently parsed xdmf tree
def tree():
        print('Found valid scalars:')
        for grid in valid_grid_names():
                print(grid,end=':')
                variables=valid_variables(grid)
                for i,variable in enumerate(variables):
                        print(['\n\t  ',''][i!=0 or i==len(variables)]+variable+(', '+'\n\t  '*((i+1)%5==0))*(i!=len(variables)-1),end='')
                print()

# function to list full path to all valid scalars
def list_vars():
        for grid in valid_grid_names():
                for var in valid_variables(grid):
                        print(grid+'/'+var)

for file in args.files:
        domain=et.parse(file).getroot()[0]
        
        if args.tree:
                tree()
                sys.exit() #only does the first file for sanity sake
        if args.vars:
                list_vars()
                sys.exit()
        file_directory=''
        if re.search('.*\/(?!.+\/)',file):
                file_directory = re.search('.*\/(?!.+\/)',file).group()
        
        #overrides to make abundance behavior more permissive 
        if re.search('(?<=abundance/)([a-z]{1,2})/?(\d+)',settings.variable.lower()): #if there is an abundance followed by a proper element tag
                match=re.search('(?<=abundance/)([a-z]{1,2})/?(\d+)',settings.variable.lower()) #
                varname=match.group(2) #eg returns '3' from 'abundance/he/3' or 'abundance/he3'
                TrueVarname=match.group(1).title()+varname #eg returns 'He3' from 'abundance/he/3' or 'abundance/he3'
                gridname='Abundance/'+match.group(1).title() #eg returns 'Abundance/He' from 'abundance/he/3' or 'abundance/he3'
                TrueGridname='Abundance'
        else: #case that it is not an abundance variable
                match=settings.variable.split('/') 
                gridname='/'.join(match[:-1]) #[:-1] selects all but the last element, '/'.join() rejoins that collection with slashes
                varname=match[-1] #[-1] selects the last element
                TrueVarname=varname 
                TrueGridname=gridname
        del match
        # Note:
        # '(?!.*\/).*' is regex to find all the parts of a path prior to the file name
        if settings.image_name:
                image_name = settings.image_name+'_'+re.search('(?!.*\/).*',file).group()[:-4]+'.'+settings.image_format
        else:
                image_name = re.search('(?!.*\/).*',TrueVarname).group().title()+'_'+re.search('(?!.*\/).*',file).group()[:-4]+'.'+settings.image_format
        coordinates=[]
        def get_coordinates( ):
                expected_dim=grid.find('Topology').get('NumberOfElements').split()
                try:
                        assert len(expected_dim)==int(grid.find('Topology').get('TopologyType')[0])
                except:
                        eprint('Error: Dimensions specified in topology tag ('+int(grid.find('Topology').get('TopologyType')[0])+') do not match typology type ('+grid.find('Topology').get('TopologyType')+')')
                        sys.exit()
                for i,coord in enumerate(grid.find('Geometry').findall('DataItem')):
                        if coord.get('Dimensions'):
                                try:
                                        assert expected_dim[len(expected_dim)-1-i] == coord.get('Dimensions')
                                except: 
                                        eprint('Error: Dimensions specified in geometry\'s DataItem do not match those specified in the typology tag')
                                        sys.exit()
                        if coord.attrib['ItemType']=='Function':
                                divisor=int(re.search('(?<=\$\d\/)\d+',coord.attrib['Function']).group())
                                sub=coord.getchildren()[0]
                        else:
                                sub=coord
                        if sub.get('Dimensions'):
                                try:
                                        assert expected_dim[len(expected_dim)-1-i] == sub.get('Dimensions')
                                except: 
                                        eprint('Error: Dimensions specified in geometry\'s '+[str(i+1)+'th',['1st','2nd','3rd'][i%3]][i<=2]+' hyperslab tag\'s Dimension attribute do not match those specified in the typology tag')
                                        sys.exit()
                        ssc={'start':0,'stride':0,'count':0} #ssc = Start-Stride-Count
                        for j,k in enumerate(sub.getchildren()[0].text.split()):
                                ssc[['start','stride','count'][j]]=int(k)
                        try:
                                assert ssc['count']==int(expected_dim[len(expected_dim)-1-i])
                        except:
                                eprint('Error: Dimensions specified in geometry\'s '+[str(i+1)+'th',['1st','2nd','3rd'][i%3]][i<=2]+' hyperslab ('+sub.getchildren()[0].text+') do not match those specified in the typology tag')
                                sys.exit()
                        coord_dataitem=sub.getchildren()[1]
                        coordpath=coord_dataitem.text
                        if i==0:
                                global hf
                                global relative_path
                                relative_path=file_directory+coordpath.split(':')[0]
                                hf=h5py.File(relative_path,'r')
                        end=ssc['start']+ssc['count']*ssc['stride']
                        if coord.attrib['ItemType']=='Function':
                                coordinates.append(np.divide(hf[coordpath.split(':')[1]][ssc['start']:end:ssc['stride']],divisor))
                        else:
                                coordinates.append(hf[coordpath.split(':')[1]][ssc['start']:end:ssc['stride']])

        try:
                grid = domain.find('*[@Name="'+gridname+'"]')
                get_coordinates()
                zeniths=coordinates[0]
                azimuths=coordinates[1]
        except AttributeError:
                eprint('Error: Invalid grid')
                eprint('\t'+settings.variable+' provided a grid not found in the XDMF')
                eprint('\tGrid tried was: '+gridname)
                sys.exit()
        rad, phi = np.meshgrid(zeniths, azimuths)
        x,y=pol2cart(rad,phi)
        try:
                attrib_elements=domain.find("*[@Name='"+gridname+"']/*[@Name='"+varname+"']/").getchildren()
                datapath=attrib_elements[1].text.split(':')[1]
        except AttributeError:
                eprint("Error: Invalid attribute")
                eprint("\t"+settings.variable+" not found in "+file)
                eprint("\tPath looked for was: "+gridname+"/"+varname)
                sys.exit()
        if attrib_elements[0].get('Dimensions'):
                ssc=[]
                arrsize=int(attrib_elements[0].get('Dimensions').split()[1])
                for i in range(0,3):
                        a=[]
                        for j in range(0,arrsize):
                                a.append(int(attrib_elements[0].text.split()[j+arrsize*i]))
                        ssc.append(a)
        else:
                eprint('Error: Dimensions spec of dataitem in hyperslab invalid ')
                sys.exit()
        start=[]
        stride=[]
        end=[]
        for i,j,k in zip(ssc[0],ssc[1],ssc[2]): 
                start.append(i)
                stride.append(j)
                end.append(i+j*k)
        if re.search('xn_c',datapath): # in case it's abundance stuff and needs an extra dimension specified for the xn_c grid
                variable=hf[datapath][start[0]:end[0]:stride[0],start[1]:end[1]:stride[1],start[2]:end[2]:stride[2],start[3]:end[3]:stride[3]]
        else:
                variable=hf[datapath][start[0]:end[0]:stride[0],start[1]:end[1]:stride[1],start[2]:end[2]:stride[2]]
        variable=variable.squeeze() #remove dimensions of size 1 so the result is a 2d array

        # Get Creation time
        try:
                ctime='Data from '+time_lib.ctime(float(domain.find('Information[@Name="ctime"]').attrib['Value']))
        except KeyError:
                eprint('Could not find ctime')
        try:
                fun=domain.find("*[@Name='"+gridname+"']/Information[@Name='Time']").getchildren()[0] #more robustly get time
                if fun is not None:
                        try:
                                assert fun.attrib['ItemType']=='Function' and fun.attrib['Function']=='$0-$1' #I don't want to write a proper function parser because that's complex/meta
                                timepath=fun.getchildren()[0].text.rsplit(':')[1] #return eg: /mesh/time
                                bouncepath=fun.getchildren()[1].text.rsplit(':')[1] #return eg: /mesh/t_bounce
                                time_bounce=hf[timepath].value-hf[bouncepath].value
                                time_elapsed=hf[timepath].value
                        # below is an attempt to accept and interpret more general math expresiions from 'function' xdmf elementsn (currently disabled as it represents a secruity hazard)
                        # except AssertionError:
                        #       try:
                        #               from math import *
                        #               expression=re.sub(r'\$(\d*)',r'var[\1]',fun.attrib['Function'])
                        #               var=[]
                        #               for n in xrange(len(fun.getchildren())):
                        #                       var.append(hf[fun.getchildren()[n].text.rsplit(':')[1]].value)
                        #               br()
                        #               time_bounce=eval(expression)
                        #               time_elapsed=time_bounce
                        except:
                                eprint('Could not retrieve time from '+gridname)
                                eprint('        Time not formatted as known pattern')
                                sys.exit()
        except: # Get Time simply as fall back:
                try:
                        bounce_time=float(domain.find("*[@Name='"+gridname+"']/Time").attrib['Value'])
                        settings.elapsed_time_enabled=False
                except KeyError:
                        eprint('Static time not found!')

        fig = plt.figure(figsize=(12.1,7.2))
        fig.set_size_inches(12.1, 7.2,forward=True)
        sp=fig.add_subplot(111)

        if settings.shock_enabled:
                plt.subplot(111)
                try:
                        theta = np.array(hf['/mesh/y_ef'][:])
                        r = np.empty(theta.size)
                        r[0:theta.size-1] = np.array(hf['analysis/r_shock'][0][:])
                        r[-1] = r[-2]
                except KeyError as e:
                        eprint(e)
                        eprint('Invalid pathway to data in h5 file.')
                        sys.exit()
                for num, arr_val in enumerate(theta):
                        r[num], theta[num] = pol2cart(r[num], arr_val)
                plt.plot(r/1e5, theta/1e5, linestyle=settings.shock_linestyle, color=settings.shock_line_color,\
                         linewidth = settings.shock_line_width, label = "Shock Radius", zorder = 4)
        if settings.nse_c_contour:
                plt.subplot(111)
                try:
                        phi1 = np.array(hf['/mesh/y_ef'][:])
                        rho1 = np.array(hf['/mesh/x_ef'][:])
                        data = np.array(hf['abundance/nse_c'][:])
                except KeyError as e:
                        eprint(e)
                        eprint('Invalid pathway to data in h5 file.')
                        sys.exit()

                data = data.reshape(phi1.size-1,rho1.size)      #Takes the 1xXxY data set form the h5 file and transforms into a 2D matrix
                data2 = np.zeros((phi1.size, rho1.size))        #Initializes an array of zeros to be filled for the purpose of adding a row
                data2[0:phi1.size-1] = data                     #Takes the data and fills it into the previously initialized array
                data2[phi1.size-1]=data[phi1.size-2]            #Copies the last row of data into the last row of data2 to control for dimension mismatch

                rho1, phi1 = np.meshgrid(rho1, phi1)            
                var1, var2 = pol2cart(rho1, phi1)
                bounds = np.linspace(0,1,1)

                nse_c = plt.contour(var1/1e5, var2/1e5, data2, levels = bounds, cmap=settings.nse_cmap,\
                                    zorder = 3, linewidths = settings.nse_c_line_widths, linestyles=settings.nse_c_linestyles)

        if settings.legend_enabled:
                if settings.shock_enabled:
                        plt.legend()
                else:
                        qprint("No legend to print. The schock wave radius is not enabled")
        if settings.particle_overlay:
                plt.subplot(111)
                try:
                        px = np.array(hf['/particle/px'])
                        py = np.array(hf['/particle/py'])
                        #pz = np.array(h5file['/particle/pz'])
                except KeyError as e:
                        eprint('Particle data could no be found')
                        sys.exit()
                px, py = pol2cart(px, py)
                particles = plt.scatter(px/1e5, py/1e5, s = settings.particle_size, color = settings.particle_color, zorder = 5)
                
        if settings.shock_contour_enabled:
                plt.subplot(111)
                try:
                        rad = np.array(hf['/mesh/x_ef'][:])
                        tht = np.array(hf['/mesh/y_ef'][:])
                        f = np.array(hf['/fluid/shock'][:])
                except KeyError as e:
                        qprint("Shock data could not be found")
                        sys.exit()
                rad, tht = np.meshgrid(rad, tht)
                var_r, var_t = pol2cart(rad, tht)
                bds = np.linspace(0,1,2)
                plt.contour(var_r/1e5, var_t/1e5, f, cmap=settings.shock_contour_cmap, levels = bds, zorder = 5, linewidths = settings.shock_contour_line_widths)

        # # Setup mouse-over string to interrogate data interactively when in polar coordinates
        # def format_coord(x, y):
        #       return 'Theta=%1.4f, r=%9.4g, %s=%1.4f'%(x, y, 'entropy',entropy[max(0,np.where(azimuths<x)[0][-1]-1),max(0,np.where(zeniths<y)[0][-1]-1)])

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
        # Also create reversed version
        cdict_r={'red':cdict['red'][::-1],'green':cdict['green'][::-1],'blue':cdict['blue'][::-1]}
        hot_desaturated_r=LinearSegmentedColormap('hot_desaturated_r',cdict_r,N=256,gamma=1.0)

        del cdict,cdict_r
        if settings.cbar_scale=='log':
                norm=LogNorm()
        else:
                norm=None
        if settings.cbar_domain_min=='auto' and settings.cbar_domain_max=='auto':
                pcolor=sp.pcolormesh(x, y, variable,cmap=[hot_desaturated,settings.cmap][settings.cmap!='hot_desaturated'],norm=norm,antialiased=settings.smooth_zones)
        else:
                pcolor=sp.pcolormesh(x, y, variable,cmap=[hot_desaturated,settings.cmap][settings.cmap!='hot_desaturated'],norm=norm,vmin=settings.cbar_domain_min,vmax=settings.cbar_domain_max,antialiased=settings.smooth_zones)
        
        for atr in ['title','x_range_label','y_range_label']:
                settings.__setattr__(atr,re.sub(r'\\var(?=[^i]|$)',TrueVarname,settings.__getattribute__(atr)))
                settings.__setattr__(atr,re.sub(r'\\variable',TrueVarname.lower(),settings.__getattribute__(atr)))
                settings.__setattr__(atr,re.sub(r'\\Variable',TrueVarname.title(),settings.__getattribute__(atr)))
                settings.__setattr__(atr,re.sub(r'\\grid',TrueGridname,settings.__getattribute__(atr)))
                settings.__setattr__(atr,re.sub(r'\\path',TrueGridname+'/'+TrueVarname,settings.__getattribute__(atr)))
        title=sp.set_title(settings.title,fontsize=settings.title_font_size)
        xlabel=sp.set_xlabel(settings.x_range_label,fontsize=settings.label_font_size)
        ylabel=sp.set_ylabel(settings.y_range_label,fontsize=settings.label_font_size)

        if settings.cbar_enabled==True:
                cbar_orientation=['vertical','horizontal'][settings.cbar_location in ['top','bottom']]
                divider=make_axes_locatable(ax)
                
                cax=divider.append_axes(settings.cbar_location,\
                                                        size=str([settings.cbar_width,'5'][str(settings.cbar_width)=='auto'])+"%",\
                                                        pad=[[.8,.4][settings.cbar_location=='top'],[.1,.8][settings.cbar_location=='left']][(cbar_orientation=='vertical')])# note that this last setting, pad, is done in a sneaky way. True + True = 2 in python. ¯\_(ツ)_/¯

                cbar=plt.colorbar(pcolor,cax=cax,orientation=cbar_orientation)
                # settings for colorbar ticks positioning:
                if settings.cbar_location in ['top','bottom']:
                        cax.xaxis.set_ticks_position(settings.cbar_location)
                else:
                        cax.yaxis.set_ticks_position(settings.cbar_location)

        # fig.suptitle('this is the figure title', fontsize=12,)
        if settings.ctime_enabled:
                plt.figtext(.99,[.01,.965][settings.cbar_location=='bottom'],ctime,horizontalalignment='right',transform=ax.transAxes)# add following to see background: bbox=dict(facecolor='red', alpha=0.5)
        if settings.bounce_time_enabled:
                plt.figtext(.01,[.01,[.965,.975][settings.elapsed_time_enabled]][settings.cbar_location=='bottom'],'Bounce time: '+format(time_bounce,'.3'),horizontalalignment='left',transform=ax.transAxes)# add following to see background: bbox=dict(facecolor='red', alpha=0.5)
        if settings.elapsed_time_enabled:
                plt.figtext(.01,[[.01,.03][settings.bounce_time_enabled],[.965,.953][settings.bounce_time_enabled]][settings.cbar_location=='bottom'],'Elapsed time: '+format(time_elapsed,'.3'),horizontalalignment='left',transform=ax.transAxes)# add following to see background: bbox=dict(facecolor='red', alpha=0.5)
        plt.tight_layout()
        directory='.'
        if args.dir:
                directory=args.dir
        if directory[-1]=='/':
                directory=directory[:-1] # remove the last slash if it's there because we will add our own

        # Comment and uncomment the next line to save the image:
        plt.savefig(directory+'/'+image_name,format=settings.image_format,facecolor=settings.background_color,orientation='landscape') 
        qprint('time elapsed:   '+str(time_lib.time()-start_time))
        del start_time
                
        if args.debug:
                if platform.system()=='Darwin':
                        from subprocess import call # for on-the-fly lightning fast image viewing on mac
                        call(['qlmanage -p '+directory+'/'+image_name+' &> /dev/null'],shell=True) # for on-the-fly lightning fast image viewing on mac
                else:
                        plt.show() #Built in interactive viewer for non-macOS platforms. Slower.
