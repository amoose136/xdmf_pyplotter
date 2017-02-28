example_option_with_one_input ([default_setting], another setting)
(a,[b]) <- represents a single option that could be a or b in practice with b as default

cmap ([hot_desaturated],viridis,inferno,plasma,magma,Blues,BuGn,BuPu,GnBu,Greens,Greys,Oranges,OrRd,PuBu,PuBuGn,PuRd,Purples,RdPu,Reds,YlGn,YlGnBu,YlOrBr,YlOrRd,afmhot,autumn,bone,cool,copper,gist_heat,grey,hot,pink,spring,summer,winter,BrBG,bwr,coolwarm,PiYG,PYGn,PRGn,PuOr,Accent,Dark2,Paired,Pastel1,Pastel2,Set1,Set2,Set3,gist_earth,terrain,ocean,gist_stern,brg,nipy_spectral,jet,rainbow,gist_rainbow,hsv,flag,prism,gist_ncar,gnuplot,gnuplot2,CMRmap,cubehlix)
cbar_scale ([lin],log)
cbar_domain (int min, [auto]) (int max,[auto])
cbar_enabled ([True],False)
cbar_location (['right'],'left','top','bottom')
title ([portion of file name], string)
title_enabled ([True],False)
text_font (TBD)
title_font_size (int n)

smooth_zones ([True],False)

image_format ([png],svg,pdf,ps,jpeg,gif,tiff,eps)
image_size ([1280],int x) ([710],int y)

x_range_km ([auto],int min_in_km) ([auto],int max_in_km)
y_range_km ([auto],int min_in_km) ([auto],int max_in_km)
x_range_text [Radius (x10^3 km)]
y_range_text [Radius (x10^3 km)]
range_text_font_size (int n)
y_range_text_rotated (True,[False])

time_format ([seconds], s, ms, milliseconds)
bounce_time_enabled ([True],False)
elapsed_time_enabled ([True],False)
output_image_filename ([prefix_#####.png],str)
output_directory?
tick label scaling (bool)
variable