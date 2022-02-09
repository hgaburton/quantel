#!/bin/bash

echo "material add Orbital
material change ambient Orbital 0.000000
material change specular Orbital 0.500000
material change diffuse Orbital 0.650000
material change shininess Orbital 0.534020
material change opacity Orbital 0.600000
# Display settings
display projection Orthographic
display nearclip set 0.000000
display depthcue off
axes location off

color Display {Background} white
color Element {H} gray
color Element {C} black
color Element {P} yellow
color Element {Cu} cyan
color Element {Cl} green
color Element {F} green2
color Type {C} black
color Surface {Grasp} red
color Labels {Springs} orange
color change rgb 19 0.00 1.00 0.22
color change rgb 0 0.0000 0.4471 0.8078
color change rgb 1 0.8353 0.0000 0.1961
color change rgb 2 0.4000 0.4000 0.4000
color change rgb 3 0.9098 0.4667 0.1333
# color change rgb 4 1.0 1.0 0.0
# color change rgb 5 0.5 0.5 0.20000000298
# color change rgb 6 0.600000023842 0.600000023842 0.600000023842
color change rgb 7 0.3922 0.6549 0.0431
# color change rgb 9 1.0 0.600000023842 0.600000023842
color change rgb 10 0.0000 0.6902 0.7255
# color change rgb 11 0.649999976158 0.0 0.649999976158
# color change rgb 12 0.5 0.899999976158 0.40000000596
# color change rgb 13 0.899999976158 0.40000000596 0.699999988079
# color change rgb 14 0.5 0.300000011921 0.0
# color change rgb 15 0.5 0.5 0.75

mol off all
mol new [lindex \$argv 0] type cube waitfor all
set sel [atomselect top all]
\$sel set radius 1.0
mol delrep 0 top
mol representation CPK 0.60000 0.30000 23.000000 21.000000
mol color Element
mol selection {all}
#mol selection {not hydrogen}
mol material Opaque
mol addrep top
mol selupdate 0 top 0
mol colupdate 0 top 0
mol smoothrep top 0 0

mol delrep 2 top
mol delrep 1 top
mol representation Isosurface [lindex \$argv 1] 0.000000 0.000000 0.000000
mol color ColorID $4
mol selection {all}
mol material Orbital
mol addrep top
mol representation Isosurface -[lindex \$argv 1] 0.000000 0.000000 0.000000
mol color ColorID $5
mol selection {all}
mol material Orbital
mol addrep top

scale by [lindex \$argv 2]

#render gs mol.ps
render POV3 [lindex \$argv 0].pov 
#render Tachyon [lindex \$argv 0].pov 

exit" > plot.tcl 

vmd -e plot.tcl -args $1 $2 $3
rm $1.png &> /dev/null
povray +W1024 +H1024 +FN +A +C -D +UA -I$1.pov -O$1.png
rm plot.tcl
