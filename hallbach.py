import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem

# create magnets
s1 = Box(mag=(0,0,-6), dim=(3,3,3), pos=(0,0,0))
s2 = Box(mag=(0,0,6), dim=(3,3,3),pos=(3,0,0))
s3 = Box(mag=(0,0,-6), dim=(3,3,3),pos=(6,0,0))
s4 = Box(mag=(0,0,6), dim=(3,3,3),pos=(9,0,0))
s5 = Box(mag=(0,0,-6), dim=(3,3,3),pos=(12,0,0))

# create collection
c = Collection(s1,s2,s3,s4,s5)

# manipulate magnets individually
#s1.rotate(45,(0,1,0), anchor=(0,0,0))
#s2.move((5,0,-4))

# manipulate collection
c.move((-6,0,0))

# calculate B-field on a grid
xdiv = 1000
zdiv = 1000
xs = np.linspace(-10,10,xdiv)
zs = np.linspace(-10,10,zdiv)
POS = np.array([(x,0,z) for z in zs for x in xs])
Bs = c.getB(POS).reshape(zdiv,xdiv,3)     #<--VECTORIZED

# create figure
fig = plt.figure(figsize=(9,5))
ax1 = fig.gca(projection='3d')  # 3D-axis
fig2 = plt.figure(figsize=(9,5))
ax2 = fig2.add_subplot(111)                   # 2D-axis

# display system geometry on ax1
displaySystem(c, subplotAx=ax1, suppress=True)

# display field in xz-plane using matplotlib
X,Z = np.meshgrid(xs,zs)

U,V = Bs[:,:,0], Bs[:,:,2]
#for i in range(xdiv):
#    for j in range(zdiv):
#        if(X[i,j]>-7.5 and X[i,j]<7.5 and Z[i,j]>-1.5 and Z[i,j]<1.5):
#            U[i,j] = 0
#            V[i,j] = 0

from mpl_toolkits.mplot3d.art3d import Poly3DCollection
colmap = plt.pcolor(X, Z, np.linalg.norm(Bs,axis=2),cmap=plt.cm.get_cmap('jet'))

#colmap = Poly3DCollection(colmap)
res = ax2.streamplot(X, Z, U, V, color='k')
lines = res.lines.get_paths()

#for i in range(xdiv):
#    for j in range(zdiv):
#        if(X[i,j]>-7.5 and X[i,j]<7.5 and Z[i,j]>-1.5 and Z[i,j]<1.5):
#           Bs[i,j,:]=Bs[i,j,:]

import matplotlib.cm as cm
m = cm.ScalarMappable(cmap=cm.jet)
m.set_array(np.linalg.norm(Bs,axis=2))
cmap=cm.get_cmap('jet')

#ax1.plot_surface(X,X/X+3,Z,facecolors=cmap(np.linalg.norm(Bs,axis=2)))
fig.colorbar(m)
for line in lines:
    old_x = line.vertices.T[0]
    old_y = line.vertices.T[1]
    new_z = old_y
    new_x = old_x
    new_y = old_x*0
    ax1.plot(new_x, new_y, new_z, 'k')
    

plt.show()