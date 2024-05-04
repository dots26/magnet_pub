import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem,Sensor
import pygmo

nx = 5
ny = 1
nz = 1

sens = np.zeros((nx*2,ny*2,1),dtype=Sensor)
for i in range(nx*2):
    for j in range(ny*2):
        for k in range(1):
            sens[i][j][k] = Sensor(pos=(i/2,j/2,k+0.75))

def sign(val):
    return(val/abs(val))

def get_bounds():
        dir_bound = ([1]*(nx*ny*nz), [3]*(nx*ny*nz))
        sign_bound = ([0]*(nx*ny*nz), [1]*(nx*ny*nz))
        lb = np.reshape([dir_bound[0],sign_bound[0]],2*nx*ny*nz)
        ub = np.reshape([dir_bound[1],sign_bound[1]],2*nx*ny*nz)
        return (lb, ub)

def fitness(x):
    x = np.array(x)
    def Create_Magnet(x):
        boxes = np.zeros((nx,ny,nz),dtype=Box)
        # create collection
        nNodes = nx*ny*nz
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    magdir = x[i*ny*nz+j*nz+k]
                    thesign = x[nNodes+i*ny*nz+j*nz+k]
                    if(round(thesign)==0):
                        thesign = -1
                    xmag = thesign*(round(magdir)==1)
                    ymag = thesign*(round(magdir)==2)
                    zmag = thesign*(round(magdir)==3)
                    boxes[i][j][k] = Box(mag=(xmag,ymag,zmag), dim=(1,1,1), pos=(i,j,k))
        return boxes
    lb = get_bounds()[0]
    ub = get_bounds()[1]
    penalty=sum((x<lb) * (-x+lb)) + sum( (x>ub)*(x-ub) ) # penalty due to box constraint
    if(penalty==0):
        boxes = Create_Magnet(x)
        magCollection = Collection()
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    magCollection= Collection(magCollection,boxes[i][j][k])
        mag_field_str = 0
        for i in range(nx*2):
            for j in range(ny*2):
                for k in range(1):
                    mag_field_str = mag_field_str + np.linalg.norm(sens[i][j][k].getB(magCollection),ord=2)
    else:
        mag_field_str=-penalty
    print(-mag_field_str)
    return [-mag_field_str]

## optimization
# define objective function
class hallbach_test:
    def __init__(self, sensors,nx,ny,nz):
        self.sensors = sensors
        self.nx = nx
        self.ny = ny
        self.nz = nz
    def fitness(self,x):
        x = np.array(x)
        def Create_Magnet(x):
            nx = self.nx
            ny = self.ny
            nz = self.nz
            boxes = np.zeros((nx,ny,nz),dtype=Box)
            # create collection
            nNodes = nx*ny*nz
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        magdir = x[i*ny*nz+j*nz+k]
                        thesign = x[nNodes+i*ny*nz+j*nz+k]
                        if(round(thesign)==0):
                            thesign = -1
                        xmag = thesign*(round(magdir)==1)
                        ymag = thesign*(round(magdir)==2)
                        zmag = thesign*(round(magdir)==3)
                        boxes[i][j][k] = Box(mag=(xmag,ymag,zmag), dim=(1,1,1), pos=(i,j,k))
            return boxes
        lb = self.get_bounds()[0]
        ub = self.get_bounds()[1]
        penalty=sum((x<lb) * (-x+lb)) + sum( (x>ub)*(x-ub) ) # penalty due to box constraint
        if(penalty==0):
            boxes = Create_Magnet(x)
            magCollection = Collection()
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        magCollection= Collection(magCollection,boxes[i][j][k])
            mag_field_str = 0
            for i in range(nx*2):
                for j in range(ny*2):
                    for k in range(1):
                        mag_field_str = mag_field_str + np.linalg.norm(sens[i][j][k].getB(magCollection),ord=2)
        else:
            mag_field_str=-penalty
        return [-mag_field_str]
    def get_bounds(self):
        nx = self.nx
        ny = self.ny
        nz = self.nz
        dir_bound = ([1]*(nx*ny*nz), [3]*(nx*ny*nz))
        sign_bound = ([0]*(nx*ny*nz), [1]*(nx*ny*nz))
        lb = np.reshape([dir_bound[0],sign_bound[0]],2*nx*ny*nz)
        ub = np.reshape([dir_bound[1],sign_bound[1]],2*nx*ny*nz)
        return (lb,ub)
    def get_name(self):
        return "Magnetic Field around Hallbach"


class sphere_fun:
    def __init__(self, dim):
        self.dim = dim
    def fitness(self, x):
        return [sum(x*x)]
    def get_bounds(self):
        return ([-1] * self.dim, [1] * self.dim)
    def get_name(self):
        return "Sphere Function"
    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)


# search space is nx*ny*nz = 32 dimensional, ranged from 0 to 5 in each dimension
algo = pygmo.algorithm(pygmo.cmaes(gen = 2000, sigma0=0.2))
prob = pygmo.problem(hallbach_test(sens,nx,ny,nz))
pop = pygmo.population(prob,size=10)

pop = algo.evolve(pop)
print('best pop')
print((pop.champion_x))
fitness(pop.champion_x)

print('all up')
fitness([3,3,3,3,3,1,1,1,1,1]) # halbach array

print('alternating updown')
fitness([3,3,3,3,3,1,0,1,0,1]) # halbach array

print('halbach array')
fitness([3,1,3,1,3,0,1,1,0,0]) # halbach array
# ## visualization
# # calculate B-field on a grid
# xdiv = 300
# zdiv = 300
# xs = np.linspace(-1,5,xdiv)
# zs = np.linspace(-2,6,zdiv)
# POS = np.array([(x,0,z) for z in zs for x in xs])
# Bs = c.getB(POS).reshape(zdiv,xdiv,3)     #<--VECTORIZED

# # create final figure
# fig = plt.figure(figsize=(9,5))
# ax1 = fig.gca(projection='3d')  # 3D-axis
# fig2 = plt.figure(figsize=(9,5))
# ax2 = fig2.add_subplot(111)                   # 2D-axis

# # display system geometry on ax1
# displaySystem(c, subplotAx=ax1, suppress=True)

# # display field in xz-plane using matplotlib
# X,Z = np.meshgrid(xs,zs)

# U,V = Bs[:,:,0], Bs[:,:,2]
# for i in range(xdiv):
#     for j in range(zdiv):
#         if(X[i,j]>-7.5 and X[i,j]<7.5 and Z[i,j]>-1.5 and Z[i,j]<1.5):
#             U[i,j] = 0
#             V[i,j] = 0

# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
# colmap = plt.pcolor(X, Z, np.linalg.norm(Bs,axis=2),cmap=plt.cm.get_cmap('jet'))

# #colmap = Poly3DCollection(colmap)
# res = ax2.streamplot(X, Z, U, V, color='k')
# lines = res.lines.get_paths()

# #for i in range(xdiv):
# #    for j in range(zdiv):
# #        if(X[i,j]>-7.5 and X[i,j]<7.5 and Z[i,j]>-1.5 and Z[i,j]<1.5):
# #           Bs[i,j,:]=Bs[i,j,:]

# import matplotlib.cm as cm
# m = cm.ScalarMappable(cmap=cm.jet)
# m.set_array(np.linalg.norm(Bs,axis=2))
# cmap=cm.get_cmap('jet')

# ax1.plot_surface(X,X/X+3,Z,facecolors=cmap(np.linalg.norm(Bs,axis=2)))
# fig.colorbar(m)
# for line in lines:
#     old_x = line.vertices.T[0]
#     old_y = line.vertices.T[1]
#     new_z = old_y
#     new_x = old_x
#     new_y = old_x*0
#     ax1.plot(new_x, new_y, new_z, 'k')
    
# #ax2.streamplot(X, Z, U, V, color='k')

# plt.show()


# halbach: fitness([3,5,2,0,3])
# halbach: fitness([2,0,3,5,2])