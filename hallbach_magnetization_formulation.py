import numpy as np
import matplotlib.pyplot as plt
from magpylib.source.magnet import Box,Cylinder
from magpylib import Collection, displaySystem,Sensor
import pygmo

nx = 5
ny = 5
nz = 3
sensmultip = 4
rhopow = 1

sens = np.zeros((nx*sensmultip,ny,1),dtype=Sensor)
for i in range(nx*sensmultip):
    for j in range(ny):
        for k in range(1):
            sens[i][j][k] = Sensor(pos=(i/sensmultip,j,k+0.65+(nz-1)))

def sign(val):
    return(val/abs(val))

def get_bounds():
        bound = ([-1]*(nx*ny*nz*3), [1]*(nx*ny*nz*3))
        return (bound)

def fitness(x):
    x = np.array(x)
    def Create_Magnet(x):
        boxes = np.zeros((nx,ny,nz),dtype=Box)
        # create collection
        nNodes = nx*ny*nz
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    magdirx = x[i*ny*nz+j*nz+k]**rhopow
                    magdiry = x[nNodes+i*ny*nz+j*nz+k]**rhopow
                    magdirz = x[nNodes*2+i*ny*nz+j*nz+k]**rhopow
                    mags = [magdirx,magdiry,magdirz]
                    # xmag = thesign*(np.argmax(mags)==0)
                    # ymag = thesign*(np.argmax(mags)==1)
                    # zmag = thesign*(np.argmax(mags)==2)
                    xmag = (magdirx)
                    ymag = (magdiry)
                    zmag = (magdirz)
                    boxes[i][j][k] = Box(mag=(xmag,ymag,zmag), dim=(1,1,1), pos=(i,j,k))
        return boxes
    lb = get_bounds()[0]
    ub = get_bounds()[1]
    penalty=sum((x<lb) * (-x+lb)) + sum( (x>ub)*(x-ub) ) # penalty due to box constraint
    nNodes = nx*ny*nz
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                magdirx = x[i*ny*nz+j*nz+k]**rhopow
                magdiry = x[nNodes+i*ny*nz+j*nz+k]**rhopow
                magdirz = x[nNodes*2+i*ny*nz+j*nz+k]**rhopow
                summag = np.linalg.norm([magdirx, magdiry , magdirz],ord=2)
                if(summag>1):
                    penalty=penalty+summag-1
    if(penalty==0):
        boxes = Create_Magnet(x)
        magCollection = Collection()
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    magCollection= Collection(magCollection,boxes[i][j][k])
        mag_field_str = 0
        for i in range(nx*sensmultip):
            for j in range(ny):
                for k in range(1):
                    #print(sens[i][j][k].getB(magCollection))
                    mag_field_str = mag_field_str + np.linalg.norm(sens[i][j][k].getB(magCollection),ord=2)
    else:
        mag_field_str=-penalty
    print("f")
    print(-mag_field_str+(1.0-summag))
    return (-mag_field_str+(1.0-summag))

def strictfitness(x):
    x = np.array(x)
    def Create_Magnet(x):
        boxes = np.zeros((nx,ny,nz),dtype=Box)
        # create collection
        nNodes = nx*ny*nz
        for k in range(nz):
            strdat = ''
            for i in range(nx):
                for j in range(ny):
                    magdirx = x[i*ny*nz+j*nz+k]**rhopow
                    magdiry = x[nNodes+i*ny*nz+j*nz+k]**rhopow
                    magdirz = x[nNodes*2+i*ny*nz+j*nz+k]**rhopow
                    mags = [abs(magdirx),abs(magdiry),abs(magdirz)]
                    if(np.argmax(mags)==0):
                        thesign = -1*(magdirx<0) + (magdirx>=0)
                    if(np.argmax(mags)==1):
                        thesign = -1*(magdiry<0) + (magdiry>=0)
                    if(np.argmax(mags)==2):
                        thesign = -1*(magdirz<0) + (magdirz>=0)
                    xmag = thesign*(np.argmax(mags)==0)
                    ymag = thesign*(np.argmax(mags)==1)
                    zmag = thesign*(np.argmax(mags)==2)
                    signstr=''
                    if(thesign==-1)
                        signstr='-'
                    if(np.argmax(mags)+1)==1)
                        axstr='x'    
                    if(np.argmax(mags)+1)==2)
                        axstr='y'    
                    if(np.argmax(mags)+1)==3)
                        axstr='z'    
                    strdat = strdat+signstr+axstr+'  '
                    # xmag = (magdirx)
                    # ymag = (magdiry)
                    # zmag = (magdirz)
                    boxes[i][j][k] = Box(mag=(xmag,ymag,zmag), dim=(1,1,1), pos=(i,j,k))
        return boxes
    lb = get_bounds()[0]
    ub = get_bounds()[1]
    penalty=sum((x<lb) * (-x+lb)) + sum( (x>ub)*(x-ub) ) # penalty due to box constraint
    nNodes = nx*ny*nz
    print(nNodes)
    if(penalty==0):
        boxes = Create_Magnet(x)
        magCollection = Collection()
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    magCollection= Collection(magCollection,boxes[i][j][k])
        mag_field_str = 0
        for i in range(nx*sensmultip):
            for j in range(ny):
                for k in range(1):
                    #print(sens[i][j][k].getB(magCollection))
                    mag_field_str = mag_field_str + np.linalg.norm(sens[i][j][k].getB(magCollection),ord=2)
    else:
        mag_field_str=-penalty
    print(-mag_field_str)
    return [-mag_field_str]


## optimization
# define objective function
class hallbach_test:
    def __init__(self, sensors):
        self.sensors = sensors
    def fitness(self,x):
        x = np.array(x)
        #print(x)
        def Create_Magnet(x):
            boxes = np.zeros((nx,ny,nz),dtype=Box)
            # create collection
            nNodes = nx*ny*nz
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        magdirx = x[i*ny*nz+j*nz+k]**rhopow
                        magdiry = x[nNodes+i*ny*nz+j*nz+k]**rhopow
                        magdirz = x[nNodes*2+i*ny*nz+j*nz+k]**rhopow
                        mags = [magdirx,magdiry,magdirz]
                        # xmag = thesign*(np.argmax(mags)==0)
                        # ymag = thesign*(np.argmax(mags)==1)
                        # zmag = thesign*(np.argmax(mags)==2)
                        xmag = (magdirx)
                        ymag = (magdiry)
                        zmag = (magdirz)
                        boxes[i][j][k] = Box(mag=(xmag,ymag,zmag), dim=(1,1,1), pos=(i,j,k))
            return boxes
        lb = get_bounds()[0]
        ub = get_bounds()[1]
        penalty=sum((x<lb) * (-x+lb)) + sum( (x>ub)*(x-ub) ) # penalty due to box constraint
        nNodes = nx*ny*nz
        for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        magdirx = x[i*ny*nz+j*nz+k]**rhopow
                        magdiry = x[nNodes+i*ny*nz+j*nz+k]**rhopow
                        magdirz = x[nNodes*2+i*ny*nz+j*nz+k]**rhopow
                        summag = np.linalg.norm([magdirx, magdiry , magdirz],ord=2)
                        if(summag>1):
                            penalty=penalty+summag-1
        if(penalty==0):
            boxes = Create_Magnet(x)
            magCollection = Collection()
            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        magCollection= Collection(magCollection,boxes[i][j][k])
            mag_field_str = 0
            for i in range(nx*sensmultip):
                for j in range(ny):
                    for k in range(1):
                        #print(sens[i][j][k].getB(magCollection))
                        mag_field_str = mag_field_str + np.linalg.norm(sens[i][j][k].getB(magCollection),ord=2)
        else:
            mag_field_str=-penalty
        print(-mag_field_str+(1-summag))
        return [-mag_field_str+(1-summag)]
    def get_bounds(self):
        bound = ([-1]*(nx*ny*nz*3), [1]*(nx*ny*nz*3)) # rhox, rhoy, rhoz
        return (bound)
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
# for i in range(3):
#     algo = pygmo.algorithm(pygmo.cmaes(gen = 500, sigma0=0.7))
#     prob = pygmo.problem(hallbach_test(sensors=sens))
#     pop = pygmo.population(prob,size=30)
#     if(i==0):
#         all_up = np.array(object=[0]*(nx*ny*nz*3),dtype=float)
#         all_up[(nx*ny*nz*2):(nx*ny*nz*3)] = 0.9
#         pop.set_x(1,all_up )
#     if(i>0):
#         pop.set_x(1,bestpop )
#     pop = algo.evolve(pop)
#     bestpop = pop.champion_x

import scipy.optimize as sc
sc.Bounds([-1.1]*(nx*ny*nz*3),[1.1]*(nx*ny*nz*3))
x = sc.minimize(fun=fitness,x0=[0.9]*(nx*ny*nz*3),method="COBYLA",
options={'maxiter':2500})

print('best pop')
# print((pop.champion_x))
# fitness(pop.champion_x)
print(x)

print('all up')
all_up = np.array([0]*(nx*ny*nz*3))
all_up[(nx*ny*nz*2):(nx*ny*nz*3)] = 1 
fitness(all_up)

# print('halbach')
# halbacharray = [1,0,-1,0,1,0,0,0,0,0,0,1,0,-1,0]
# fitness(halbacharray)

# halbacharray2 = [0,-1,0,1,0,0,0,0,0,0,1,0,-1,0,1]
# fitness(halbacharray2)
