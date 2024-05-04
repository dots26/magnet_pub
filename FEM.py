from fenics import *
from dolfin import *
from mshr import *
from math import sin, cos, pi, hypot
import matplotlib.pyplot as plt



# try:
#     from pyadjoint import ipopt  # noqa: F401
# except ImportError:
#     print("""This example depends on IPOPT and Python ipopt bindings. \
#   When compiling IPOPT, make sure to link against HSL, as it \
#   is a necessity for practical problems.""")
#     raise

# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

V = Constant(0.4)  # volume bound on the control
up = Constant([0,1]) # initial direction upwards
p = Constant(3)  # power used in the solid isotropic material
# with penalisation (SIMP) rule, to encourage the control
# solution to attain either 0 or 1
eps = Constant(1.0e-3)  # epsilon used in the solid isotropic material
alpha = Constant(1.0e-8)  # regularisation coefficient in functional


def k(a):
    """Solid isotropic material with penalisation (SIMP) conductivity
  rule, equation (11)."""
    return eps + (1 - eps) * a ** p

innerrad = 1.0   # inner radius of iron cylinder
outerrad = 2.0  # outer radius of iron cylinder
c_1 = 1 # radius for inner circle of copper wires
c_2 = 2 # radius for outer circle of copper wires
r = 0.1   # radius of copper wires
R = 5.0   # radius of domain
n = 10    # number of windings

domain = Circle(Point(0, 0), R)
cylinder = Circle(Point(0, 0), outerrad) - Circle(Point(0, 0), innerrad)
domain.set_subdomain(1, cylinder)

n = 250
#mesh = UnitSquareMesh(n, n)
mesh = generate_mesh(domain, 150)

A = VectorFunctionSpace(mesh, "Lagrange", 1)  # function space for control
P = FunctionSpace(mesh, "Lagrange", 1)  # function space for solution

center = Point(0, 0)
class Cyl_boundary(SubDomain):
    def inside(self, x, on_boundary): 
        center_dist = hypot(x[0]-center[0], x[1]-center[1]) 
        return (near(center_dist,innerrad,1e-3) or near(center_dist,outerrad,1e-3)) #and on_boundary

class Cyl_inside(SubDomain):
    def inside(self, x, on_boundary): 
        center_dist = hypot(x[0]-center[0], x[1]-center[1]) 
        return (center_dist>innerrad and center_dist<outerrad)

class Exterior(SubDomain):
    def inside(self, x, on_boundary):
        center_dist = hypot(x[0]-center[0], x[1]-center[1]) 
        return center_dist>outerrad

# the Dirichlet BC; the Neumann BC will be implemented implicitly by
# dropping the surface integral after integration by parts
exterior = Exterior()
cyl_bound = Cyl_boundary()
cyl_bound2 = Cyl_inside()
bc1 = DirichletBC(P, 0.0, cyl_bound)
bc2 = DirichletBC(P, 0.0, exterior)
bcs = []#bc1]#,bc2]

# f = interpolate(Constant(1.0e-2), P)  # the volume source term for the PDE
sub_domains = MeshFunction( 'size_t', mesh,0)
cyl_bound.mark(sub_domains, 3)
cyl_bound2.mark(sub_domains, 2)
exterior.mark(sub_domains, 1)
ds = Measure('ds', domain=mesh, subdomain_data=sub_domains)

file = File("subdomains.pvd")
file << sub_domains

#for x in mesh.coordinates():
#   if cyl_bound.inside(x, True): print('%s is on x = cylinder boundary' % x)
    #if exterior.inside(x, True): print('%s is on x = exterior ' % x)

# V = FunctionSpace(mesh, "CG", 1)
# u = Function(V)
# bc = DirichletBC(V, Constant(1), cyl_bound)
# bc.apply(u.vector())
# plot(u)
# plt.show()



def forward(m):
    """Solve the forward problem for a given magnetization m(x)."""
    #u = Function(P, name="mag_scalar_pot")
    u = TrialFunction(P)
    v = TestFunction(P)
    n = FacetNormal(mesh)
    F = inner(grad(v),grad(u)) * dx 

    rhom = -1*div(m)
    sigmam = assemble(inner(m, n)*ds(2))    
    u = Function(P, name="mag_scalar_pot")
    L = rhom*v*dx - sigmam*v*ds(2)
    solve(F == L, u,bcs)#, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7,
                                                             # "maximum_iterations": 20}})  
    plot(inner(u,u))
    plt.show()
    return u


if __name__ == "__main__":
    v = Expression(('0','(x[0]*x[0]+x[1]*x[1]<4 && x[0]*x[0]+x[1]*x[1]>1)'), degree = 1)
    a = interpolate(v, A)  # initial guess.
    u = forward(a)  # solve the forward problem once.


V_g = VectorFunctionSpace(mesh, 'Lagrange', 1)
v = TestFunction(V_g)
w = TrialFunction(V_g)

agrad = inner(-grad(u), v)*dx
Lgrad = inner(w, v)*dx
grad_u = Function(V_g)
solve(Lgrad==agrad, grad_u)
grad_u.rename('grad(u)', 'continuous gradient field')

vtkfile_B = File('field2.pvd')
vtkfile_B << u
vtkfile_B << grad_u

lb = [-1.0, -1.0]
ub = [1.0, 1.0]