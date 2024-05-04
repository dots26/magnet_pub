from fenics import *

from fenics_adjoint import *
import matplotlib.pyplot as plt


# turn off redundant output in parallel
parameters["std_out_all_processes"] = False

V = Constant(0.4)  # volume bound on the control
p = Constant(5)  # power used in the solid isotropic material
# with penalisation (SIMP) rule, to encourage the control
# solution to attain either 0 or 1
eps = Constant(1.0e-3)  # epsilon used in the solid isotropic material
alpha = Constant(1.0e-8)  # regularisation coefficient in functional


def k(a):
    """Solid isotropic material with penalisation (SIMP) conductivity
  rule, equation (11)."""
    return eps + (1 - eps) * a ** p
n = 250
mesh = UnitSquareMesh(n, n)
A = FunctionSpace(mesh, "CG", 1)  # function space for control
P = FunctionSpace(mesh, "CG", 1)  # function space for solution

class WestNorth(SubDomain):
    """The top and left boundary of the unitsquare, used to enforce the Dirichlet boundary condition."""

    def inside(self, x, on_boundary):
        return (x[0] == 0.0 or x[1] == 1.0) and on_boundary


# the Dirichlet BC; the Neumann BC will be implemented implicitly by
# dropping the surface integral after integration by parts
bc = [DirichletBC(P, 0.0, WestNorth())]
f = interpolate(Constant(1.0e-2), P)  # the volume source term for the PDE

def forward(a):
    """Solve the forward problem for a given material distribution a(x)."""
    T = Function(P, name="Temperature")
    v = TestFunction(P)

    F = inner(grad(v), k(a) * grad(T)) * dx - f * v * dx
    solve(F == 0, T, bc, solver_parameters={"newton_solver": {"absolute_tolerance": 1.0e-7,
                                                              "maximum_iterations": 20}})
    plot(T)
    plt.show()
    return T

if __name__ == "__main__":
    a = interpolate(V, A)  # initial guess.
    T = forward(a)  # solve the forward problem once.
controls = File("output/control_iterations.pvd")
a_viz = Function(A, name="ControlVisualisation")


def eval_cb(j, a):
    a_viz.assign(a)
    controls << a_viz

J = assemble(f * T * dx + alpha * inner(grad(a), grad(a)) * dx)
m = Control(a)
Jhat = ReducedFunctional(J, m, eval_cb_post=eval_cb)  # define J and dJ/du using dolfin ##!!

lb = 0.0
ub = 1.0
class VolumeConstraint(InequalityConstraint):
    """A class that enforces the volume constraint g(a) = V - a*dx >= 0."""

    def __init__(self, V):
        self.V = float(V)

        self.smass = assemble(TestFunction(A) * Constant(1) * dx)
        self.tmpvec = Function(A)

    def function(self, m):
        from pyadjoint.reduced_functional_numpy import set_local
        set_local(self.tmpvec, m)
        integral = self.smass.inner(self.tmpvec.vector())
        if MPI.rank(MPI.comm_world) == 0:
            print("Current control integral: ", integral)
        return [self.V - integral]

    def jacobian(self, m):
        return [-self.smass]

    def output_workspace(self):
        return [0.0]

    def length(self):
        """Return the number of components in the constraint vector (here, one)."""
        return 1