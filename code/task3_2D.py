from fipy import *
from scipy.integrate import simpson

# Model Constants
Vm = 1.e-5;  M = 1.e-17;  D=M*Vm**2
f0 = 9.989e7;  a=f0/Vm
Kc = 2.e-8
C1=0.16;  C2=0.23

#mesh
nx=ny=60
dx = dy = 2.e-10
mesh = Grid2D(nx=nx, ny=ny, dx=dx, dy=dy)

#time discretization:
dt = 1
steps = 1000

#field varialbe
C = CellVariable(name='C(x,y)',mesh=mesh, hasOld=1)
psi = CellVariable(mesh=mesh, hasOld=1)

#initial condition
C0 = GaussianNoiseVariable(mesh=mesh, mean=0.195, variance=0.002, hasOld=0)
C.setValue(value = C0.value)

#equations
dfdC = a* 2 * (C-C1) * (C2 - C) * (C1+C2 - 2*C)
d2fdC2 = a* ( 2*(C - C1)**2 - 8*(C - C1)*(C2 - C) + 2*(C2-C)**2 )

eq1 = (TransientTerm(var=C) == DiffusionTerm(coeff=D, var=psi))

eq2 = (ImplicitSourceTerm(coeff=1., var=psi) \
       == ImplicitSourceTerm(coeff=d2fdC2, var=C) - d2fdC2*C + dfdC \
           - DiffusionTerm(coeff=Kc, var=C))

eq = eq1 & eq2

            
#viewer
viewer = Viewer(vars=(C,), datamin=C1, datamax=C2)
viewer.axes.set_xlabel("x",fontsize=14)
viewer.axes.set_ylabel("y",fontsize=14)

#solver
for step in range(steps):
    print('step = ',step)
    C.updateOld()
    psi.updateOld()
    res = 1.e5
    while res > 1.e-4:
        res = eq.sweep(dt=dt)
        viewer.plot()
