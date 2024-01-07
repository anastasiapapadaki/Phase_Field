import matplotlib.pyplot as plt
from fipy import *
from scipy.integrate import simpson

def Energies(C,dx):
    # Energy densities
    fbulk = a*( (C-C1)*(C2-C)  )**2
    fgrad = 0.5*Kc*numerix.gradient(C,dx,edge_order=2)**2
    
    # Integrate to obtain Energies
    Fb=simpson(fbulk,dx=dx)
    Fg=simpson(fgrad,dx=dx)
    Fint = Fb + Fg
    
    return Fb,Fg,Fint


# Model Constants
Vm = 1.e-5;  M = 1.e-17;  D=M*Vm**2
f0 = 9.989e7;  a=f0/Vm
Kc = 5.e-8
C1=0.16;  C2=0.23

#mesh
dx = 1e-9 #5.e-10
nx=300
Lx=nx*dx
mesh = Grid1D(dx=dx, nx=nx)
x = mesh.cellCenters[0]

#time discretization:
dt = 1
steps = 30

#field varialbe
C = CellVariable(name=r'$C(x)$',mesh=mesh, hasOld=1)
psi = CellVariable(mesh=mesh, hasOld=1)

#initial condition
# C0 = GaussianNoiseVariable(mesh=mesh, mean=0.195, variance=0.001, hasOld=0)
# C.setValue(value = C0.value)
C.setValue(C1)
C.setValue(C2, where=x > Lx/2)

#boundary conditions
C.faceValue.constrain(value=C1, where=mesh.facesLeft)
C.faceValue.constrain(value=C2, where=mesh.facesRight)
psi.faceGrad.constrain(value=0., where=mesh.facesLeft)
psi.faceGrad.constrain(value=0., where=mesh.facesRight)

#equations
dfdC = a* 2 * (C-C1) * (C2 - C) * (C1+C2 - 2*C)
d2fdC2 = a* ( 2*(C - C1)**2 - 8*(C - C1)*(C2 - C) + 2*(C2-C)**2 )

eq1 = (TransientTerm(var=C) == DiffusionTerm(coeff=D, var=psi))

eq2 = (ImplicitSourceTerm(coeff=1., var=psi) \
       == ImplicitSourceTerm(coeff=d2fdC2, var=C) - d2fdC2*C + dfdC \
           - DiffusionTerm(coeff=Kc, var=C))

eq = eq1 & eq2

            
#viewer
viewer = Viewer(vars=C,xmin=0,xmax=Lx,datamax=1.01*C2,datamin=0.99*C1,\
                legend='upper left')
viewer.axes.set_xlabel("x",fontsize=14)
viewer.axes.set_ylabel("$C$",fontsize=14)

#solver
for step in range(steps):
    print('step = ',step)
    C.updateOld()
    psi.updateOld()
    res = 1.e5
    while res > 1.e-4:
        res = eq.sweep(dt=dt)
        viewer.plot()

Fb,Fg,Fint = Energies(C.value,dx)
print('F_bulk =',Fb)
print('F_grad =',Fg)
print('F_inte =',Fint)


# Calculate interface thickness
tol = 0.0005
zone = numerix.logical_and(abs(C2-C.value)>tol, abs(C1-C.value)>tol)
Cmin = min(C.value[zone]); Cmax = max(C.value[zone])
Xmin = min(x.value[zone]); Xmax = max(x.value[zone])
print('interface with is (in nm)',(Xmax-Xmin)*1e9)

#viewer.axes.plot(x.value,C.value,'r*')
viewer.axes.plot([Xmin,Xmax],[Cmin,Cmax],'ro')

plt.savefig("test.png",dpi=300)
plt.show()

