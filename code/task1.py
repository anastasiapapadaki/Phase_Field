from fipy import *
from scipy.integrate import simpson
import matplotlib.pyplot as plt


def Energies(phi,dx):
    # Energy densities
    fbulk = f0*( phi*(1-phi) )**2
    fgrad = 0.5*K*numerix.gradient(phi,dx,edge_order=2)**2
    
    # Integrate to obtain Energies
    Fb=simpson(fbulk,dx=dx)
    Fg=simpson(fgrad,dx=dx)
    Fint = Fb + Fg
    
    return fbulk,fgrad,Fb,Fg,Fint


#mesh
dx = 1.
Lx = 200.
nx = int(Lx/dx)
mesh = Grid1D(dx=dx, nx=nx)
x = mesh.cellCenters[0]

#time discretization:
dt = 1
steps = 50

#field varialbe
phi = CellVariable(name=r'$\phi(x)$',mesh=mesh, hasOld=1)

#initial condition
# phi0 = GaussianNoiseVariable(mesh=mesh, mean=0.5, variance=0.01, hasOld=0)
# phi.setValue(value = phi0.value)
phi.setValue(1.)
phi.setValue(0., where=x > Lx/2)

#boundary condtions
phi.faceValue.constrain(value=1., where=mesh.facesLeft)
phi.faceValue.constrain(value=0., where=mesh.facesRight)
# phi.faceGrad.constrain(value=0., where=mesh.facesLeft)
# phi.faceGrad.constrain(value=0., where=mesh.facesRight)

#equation coefficients
f0 = 1. ; L = 1. ; K=4. ; 

#equation
s1 = -( 2*(1-2*phi)**2 - 4*phi*(1-phi) )*f0
s0 = -( 2*phi*(1-phi)*(1-2*phi) )*f0 - s1*phi

eq = TransientTerm(coeff=1/L) == DiffusionTerm(coeff=K) \
        + ImplicitSourceTerm(coeff=s1)  + s0
            
#viewers
viewer = Viewer(vars=phi,xmin=0.,xmax=Lx,datamax=1.05,datamin=-0.05,\
                legend='upper right')
viewer.axes.set_xlabel("x",fontsize=14)
viewer.axes.set_ylabel("$\phi(x)$",fontsize=14)

# Energies
fbulk = CellVariable(name=r'$f_{bulk}$', mesh=mesh, hasOld=0)
fgrad = CellVariable(name=r'$f_{grad}$',mesh=mesh, hasOld=0)

Fint=numerix.zeros(steps+1) # array to store Fint(t)
# calculate initial energies:
fbulk.value,fgrad.value,Fb,Fg,Fint[0] = Energies(phi.value,dx)

#solver
for step in range(steps):
    print(step)
    phi.updateOld()
    res = 1.e5
    while res > 1.e-4:
        res = eq.sweep(var=phi, dt=dt)
        viewer.plot()
    
    fbulk.value,fgrad.value,Fb,Fg,Fint[step+1] = Energies(phi.value,dx)
  
#----------------------------------------
# Other plots

f = CellVariable(name=r'$f_{total}$',mesh=mesh, hasOld=0)     
f.value = fbulk.value + fgrad.value

viewer2 = Viewer(vars=(fbulk,fgrad,f),xmin=0.,xmax=Lx,datamin=0,\
                 datamax=1.05*max(f))
viewer2.axes.set_xlabel("x",fontsize=14)
viewer2.axes.set_ylabel("$f(x)$",fontsize=14)
viewer2.plot()

#--------------------------------------------------
plt.figure(3)
t = numerix.arange(0,(steps+1)*dt,dt)
plt.plot(t,Fint)
plt.xlabel("t",fontsize=14)
plt.ylabel("F",fontsize=14)
plt.xlim(0,steps*dt)
plt.show()
