import numpy as np
from mayavi import mlab

"""A very pretty picture of spherical harmonics translated from
the octaviz example."""
pi = np.pi
cos = np.cos
sin = np.sin
R = 150
r = 3

#phi = 3*np.pi/36
#beta_max = R*2*np.pi/r*np.tan(phi)
beta_max = 30*(2*np.pi)
beta = np.linspace(0.0*pi,beta_max,500)
gamma = np.linspace(0.0*pi,2.0*pi,500)

phi = np.arctan(beta_max*r/(R*2*np.pi))
print('网环捻角为', phi,'rad','(',phi*180/np.pi, '°)')
x0 = R*cos(gamma)
y0 = R*sin(gamma)
z0 = np.zeros_like(x0)

xi11 = 1*np.pi/3
x1 = (R + r * sin(xi11 + beta)) * cos(gamma)
y1 = (R + r * sin(xi11 + beta)) * sin(gamma)
z1 = r * cos(xi11 + beta)

xi12 = 2*np.pi/3
x2 = (R + r * sin(xi12 + beta)) * cos(gamma)
y2 = (R + r * sin(xi12 + beta)) * sin(gamma)
z2 = r * cos(xi12 + beta)

xi13 = 3*np.pi/3
x3 = (R + r * sin(xi13 + beta)) * cos(gamma)
y3 = (R + r * sin(xi13 + beta)) * sin(gamma)
z3 = r * cos(xi13 + beta)

xi14 = 4*np.pi/3
x4 = (R + r * sin(xi14 + beta)) * cos(gamma)
y4 = (R + r * sin(xi14 + beta)) * sin(gamma)
z4 = r * cos(xi14 + beta)

xi15 = 5*np.pi/3
x5 = (R + r * sin(xi15 + beta)) * cos(gamma)
y5 = (R + r * sin(xi15 + beta)) * sin(gamma)
z5 = r * cos(xi15 + beta)

xi16 = 6*np.pi/3
x6 = (R + r * sin(xi16 + beta)) * cos(gamma)
y6 = (R + r * sin(xi16 + beta)) * sin(gamma)
z6 = r * cos(xi16 + beta)

mlab.plot3d(x0, y0, z0,tube_radius=0.5*r, colormap='Spectral')
mlab.plot3d(x1, y1, z1,tube_radius=0.5*r, colormap='Spectral')
mlab.plot3d(x2, y2, z2,tube_radius=0.5*r, colormap='Spectral')
mlab.plot3d(x3, y3, z3,tube_radius=0.5*r, colormap='Spectral')
mlab.plot3d(x4, y4, z4,tube_radius=0.5*r, colormap='Spectral')
mlab.plot3d(x5, y5, z5,tube_radius=0.5*r, colormap='Spectral')
mlab.plot3d(x6, y6, z6,tube_radius=0.5*r, colormap='Spectral')

mlab.show()

