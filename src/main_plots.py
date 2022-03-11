
#%% -----------------------------------------------------------------------------------------------    

import matplotlib.pyplot as plt
import numpy as np


def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


#%% --------------------- 3D Cooke's membrane plot with cross sections ---------------------------
fig = plt.figure()
ax = plt.axes(projection='3d', adjustable='box')

# plotting the contour lines
ax.plot([0,0], [0,0], [0,0.733], c = 'black')
ax.plot([0,0], [1,1], [0,0.733], c = 'black')
ax.plot([0,0], [0,1], [0,0], c = 'black')
ax.plot([0,0], [0,1], [0.733,0.733], c = 'black')


ax.plot([0.8,0.8], [0,0], [0.733,1], c = 'black')
ax.plot([0.8,0.8], [1,1], [0.733,1], c = 'black')
ax.plot([0.8,0.8], [0,1], [0.733,0.733], c = 'black')
ax.plot([0.8,0.8], [0,1], [1,1], c = 'black')

ax.plot([0,0.8], [0,0], [0,0.733], c = 'black')
ax.plot([0,0.8], [0,0], [0.733,1], c = 'black')
ax.plot([0,0.8], [1,1], [0,0.733], c = 'black')
ax.plot([0,0.8], [1,1], [0.733,1], c = 'black')


# surcae on back and front
yy, zz = np.meshgrid(np.linspace(-0.1, 1.1, 12), np.linspace(-0.1, 0.833, 12)) 
x = np.ones_like(yy)*0
ax.plot_surface(x, yy, zz, alpha=0.8)

yy, zz = np.meshgrid(np.linspace(0,1, 12), np.linspace(0.733,1, 12)) 
x = np.ones_like(yy)*0.8
ax.plot_surface(x, yy, zz, alpha=0.8)


# surface of evaluation crossections
yy, zz = np.meshgrid(np.linspace(0,1, 12), np.linspace(0.18,0.84, 12)) 
x = np.ones_like(yy)*0.22
ax.plot_surface(x, yy, zz, alpha=0.3, color = 'tab:green')

yy, zz = np.meshgrid(np.linspace(0,1, 12), np.linspace(0.45,0.92, 12)) 
x = np.ones_like(yy)*0.51
ax.plot_surface(x, yy, zz, alpha=0.3, color = 'tab:green')

# yy, zz = np.meshgrid(np.linspace(0,1, 12), np.linspace(0.6,0.97, 12)) 
# x = np.ones_like(yy)*0.69
# ax.plot_surface(x, yy, zz, alpha=0.3)


# final configureations of the plot
ax.set_xlabel(r'$x [m]$')
ax.set_ylabel(r'$y [m]$')
ax.set_zlabel(r'$z [m]$')
ax.xaxis.set_ticks(np.arange(0, 0.801, 0.4))
ax.yaxis.set_ticks(np.arange(0, 1.01, 0.5))
ax.zaxis.set_ticks(np.arange(0, 1.01, 0.5))


ax.set_box_aspect([1,1,1])
set_axes_equal(ax)  
plt.show()

#%% ------------------------------ plot 2D fiew for Cooke's membrane -----------------------------
fig, ax = plt.subplots()

thickness = 0.7 

# clamping left
ax.plot([0,-0.03], [0.0,0.03], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.05,0.08], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.1,0.13], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.15,0.18], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.2,0.23], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.25,0.28], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.3,0.33], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.35,0.38], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.4,0.43], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.45,0.48], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.5,0.53], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.55,0.58], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.6,0.63], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.65,0.68], c = 'black', linewidth=thickness)
ax.plot([0,-0.03], [0.7,0.73], c = 'black', linewidth=thickness)

# size description
ax.plot([0,0], [-0.1,-0.15], c = 'black', linewidth=thickness)
ax.plot([0.8,0.8], [-0.1,-0.15], c = 'black', linewidth=thickness)
ax.plot([0,0.8], [-0.125,-0.125], c = 'black', linewidth=thickness)

ax.plot([0.9,0.95], [0,0], c = 'black', linewidth=thickness)
ax.plot([0.9,0.95], [0.733,0.733], c = 'black', linewidth=thickness)
ax.plot([0.9,0.95], [1,1], c = 'black', linewidth=thickness)
ax.plot([0.925, 0.925], [0,1], c = 'black', linewidth=thickness)

# axis
plt.arrow(x=0, y=0, dx=0, dy=1, linewidth=0.6, color = 'red', head_width = 0.02, head_length = 0.03) 
plt.arrow(x=0, y=0, dx=0.5, dy=0, linewidth=0.6, color = 'red', head_width = 0.02, head_length = 0.03) 

# geometry lines
ax.plot([0,0], [0,0.733], c = 'black')
ax.plot([0,0.8], [0,0.733], c = 'black')
ax.plot([0,0.8], [0.733,1], c = 'black')
ax.plot([0.8,0.8], [0.733,1], c = 'black')
ax.plot([0,0.8], [0.733, 0.733], c = 'black', linestyle = '--', linewidth=thickness)


# add text to axis and geometry lines
ax.text(0.02, 1.02, r'$z [m]$', c = "red", fontsize = 10)
ax.text(0.52, 0.02, r'$x [m]$', c = "red", fontsize = 10)

ax.text(0.35, -0.1, r'$0.8 \, m$', c = "black", fontsize = 10)
ax.text(0.85, 0.3, r'$0.733 \, m$', c = "black", fontsize = 10, rotation=90)
ax.text(0.85, 0.75, r'$0.267 \, m$', c = "black", fontsize = 10, rotation=90)


# final configurations
ax.set_aspect("equal")
ax.set_xlabel(r'$x [m]$')
ax.set_ylabel(r'$z [m]$')


plt.show()