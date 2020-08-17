import numpy as np
import optimiser

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib import animation
from itertools import zip_longest

# =========================================================================================================================
# =========================================================================================================================
# =======================  SET UP THE STYLE OF MATPLOTLIB FIGURE TO BE GENERATED ==========================================
# =========================================================================================================================
# =========================================================================================================================
# Define the style of the plot
# A list of default styles is given at https://matplotlib.org/gallery/style_sheets/style_sheets_reference.html
plt.style.use('bmh')
plt.rcParams['axes.unicode_minus'] = False
# Define fonts to be used as standard for the plot
# The Computer Modern roman font (cmr10) is used here to provide a close match to LaTeX text. 
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'cmr10'
plt.rcParams['font.weight'] = 'light'
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=["r", "k", "c"]) 
# Set the width and colour of the border around the plot
plt.rcParams['axes.linewidth'] = 3.5
plt.rc('axes',edgecolor='black')

# Set up figure environment
fig, ax = plt.subplots(figsize=(11, 6))

# Define the title of the plot and the labels for the x and y axes
# plt.title(r'$\bar{q}$ calculations for strong tetrahedra only with $\epsilon_{AA}=1, \epsilon_{BB}=5$', size=26)
ax.set_xlabel(r'$x$', size=25)
ax.set_ylabel(r'$y$', size=25, rotation=90)

# Generate a mesh over which we will evaluate our function to define a contour plot.
xmin, xmax, xstep = -5.0, 5.0, .01
ymin, ymax, ystep = -5.0, 5.0, .01
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep), np.arange(ymin, ymax + ystep, ystep))

ax.xaxis.set_ticks(np.arange(xmin, xmax+0.0001, 1.0))
ax.yaxis.set_ticks(np.arange(ymin, ymax+0.0001, 1.0))
# Set the size of the axes parameters (i.e., the x and y values)
plt.tick_params(labelsize=20)
# Add or remove grid lines from the plot
ax.grid(False)

# Set the width of the ticks on the x and y axes
ax.xaxis.set_tick_params(width=3.5, length=5.0)
ax.yaxis.set_tick_params(width=3.5, length=5.0)

# Set the facecolour of the plot to white, as it is grey for the bmh style
ax.set_facecolor('xkcd:white')

# =========================================================================================================================
# Class for generating animation with Matplotlib.
# Taken from http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/
# =========================================================================================================================
class TrajectoryAnimation(animation.FuncAnimation):
    def __init__(self, *paths, labels=[], fig=None, ax=None, frames=None, 
                 interval=5, repeat_delay=5, blit=True, **kwargs):

        if fig is None:
            if ax is None:
                fig, ax = plt.subplots()
            else:
                fig = ax.get_figure()
        else:
            if ax is None:
                ax = fig.gca()

        self.fig = fig
        self.ax = ax
        
        self.paths = paths

        if frames is None:
            frames = max(path.shape[1] for path in paths)
  
        self.lines = [ax.plot([], [], label=label, lw=2)[0] 
                      for _, label in zip_longest(paths, labels)]
        
        self.points = [ax.plot([], [], 'o', color=line.get_color())[0] 
                       for line in self.lines]

        super(TrajectoryAnimation, self).__init__(fig, self.animate, init_func=self.init_anim,
                                                  frames=frames, interval=interval, blit=blit,
                                                  repeat_delay=repeat_delay, **kwargs)

    def init_anim(self):
        for line, point in zip(self.lines, self.points):
            line.set_data([], [])
            point.set_data([], [])
        return self.lines + self.points

    def animate(self, i):
        for line, point, path in zip(self.lines, self.points, self.paths):
            line.set_data(*path[::,:i])
            point.set_data(*path[::,i-1:i])
        return self.lines + self.points
# =========================================================================================================================
# =========================================================================================================================

# =========================================================================================================================
# =========================================================================================================================
# =======================  DEFINE THE TEST FUNCTIONS FOR THE OPTIMISATION ALGORITHMS ======================================
# =========================================================================================================================
# =========================================================================================================================

# Beale Function
f  = lambda x, y: (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
g  = lambda x, y: [2*( (1.5- x+x*y)*(y-1) + (2.25-x+x*y**2)*(y**2-1) + (2.625-x+x*y**3)*(y**3-1)   ),
                   2*( (1.5- x+x*y)*x     + (2.25-x+x*y**2)*(2*x*y)  + (2.625-x+x*y**3)*(3*x*y**2) )]

sd = optimiser.Optimise(X=[3.,4.],function=f,gradient=g,err=1.e-9,method=1)
cg = optimiser.Optimise(X=[3.,4.],function=f,gradient=g,err=1.e-9,method=2)

print(sd.minimum, sd.nsteps)
print(cg.minimum, cg.nsteps)

# Booth Function
f  = lambda x, y: (x + 2*y - 7)**2 + (2*x + y - 5)**2
g  = lambda x, y: [2*( (x + 2*y - 7)     + (2*x + y - 5)*(2) ),
                   2*( (x + 2*y - 7)*(2) + (2*x + y - 5) )]

sd = optimiser.Optimise(X=[3.,4.],function=f,gradient=g,err=1.e-9,method=1)
cg = optimiser.Optimise(X=[3.,4.],function=f,gradient=g,err=1.e-9,method=2)

print(sd.minimum, sd.nsteps)
print(cg.minimum, cg.nsteps)

# Sphere Function
f  = lambda *x: np.sum(np.asarray(x)**2)
g  = lambda *x: 2*np.asarray(x)

sd = optimiser.Optimise(X=[13.,40.,51.,6.],function=f,gradient=g,err=1.e-9,method=1)
cg = optimiser.Optimise(X=[13.,40.,51.,6.],function=f,gradient=g,err=1.e-9,method=2)

print(sd.minimum, sd.nsteps)
print(cg.minimum, cg.nsteps)

# Rosenbrock Function
def func(*args):
    X = np.asarray(args)
    fx = 0
    for i in range(len(X)-1):
        fx += 100*(X[i+1]-X[i]**2)**2 + (1-X[i])**2
    return fx

def grad(*args):
    X = np.asarray(args)
    G = np.zeros(len(X))
    for i in range(len(X)-1):
        G[i]   += 200*(X[i+1]-X[i]**2)*(-2*X[i]) - 2*(1-X[i])
        G[i+1] += 200*(X[i+1]-X[i]**2)
    return G

x0 = [0.,3]
sd = optimiser.Optimise(X=x0,function=func,gradient=grad,err=1.e-9,method=1)
cg = optimiser.Optimise(X=x0,function=func,gradient=grad,err=1.e-9,method=2)

print(sd.minimum, sd.nsteps)
print(cg.minimum, cg.nsteps)
# =========================================================================================================================
# =========================================================================================================================

# =========================================================================================================================
# =========================================================================================================================
# ==================================  PRODUCE VIDEO OF THE PATH TO THE MINIMUM ============================================
# =========================================================================================================================
# =========================================================================================================================
path1 = sd.path
path2 = cg.path

# Evaluate the function over the mesh.
z = func(x, y)

# The minimum of the function (to be used for plotting purposes)
minima = np.array([1., 1.]).reshape(-1, 1)

ax.contour(x, y, z, levels=np.logspace(-3, 6, 35), norm=LogNorm(), cmap='jet')
cp = ax.contourf(x, y, z, levels=np.logspace(-3, 6, 500), norm=LogNorm(), cmap='rainbow', alpha=0.8)

ax.plot(*minima, 'r*', markersize=17, markerfacecolor='gold',markeredgecolor='black')

ax.set_xlim((xmin, xmax))
ax.set_ylim((ymin, ymax))

anim = TrajectoryAnimation(*[path1,path2], labels=['Steepest Descent','Conjugate Gradient'], ax=ax)

ax.legend(loc='upper left', facecolor='w', framealpha=0.0, fontsize=20)
plt.tight_layout()
plt.show()

# Set up formatting for the movie files
# Writer = animation.writers['ffmpeg']
# writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
# anim.save('linesearch.mp4', writer=writer)