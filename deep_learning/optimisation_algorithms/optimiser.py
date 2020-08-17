import math
import numpy as np

class Optimise:
    """
    Performs local optimisation on a multidimensional function using the steepest descent,
    conjugate gradient algorithms with a backtracking line search algorithm.

    Attributes
    ----------
    X : ndarray
        List of initial coordinates for the function to be optimised.
    function : function
        Function to be optimised.
    gradient : function
        Gradient of the function to be optimised. If the function is multidimensional this should be 
        a list of partial derivatives given in the same order as X.
    err : float
        Threshold convergence value for the local optimisation algorithms,
    method : int
        ID of the local optimisation algorithm to be used:
            1 - Steepest Descent
            2 - Conjugate Gradient

    Methods
    -------
    steepest_descent()
        Performs a local optimisation using the Steepest Descent algorithm.
    conjugate_gradient()
        Performs a local optimisation using the Conjugate Gradient algorithm.
    backtrack()
        Determines a step size for each iteration of a local optimisation using
        the backtracking algorithm with Armijo condition.

    Returns
    -------
    nsteps : int
        Number of iterations required to reach the local minimum.
    path : ndarray
        List of coordinates from each iteration of the local minimisation.
    minimum : ndarray
        Coordinates of the local minimum of the basin associated with the 
        initial conditions. 
    steps : ndarray
        List of stepsizes used at each iteration of the local optimisation.
    """
    def __init__(self, X, function, gradient, err, method=2):
    
    #   Initialise input parameters for the optimisation algorithms
        self.X      = np.array(X)
        self.f      = function
        self.g      = gradient
        self.err    = err
    #   Initialise parameters describing the convergence of the optimisation algorithms
        self.nsteps = 0
        self.path   = []
        self.steps  = []
    #   Perform local optimisation.
        if(method==1):
            self.steepest_descent()
        elif(method==2):
            self.conjugate_gradient()
    #   Extract the coordinates of the local minimum and the path taken to it.
        self.minimum = self.path[-1]
        self.path    = np.array(self.path).T

    def steepest_descent(self):
    #   Define the initial coordinates for iteration i=0
        x0 = self.X
        xi = x0
    #   Compute the gradient and the square of its magnitude at i=0
        gi = np.array(self.g(*xi))
        gd = np.dot(gi,gi)
    #   Add the initial coordinates the path to the local minimum.
        self.path.append(xi)
    #   Calculate the square of the convergence threshold.
        errsq = self.err**2
    #   Iteratively update the coordinates using the Steepest Descent algorithm
    #   until the convergence criterion is met.
        while gd > errsq:
        #   Determine the step size for this iteration using the backtracking algorithm.
            a = self.backtrack(xi=xi,gi=gi,di=-gi, a0=1)
        #   Update the coordinates
            xi = xi - a*gi
        #   Calculate the gradient and the square of its magnitude at the new coordinates
            gi = np.array(self.g(*xi))
            gd = np.dot(gi,gi)
        #   Update parameters describing the convergence of the optimisation algorithm.
            self.path.append(xi)
            self.nsteps += 1
            self.steps.append(a)

    def conjugate_gradient(self):
    #   Define the initial coordinates for iteration i=0  
        x0 = self.X
        xi = x0
    #   Compute the gradient and the square of its magnitude at i=0
        gi = np.array(self.g(*xi))
        gd = np.dot(gi,gi)
    #   Compute the search direction, taking it to be equal to the negative of the 
    #   gradient at iteration i=0.
        di = -gi
    #   Calculate the square of the convergence threshold.
        errsq = self.err**2
    #   Iteratively update the coordinates using the Conjugate Gradient algorithm
    #   until the convergence criterion is met.    
        while gd > errsq:
        #   Determine the step size for this iteration using the backtracking algorithm.
            a = self.backtrack(xi=xi,gi=gi,di=di,a0=1)
        #   Update the coordinates
            xi = xi + a*di
        #   Save the old gradient and search direction, which will be used to calculate 
        #   the search direction for the next iteration.
            gj = np.copy(gi)
            dj = np.copy(di)
        #   Calculate the gradient and the square of its magnitude at the new coordinates
            gi = np.array(self.g(*xi))
            gd   = np.dot(gi,gi)
        #   Calculate the search direction for the next iteration.
            b  = np.dot(gi,(gi-gj)) / np.dot(gj,gj)
            di = b*dj - gi
        #   Update parameters describing the convergence of the optimisation algorithm.
            self.path.append(xi)
            self.nsteps += 1
            self.steps.append(a)

    def backtrack(self, xi, gi, di, a0, c1=0.5, tau=0.5):
    #   Calculate the value of the function at the coordinates for the 
    #   current iteration of the optimisation algorithm.
        fi = self.f(*xi)
    #   Calculate the dot product of the gradient and the search direction,
    #   to be used to evaluate the Armijo condition.  
        gi = np.dot(gi, di)
        ai = a0
    #   While the step size does not provide a sufficient decrease in the function f(X),
    #   adjust the step size using the contraction factor tau.
        while( self.f( *(xi+ai*di) ) > (fi + c1*ai*gi) ):
            ai *= tau

        return ai