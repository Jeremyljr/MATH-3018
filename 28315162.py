'''
Coursework 1
Name : Jeremy Lee Jia Ren
ID : 28315162  
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root

def f(t, q, options=None):
    '''
    RHS function for task 2
    
    Parameters: 

        t : int
            The input time tn
        q : Vector
            Given data, qn
        options : list
            A list containing additional options, default as none

    Returns: 

        xn : array of float
            Approximate solution of the function
        yn : array of float 
            Approximate solution at loctaion x 

    '''
    xn, yn = q
    dqdt = np.zeros_like(q)
    dqdt[0] = yn + 1
    dqdt[1] = 6 * t
    return dqdt

def MyRK3_step(f, t, qn, dt, options=None):
    '''
    Solving an IVP of y' = Ay + b, 
    where y(0)=0 in the interval of N steps in RK3
    
    Parameters: 

        f : function 
            A function that defines an ODE
        t : int
            The time tn
        qn : Vector
            Given data, qn
        dt : int
            Time step delta t
        options : list
            A list containing additional options, default as none

    Returns: 

        x : array of float
            Approximate solution of the function
        y : array of float 
            Approximate solution at loctaion x 

    '''
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and\
        np.all(np.isreal(t))), "t must be real, finite and not NaN" 
    assert((not np.any(np.isnan(dt))) and np.all(np.isfinite(dt)) and\
        np.all(np.isreal(dt) and dt>0)), "dt must be real, finite, non-zero and not NaN" 

    k1 = f(t, qn, options)
    yp1 = qn + k1 * (dt/2)
    k2 = f(t+dt/2, yp1, options)
    yp2 = qn - (k1 * dt) + (k2 * 2*dt)
    k3 = f(t+dt, yp2, options)
    return qn + (dt/6)*(k1 + 4*k2 +k3)

def q_exact(t, options=None):
    '''
    Exact solution for the function specified in task 2
    
    Parameters: 

        t : int
            The input time tn
        options : list
            A list containing additional options, default as none

    Returns: 

        xn : array of float
            Approximate solution of the function
        yn : array of float 
            Approximate solution at loctaion x 

    ''' 
    return np.array([t**3 + t, 3*t**2])

def question_2():
    '''
    Solves trancation error for task 2 after a single step
    '''
    t, dt = np.linspace(0, 1, 5, retstep=True)
    q = np.zeros((len(t), 2))
    q[0, : ] = 0 #initial condition
    q[1, : ] = MyRK3_step(f, t[0], q[0, :], dt) #first step

    Error = np.linalg.norm(q[1, :] - q_exact(t[1])) #error at first step
    print('Task 2: The tracation error is ' + str(Error) + ' after a single step')

question_2()

def f2(t, q, options=None):
    '''
    returns array of numbers for task 3
    
    Parameters: 

        t : int
            The time tn
        q : Vector
            Given data, qn
        options : list
            A list containing additional options, default as none

    Returns: 

        x : array of float
            Approximate solution of the function
        y : array of float 
            Approximate solution at loctaion x 

    '''
    return t-q

def q2Exact(t, options = None):
    '''
    returns array of numbers for exact solution for task 3
    
    Parameters: 

        t : int
            The time tn
        options : list
            A list containing additional options, default as none 

    Returns: 

        x : array of float
            Approximate solution of the function
        y : array of float 
            Approximate solution at loctaion x 

    '''
    return np.exp(-t) + t-1

def question_3():
    '''
    Solves trancation error for task 3 after a single step
    For a single step, it is expected to converge at fourth order
    '''
    num_points = 5*2**np.arange(10)
    '''
    powers of 2 in log plot will be evenly spaced, 
    np.arange(10) gives all integers from 0 to 10
    2**np.arange gives (2**0, 2**1, 2**2, ...) = (1, 2, 4, ...)
    5 * 2**np.arange gives (10, 20, 40, ...)
    '''
    error_points = np.zeros(len(num_points)) #array of same size as points in num_points
    dx_all_points = np.zeros_like(error_points)
    for n in range(len(num_points)): #Loops every step for each resolution
        x, dx = np.linspace(0, 1, num_points[n], retstep=True) #changing resolution with num_points(n)
        dx_all_points[n] = dx #storing all dx points in an array 
        y = np.zeros((len(x), ))
        y[0] = 0 #initial condition
        y[1] = MyRK3_step(f2, x[0], y[0], dx) #first step
        error_points[n] = np.abs(y[1] - q2Exact(x[1])) #computes at first point only

    p = np.polyfit(np.log(dx_all_points), np.log(error_points), 1) #best fit polynomial with s = 1 that mathces one array to another array
    plt.loglog(dx_all_points, error_points, 'kx', label= 'RK3') #loglog plot of errors against dx
    plt.loglog(dx_all_points, np.exp(p[1])* dx_all_points**(p[0]), 'b-') 
    plt.legend(('RK3', "$\propto \Delta t^{{{: .3f}}}$".format(p[0])), loc = 'upper left') #shows the convergence rate from polyfit component p[0]
    plt.xlabel('$dt$')
    plt.ylabel('$|$Error$|$')
    plt.title("Solution for task 3")
    plt.show()

question_3()

def MyGRRK3_step(f, t, qn, dt, options=None):
    '''
    Solving an IVP implicitly using Gauss-Radau Rungga-Kutta method 
    for one step in the algorithm
    
    Parameters: 

        f : function 
            A function that defines an ODE
        t : int
            The time tn
        qn : Vector
            Given data, qn
        dt : int
            Time step delta t
        options : list
            A list containing additional options, default as none

    Returns: 

        x : array of float
            Approximate solution of the function
        y : array of float 
            Approximate solution at loctaion x 

    '''
    assert((not np.any(np.isnan(qn))) and np.all(np.isfinite(qn)) and\
        np.all(np.isreal(qn))), "qn must be real, finite and not NaN" 
    assert((not np.any(np.isnan(t))) and np.all(np.isfinite(t)) and\
        np.all(np.isreal(t))), "t must be real, finite and not NaN" 
    assert((not np.any(np.isnan(dt))) and np.all(np.isfinite(dt)) and\
        np.all(np.isreal(dt) and dt>0)), "dt must be real, finite, non-zero and not NaN" 
    
    def F(K):
        rtk = np.split(K,2) #np.split splits the K value size into 2 parts, k1 and k2
        k1 = f(t+1/3*dt, qn+dt/12*(5 * rtk[0] - rtk[1]), options)
        k2 = f(t+dt, qn+dt/4*(3 * rtk[0] + rtk[1]), options)
        K_func = np.concatenate((k1, k2))
        return K - K_func

    k_1 = f(t+dt/3, qn, options) #initial guess
    k_2 = f(t+dt, qn, options) #initial guess
    if isinstance(k_1, np.ndarray) == True: 
        K0 = np.concatenate((k_1, k_2))
    else:
        K0 = np.array([k_1, k_2])
    #The if and else statement above checks if k_1 and k_2 is an array
    sol = root(F, K0) 
    #blackbox solver scipy.optimize.root to find roots of function
    q_next = sol.x
    
    split = np.split(q_next, 2)
    return qn + dt/4 * (3 * split[0] + split[1])

def question_5(): 
    '''
    Task 5 repeats the steps in task 3 just with GRRK3 rather than RK3
    For a single step, it is expected to converge at 4th order
    '''
    num_points = 5*2**np.arange(10)
    error_points = np.zeros(len(num_points))
    dx_all_points = np.zeros_like(error_points)
    for n in range(len(num_points)): 
        x, dx = np.linspace(0, 1, num_points[n], retstep=True) 
        dx_all_points[n] = dx 
        y = np.zeros((len(x), ))
        y[0] = 0 
        y[1] = MyGRRK3_step(f2, x[0], y[0], dx)
        error_points[n] = np.abs(y[1] - q2Exact(x[1])) 

    p = np.polyfit(np.log(dx_all_points), np.log(error_points), 1) 
    plt.loglog(dx_all_points, error_points, 'kx', label = 'GRRK3') 
    plt.loglog(dx_all_points, np.exp(p[1])* dx_all_points**(p[0]), 'b-', label = 'Exact solution')
    plt.legend(('GRRK3', "$\propto \Delta t^{{{: .3f}}}$".format(p[0])), loc = 'upper left')
    plt.xlabel('$dt$')
    plt.ylabel('$|$Error$|$')
    plt.title("Solution for task 5")
    plt.show()

question_5()

options = {'gamma' : -2, 'omega' : 5, 'epsilon' : 0.05}
Gamma = options['gamma']
Epsilon = options['epsilon']
Omega = options['omega']

def p_robinson(t, qn, options):
    '''
    RHS function for the Modified Prothero-Robinson Problem
    
    Parameters: 

        t : int
            The input time tn
        q : Vector
            Given data, qn
        options : list
            A list containing additional options, default as none

    Returns: 

        xn : array of float
            Approximate solution of the function
        yn : array of float 
            Approximate solution at loctaion x 

    '''
    x, y = qn
    a = np.array([[Gamma, Epsilon], [Epsilon, -1]])
    b_1 = (-1 + x**2 - np.cos(t))/(2*x)
    b_2 = (-2 + y**2 - (np.cos(Omega * t)))/ 2*y
    b = np.array([b_1, b_2])
    ab = a @ b
    c = np.array([np.sin(t) / (2*x), Omega * np.sin(Omega * t)/ (2*y)])
    return ab - c

def q_rob_exact(t, options):
    '''
    Exact solution for the Modified Prothero-Robinson Problem
    
    Parameters: 

        t : int
            The input time tn
        options : list
            A list containing additional options, default as none

    Returns: 

        xn : array of float
            Approximate solution of the function
        yn : array of float 
            Approximate solution at loctaion x 

    '''
    ex = np.sqrt(1 + np.cos(t))
    wai = np.sqrt(2 + np.cos(Omega * t))
    return np.array([ex, wai]).T

def question_6(): 
    '''
    Using the Modified Prothero-Robinson Problem, RK3 and GRRK algorithm is tested here using 
    "non-stiff" values. Subplots of the numerical and exact solution is plotted. 
    '''
    t, dt = np.linspace(0, 1, 21, retstep=True)
    q = np.zeros((len(t), 2))
    w = np.zeros((len(t), 2)) #w is the array that will store GRRK3 values
    q[0, : ] = np.array([np.sqrt(2), np.sqrt(3)]) #RK3 initial values
    w[0, : ] = np.array([np.sqrt(2), np.sqrt(3)]) #GRRK initial value
    for i in range(0, len(t)-1): #compute all points of y using RK3 and GRRK
            q[i+1, :] = MyRK3_step(p_robinson, t[i], q[i, :], dt)
            w[i+1, :] = MyGRRK3_step(p_robinson, t[i], w[i, :], dt)
    rob = q_rob_exact(t, options) #array of exact solution of P-Robinson problem

    fig, ax = plt.subplots(1, 2)
    ax[0].plot(t, q[:,0], 'g+', label = 'RK3')
    ax[0].plot(t, w[:,0], 'bx',label = 'GRRK3')
    ax[0].plot(t, rob[:, 0], 'k-', label = 'Exact')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$x$')
    ax[0].legend()

    ax[1].plot(t, q[:,1], 'g+', label = 'RK3')
    ax[1].plot(t, w[:,1], 'bx',label = 'GRRK3')
    ax[1].plot(t, rob[:, 1], 'k-', label = 'Exact')
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$y$')

    plt.legend()
    fig.suptitle("Solution for task 6")
    fig.tight_layout()
    plt.show()

question_6()

def question_7():
    '''
    Convergence plot for changing values of dt.
    For both algorithms, expect converging at third order
    '''
    delta = (0.1/ 2**np.arange(8))
    points = 1/delta
    error_points_RK3 = np.zeros(len(points))
    error_points_GRRK = np.zeros(len(points))
    dx_all_points = np.zeros_like(error_points_RK3)
    for n in range(len(points)): #Loops every step for each resolution
        t, dt = np.linspace(0, 1, int(points[n]), retstep=True) #changing resolution with points(n)
        dx_all_points[n] = dt #storing all dt points in an array 
        y = np.zeros((len(t), 2))
        w = np.zeros((len(t), 2))
        y[0, :] = np.array([np.sqrt(2), np.sqrt(3)]) 
        w[0, :] = np.array([np.sqrt(2), np.sqrt(3)]) #initial condition
        for i in range(len(t)-1): #Loops for each value of t
            y[i+1, :] = MyRK3_step(p_robinson, t[i], y[i, :], dt, options)
            w[i+1, :] = MyGRRK3_step(p_robinson, t[i], w[i, :], dt, options)     
        error_points_RK3[n] = dt * np.sum(np.abs((y - q_rob_exact(t, options))))
        error_points_GRRK[n] = dt * np.sum(np.abs((w - q_rob_exact(t, options))))
    
    p1 = np.polyfit(np.log(dx_all_points), np.log(error_points_RK3), 1) #best fit polynomial with s = 1 that mathces one array to another array
    plt.loglog(dx_all_points, error_points_RK3, 'bx', label='RK3') #loglog plot of errors against dx
    plt.loglog(dx_all_points, np.exp(p1[1])* dx_all_points**(p1[0]),'b-', 
                label="$\propto \Delta t^{{{: .3f}}}$".format(p1[0]))
    
    p2 = np.polyfit(np.log(dx_all_points), np.log(error_points_GRRK), 1) #best fit polynomial with s = 1 that mathces one array to another array
    plt.loglog(dx_all_points, error_points_GRRK, 'gx', label='GRRK3') #loglog plot of errors against dx
    plt.loglog(dx_all_points, np.exp(p2[1])* dx_all_points**(p2[0]), 'g-', 
                label ="$\propto \Delta t^{{{: .3f}}}$".format(p2[0]))
    
    plt.xlabel('$dt$')
    plt.ylabel('$|$Error$|$')
    plt.legend(loc = "upper left")
    plt.title("Solution for task 7")
    plt.show()

question_7()

options = {'gamma' : -2*10**5, 'omega' : 20, 'epsilon' : 0.5}
Gamma = options['gamma']
Epsilon = options['epsilon']
Omega = options['omega']

def question_8():
    '''
    For RK3 algorithm, for the Modified Prothero-Robinson Problem, with "stiff" values
    and dt = 0.001. For this timestep, expect the algorithm to fail
    '''
    t, dt = np.linspace(0, 1, 1001, retstep=True)
    q = np.zeros((len(t), 2))
    q[0, : ] = np.array([np.sqrt(2), np.sqrt(3)])
    for i in range(0, len(t)-1): #compute all points of y using RK3 
            q[i+1, :] = MyRK3_step(p_robinson, t[i], q[i, :], dt)
    rob = q_rob_exact(t, options)
    
    fig, ax = plt.subplots(1,2)
    ax[0].plot(t, q[:, 0],'b-', label = 'RK3')
    ax[0].plot(t, rob[:, 0],'g-', label = 'Exact')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$x$')


    ax[1].plot(t, q[:, 1],'b-', label = 'RK3')
    ax[1].plot(t, rob[:, 1],'g-', label = 'Exact')
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$y$')

    fig.suptitle("Solution for task 8")
    fig.tight_layout()
    plt.legend(loc='upper right')
    plt.show()

question_8()

def question_9():
    '''
    For GRRK3 algorithm, for the Modified Prothero-Robinson Problem, with "stiff" values
    and dt = 0.005. For this timestep, expect the algorithm pass showing same plot as
    figure 1 in CW brief
    '''
    t, dt = np.linspace(0, 1, 200, retstep=True)
    q = np.zeros((len(t), 2))
    q[0, : ] = np.array([np.sqrt(2), np.sqrt(3)])
    for i in range(0, len(t)-1): #compute all points of y using RK3 
            q[i+1, :] = MyGRRK3_step(p_robinson, t[i], q[i, :], dt)
    rob = q_rob_exact(t, options)
    
    fig, ax = plt.subplots(1,2)
    ax[0].plot(t, q[:, 0],'b-', label = 'GRRK3')
    ax[0].plot(t, rob[:, 0],'g-', label = 'Exact')
    ax[0].set_xlabel(r'$t$')
    ax[0].set_ylabel(r'$x$')
    ax[0].legend()

    ax[1].plot(t, q[:, 1],'b-', label = 'GRRK3')
    ax[1].plot(t, rob[:, 1],'g-', label = 'Exact')
    ax[1].set_xlabel(r'$t$')
    ax[1].set_ylabel(r'$y$')

    fig.suptitle("Solution for task 9")
    plt.legend(loc= 'upper right')
    fig.tight_layout()
    plt.show()

question_9()

def question_10():
    '''
    Convergence plot for changing values of dt for GRRK3 algorithm

    ----------------------------------------------------------------------------------
    To improve the convergence rate to what we expect, we can exclude the final points 
    in the plot as it may be contaminated by the higher order terms. 
    i.e. 
    p2 = np.polyfit(np.log(dx_all_points[:-1]), np.log(error_points_GRRK[:-1]), 1)
    ----------------------------------------------------------------------------------

    '''
    delta = (0.05/ 2**np.arange(8))
    points = 1/delta
    error_points_GRRK = np.zeros(len(points))#array of same size as points in num_points
    dx_all_points = np.zeros_like(error_points_GRRK)
    for n in range(len(points)): 
        x, dx = np.linspace(0, 1, int(points[n]), retstep=True) #changing resolution with num_points(n)
        dx_all_points[n] = dx #storing all dx points in an array 
        w = np.zeros((len(x), 2))
        w[0, :] = np.array([np.sqrt(2), np.sqrt(3)]) #initial condition
        for i in range(len(x)-1):
            w[i+1, :] = MyGRRK3_step(p_robinson, x[i], w[i, :], dx, options)     
        error_points_GRRK[n] = dx * np.sum(np.abs((w - q_rob_exact(x, options))))
    p2 = np.polyfit(np.log(dx_all_points), np.log(error_points_GRRK), 1) #best fit polynomial with s = 1 that mathces one array to another array
    plt.loglog(dx_all_points, error_points_GRRK, 'bx', label='GRRK3') #loglog plot of errors against dx
    plt.loglog(dx_all_points, np.exp(p2[1])* dx_all_points**(p2[0]), 'g-', 
                label ="$\propto \Delta t^{{{: .3f}}}$".format(p2[0]))
    
    plt.xlabel('$dt$')
    plt.ylabel('$|$Error$|$')
    plt.legend(loc = "upper left")
    plt.title("Solution for task 10")
    plt.show()

question_10()