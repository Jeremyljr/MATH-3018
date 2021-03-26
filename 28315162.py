import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, solve_bvp
from scipy.optimize import brentq, newton, root

def my_func(t, y, y_dot, alpha=1, beta=1): #L function 
    return alpha*y_dot**2 + beta*(t**2 - 1)*(y_dot)**3 - y

def centralDifference(f, t, y, y_dot, alpha, beta, h): #Finds the partial derivative
    dLdy = (f(t, y+h, y_dot, alpha, beta) - f(t, y-h, y_dot, alpha, beta))/(2*h)
    d2LdTdY_dot = (f(t+h, y, y_dot+h, alpha, beta) - f(t-h, y, y_dot+h, alpha, beta) - f(t+h, y, y_dot-h, alpha, beta) + f(t-h, y, y_dot-h, alpha, beta))/(4*h**2)
    d2LdYdY_dot = (f(t, y+h, y_dot+h, alpha, beta) - f(t, y-h, y_dot+h, alpha, beta) - f(t, y+h, y_dot-h, alpha, beta) + f(t, y-h, y_dot-h, alpha, beta))/(4*h**2)
    d2Ldy_dot_2 = (f(t, y, y_dot+h, alpha, beta) - 2*f(t, y, y_dot, alpha, beta) + f(t, y, y_dot-h, alpha, beta))/h**2

    return [dLdy, d2LdTdY_dot, d2LdYdY_dot, d2Ldy_dot_2]
#----------------------Check central differencing values converge----------------------#
#'''
def central_diff_test():
    #totally arbitory values 
    x = 1 
    y = 0.3
    dy = 0.1
    A = 1
    B = 1

    h_val = 1e-1 / 1.5**np.arange(15)
    error = np.zeros(len(h_val))
    error_2 = np.zeros(len(h_val))
    error_3 = np.zeros(len(h_val))
    error_4 = np.zeros(len(h_val))
    for i in range(len(error)):
        cenDiff = centralDifference(my_func, x, y, dy, alpha=1, beta=1, h= h_val[i])
        error[i] = np.abs(-1 - cenDiff[0])
        error_2[i] = np.abs((6*B*x*dy**2) - cenDiff[1])
        error_3[i] = np.abs(0- cenDiff[2])
        error_4 [i] = np.abs( (2*A+6*B*(x**2 - 1)*(dy)) - cenDiff[3])

    print('For a h value of ' + str(h_val[-1]),',the errors for all 4 derivatives respectively are, ' 
    + str(error[-1]) + ', ' + str(error_2[-1]) + ', ' + str(error_3[-1]) + ' and, ' + str(error_4[-1])) 
#'''

#----------------------Test if algorithm works as expected by checking with b = 0 ----------------------#
#'''
def exact_sol_test():
    def test_rhs(x,q): 
        y, dy = q
        dqdx = np.zeros_like(q)
        #cd = centralDifference(my_func, x, y, dy, alpha=5, beta=0, h=1e-4)

        dqdx[0] = dy
        dqdx[1] = 3*dy - 2*y
        #(cd[0]-cd[1]-(cd[2]*y)) / cd[3]
        return dqdx

    def test_exact(x):
        return (np.exp(2.0*x - 1.0) - np.exp(x - 1.0)) / (np.exp(1.0) - 1.0)

    def test_shooting_residual(dy_zero, tolerance): #setting up for phi(z) = 0 
        ics = [0, dy_zero]
        soln = solve_ivp(test_rhs, [0, 1], ics,
                        rtol=tolerance, atol=1e-3*tolerance)
        residual = soln.y[0, -1] - 1  # Error in satisfying BC at t=1
        return residual

    def test_shooting(tolerance):
        interval = [-1, 1]  # For the root find; just guessing here
        dy_zero = brentq(test_shooting_residual, interval[0], interval[1],
                        args=(tolerance,), xtol=1e-4*tolerance)
        #solve BVP
        #print(dy_zero)
        ics = [0, dy_zero]
        soln = solve_ivp(test_rhs, [0, 2], ics, dense_output=True,
                        rtol=tolerance, atol=1e-3*tolerance)
        return soln

    x = np.linspace(0, 1, 101)
    soln = test_shooting(1e-8)
    plt.plot(x, soln.sol(x)[0, :], ls="-", label=r"$tolerance=10^{-3}$")
    plt.plot(x, test_exact(x), 'kx')
    plt.title("Price curve for alpha = beta")
    plt.xlabel("$t$")
    plt.ylabel("y($t$)")
    plt.legend()
    plt.show()
#'''

#----------------------For a & b = 5----------------------#
#'''
def mod_a():
    def rhs(x, q): #converting to single ODE
        y, dy = q
        dqdx = np.zeros_like(q)
        cd = centralDifference(my_func, x, y, dy, alpha=5, beta=5, h=1e-4)

        dqdx[0] = dy
        dqdx[1] = (cd[0]-cd[1]-(cd[2]*y)) / cd[3]
        return dqdx

    def shooting_residual(dy_zero, tolerance): #setting up for phi(z) = 0 
        ics = [1, dy_zero]
        soln = solve_ivp(rhs, [0, 1], ics,
                        rtol=tolerance, atol=1e-3*tolerance)
        residual = soln.y[0, -1] - 0.9  # Error in satisfying BC at t=1
        return residual

    def shooting(tolerance):
        interval = [-0.1, 0.1]  # For the root find; just guessing here
        dy_zero = brentq(shooting_residual, interval[0], interval[1],
                        args=(tolerance,), xtol=1e-4*tolerance)
        #solve BVP
        #print(dy_zero)
        ics = [1, dy_zero]
        soln = solve_ivp(rhs, [0, 2], ics, dense_output=True,
                        rtol=tolerance, atol=1e-3*tolerance)
        return soln

    x = np.linspace(0, 1, 101)
    soln = shooting(1e-3)
    plt.plot(x, soln.sol(x)[0, :], ls="-", label=r"$tolerance=10^{-3}$")
    plt.title("Price curve for alpha = beta")
    plt.xlabel("$t$")
    plt.ylabel("y($t$)")
    plt.legend()
    plt.show()
#'''

#----------------------For a = 7/4 & b = 5----------------------#
#'''
def mod_b():
    def rhs_2(x, q): #converting to single ODE
        y, dy = q
        dqdx = np.zeros_like(q)
        cd = centralDifference(my_func, x, y, dy, alpha=7/4, beta=5, h=1e-4)

        dqdx[0] = dy
        dqdx[1] = (cd[0]-cd[1]-(cd[2]*y)) / cd[3]
        return dqdx

    def shooting_residual_2(dy_zero, tolerance): #setting up for phi(z) = 0 
        ics = [1, dy_zero]
        soln = solve_ivp(rhs_2, [0, 1], ics,
                        rtol=tolerance, atol=1e-3*tolerance)
        residual = soln.y[0, -1] - 0.9  # Error in satisfying BC at x=2
        return residual

    def shooting_2(tolerance):
        interval = [-1, 0.1]  # For the root find; just guessing here
        dy_zero = brentq(shooting_residual_2, interval[0], interval[1],
                        args=(tolerance,), xtol=1e-4*tolerance)
        #solve BVP
        ics = [1, dy_zero]
        soln = solve_ivp(rhs_2, [0, 2], ics, dense_output=True,
                        rtol=tolerance, atol=1e-3*tolerance)
        return soln

    x = np.linspace(0, 1, 101)
    soln_2 = shooting_2(1e-3)
    plt.plot(x, soln_2.sol(x)[0, :], ls=":", label=r"$tolerance=10^{-3}$")
    plt.title("Price curve for alpha=7/4, beta = 5")
    plt.xlabel("$t$")
    plt.ylabel("y($t$)")
    plt.legend()
    plt.show()
#'''



# soln_accurate = shooting(1e-3)
# y_accurate = soln_accurate.sol(x)[0, :]
# tolerances = 1e-1 / 1.5**np.arange(15)
# errors = np.zeros((len(tolerances),))
# for i in range(len(tolerances)):
#     soln = shooting(tolerances[i])
#     y_soln = soln.sol(x)[0, :]
#     errors[i] = np.linalg.norm(y_accurate - y_soln, 2)
# plt.loglog(tolerances, errors, 'kx')
# plt.loglog(tolerances, tolerances, 'b-')
# plt.xlabel('tol')
# plt.ylabel('Errors')
# plt.show()
#'''