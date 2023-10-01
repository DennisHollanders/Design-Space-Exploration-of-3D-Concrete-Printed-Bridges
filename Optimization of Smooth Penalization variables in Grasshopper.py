#import libraries
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from scipy.optimize import check_grad

def equation(x: float, t:float, c:float, s:float , d:float ) -> float:
    """
    Sigmoid function for a given upper and lower yield strength

    :param c: yield strength in compresison
    :param t: yield strength in tension
    :param x: x of the curve (strain)
    :param s: Sigmoid control variable
    :param d: Sigmoid control variable

    :return: y of the curve (stress)
    """
    return  -t + (c + t) * expit(-1*s * (x - d))


def objective_s(s:float, *args) -> float:
    """
    the objective function returns the summed difference between the y values of a certain amount of points 
    on the sigmoid curve and bi-linear youngs modulus penalization

    :param c: yield strength in compresison
    :param t: yield strength in tension
    :param x: x of the curve (strain)
    :param s: Sigmoid control variable
    :param d: Sigmoid control variable

    :return: total distance between the points on both curves
    """
    t, c, E, x_values = args
    # Calculate d using the provided formula
    # d and s are dependant on each because the curve goes through 0,0 (this is therefore a known point and can be used to calculate the ratio between those two points)
    d = -1 * np.log(c / t) / s
    # Calculate the difference between the two curves
    difference = np.sum(np.maximum(0, np.abs(E * x_values) - equation(x_values, t, c, s, d)) ** 2
                                 + (np.abs(E * x_values) - equation(x_values, t, c, s, d)) ** 2)
    return difference

# Define some constants and parameters
c = Yield_strength_compression 
t = Yield_strength_tension
E = Youngs_modulus
num_points = 10


# Calculate upper and lower bounds for x_values
upper_bound_x = ( c / E )
lower_bound_x = -(t / (E/Ratio_Nefs))

# Create arrays of x_values for different cases
x_values_positive = np.linspace(0, upper_bound_x, num_points)
x_values_negative = np.linspace(lower_bound_x, 0, num_points)
x_values_eq = np.linspace(lower_bound_x, upper_bound_x * 2, 50)

# Calculate corresponding y_values for different cases
y_values_positive = E * x_values_positive
y_values_negative = (E / Ratio_Nefs) * x_values_negative

# Concatenate x, y values
x_values = np.concatenate((x_values_negative, x_values_positive))
y_values = np.concatenate((y_values_negative, y_values_positive))

# Initial guess for the sigmoid parameter s
s_initial_guess = -30000

# Use the minimize function to find the optimized value of s
result = minimize(objective_s, s_initial_guess, args=(t, c, E, x_values))

# Extract the optimized 's' value from the result and round it to an arbitrary 7 decimals
optimized_s = np.round(result.x,7)[0]

# Calculate d using the optimized s
d = -1 * np.log(c / t) / optimized_s
d_result = d