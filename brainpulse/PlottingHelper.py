import matplotlib.pyplot as plt
import numpy as np
from fenics import *

def extract_cross_section(function, points, times=None, filter_function=None):
    if isinstance(function, list):
        return extract_cross_section_from_list(function, points, filter_function=filter_function)
    if isinstance(function, Expression):
        if times is None:
            raise ValueError("times must be specified for expression cross section")
        return extract_cross_section_from_expression(function,
                                                     points, times)
 

def extract_cross_section_from_list(functions, points,filter_function=None):
    npoints = len(points)
    nt = len(functions)
    value_dim = functions[0].value_dimension(0)
    values = np.ndarray((nt, npoints, value_dim))
    if filter_function is None:
        filter_function = Expression("1", degree=0)
    for k, p in enumerate(points):
        for i,f in enumerate(functions):
            try:
                values[i, k, :] = f(p)*filter_function(p)
            except Exception as err:
                #print(err)
                #print(p.array())
                values[i, k, :] = 0.0
            for m in range(value_dim):
                values[i, k, m] = MPI.sum(MPI.comm_world, values[i, k, m])
    return values

def extract_cross_section_from_expression(expression, points, times):
    npoints = len(points)
    nt = len(times)
    try:
        value_dim = expression[0].value_dimension(0)
    except:
        value_dim = 1
    values = np.ndarray((nt, npoints, value_dim))
    for k, p in enumerate(points):
        for i,t in enumerate(times):
            expression.t = t
            values[i, k, :] = expression(p) 
    return values



def compute_order(error, h):
    h = np.array(h)
    err_ratio = np.array(error[:-1]) / np.array(error[1:])
    return np.log(err_ratio)/np.log(h[:-1] / h[1:])