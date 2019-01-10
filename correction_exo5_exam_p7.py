# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 19:32:03 2019

@author: Ibrwa
"""


##########################################################################################
# Application : find the extremum of a function                                          #
# Program name: correction_exo5_exam_p7.py                                               #
# Description : This program find the extremum of function by using the rule of derivate:#
#               if x0 is a extremum of an function then f'(x0) = 0                       #
#________________________________________________________________________________________#
# Developped     : Under spyder in Windows                                               #
#________________________________________________________________________________________#
# Creation    : 01/01/2019 by Ibrwa                                                      #    
# Last Updated: 07/01/2019                                                               #
#________________________________________________________________________________________#


# import th library numpy and scipy: we import scipy for to find extremum directly and for comparison

import numpy as np 
from scipy import optimize
import matplotlib.pyplot as plt

### Create a global variable pi 

pi = np.pi;

### Create the main function : f(x) = exp(-x**2+sin(x-pi/3))

def calc_f(x):
    """
    This function returns the value exp(-x**2+sin(x-pi/3))
    """
    return np.exp(-x**2 + np.sin(x - (pi/3)))

## calcul de la derivee seconde en x
## definition de la fonction derivee de f 
def calc_f_prime(x):
    """
    This function returns the value exp(-x**2+sin(x-pi/3))
    Rappel : e(U)' = U'e(U) et sin(U)' = U'cos(U)
    """
    return (-2*x + np.cos(x - (pi/3))) * (np.exp(-x**2 + np.sin(x - (pi/3))))


## definition of the function of derivate of g 
def calc_g(x):
    """
    This function returns the value sin(exp(-x))
    """
    return np.sin(np.exp(-x))

## definition of the function of derivate of g 
def calc_g_prime(x):
    """
    This function returns the value sin(exp(-x))
    Rappel: sin(U)' = U' cos(U)
    """
    return -np.exp(-x)*np.cos(np.exp(-x))


### create another function that approximates the derivate of x0 by using the definition of tengeante formula 

def calc_deriv(function,x0,delta):
    """ 
    This function approximates the derivates by using the formula of tengente approximation
    Approximation de la derivee en utlisant la formule du taux d'accroissement
    """
    return (function(x0+delta)-function(x0-delta))/(2*delta)


def plot_function(xlims=(-2.,2.0),math_symbol="",f=calc_f):
    """
    The puprose of this function is to plot a function function f within xlims boundary and with a title 
    parameters: xlims = tuple of boundary of x axis 
                label = title of the function --> can contains mathematical functions 
                math_symbol = Mathematical symbols to all to the label 
    """
    # Set 100 points between xlims
    x = np.linspace(xlims[0],xlims[1],num = 100)
    # Set size of figure
    plt.figure(figsize=(8,8))
    # plot x and y 
    plt.plot(x,f(x))
    plt.title("Fonction : " +  math_symbol)
    


    

##### Create the function to find the maximum #####
def find_extremum_tengente_approx(xlims,f,delta):
    """
    This function find extremum of function finding the point x0 in which f'(x0) = 0
    input parameters: 
        xlims = interval in which we search the extremum 
        f = function 
        delta = used for the formula to approximate the tengente
    """
    ## Create the vector of X 
    ### Create the number of points in the intervalle 
    num = (xlims[1]-xlims[0])/delta
    x = np.linspace(xlims[0],xlims[1],num=num,endpoint = True)
    ## calcate the vector of derivate of x using tengente approximation 
    dx = calc_deriv(f,x,delta)
    ### find the extremum of dx veector by using the idx of value most close to 0 using abs function 
    id_extremum = np.abs(dx).argmin()
    ### reutnr the tuple x, fx
    return x[id_extremum],f(x[id_extremum])


###### Pour la fonction f#######
    

## plot the function: plot_function
plot_function(math_symbol=r"$e^{-x^{2} + sin(x - \pi/3)}$")

x0_f, f_x0 = find_extremum_tengente_approx((-2,2),calc_f,0.0001)

print("La fonction f(x) a minimum/maximum à x = " + str(round(x0_f,3)))

## plottig f(x) with vertical reference at x = x0

#### We plot again the figure to check the value of maximum 
# Set size of figure
plt.figure(figsize=(8,8))
# plot x and y 
plt.plot(np.linspace(-2,2),calc_f(np.linspace(-2,2)))
plt.xlim(xmin = -2.0, xmax = 2.0 )
plt.ylim(ymin = 0.0, ymax = 0.55 )
plt.axvline(x = x0_f,linewidth = 4, color='r',ymax = 0.845)
plt.annotate(s = "x = " + str(round(x0_f,3)),xy=(x0_f-0.15,0.5))
# Set title with mathematical symbols 
plt.title("Fonction f(x) = " + r"$e^{-x^{2} + sin(x - \pi/3)}$")
# display the graphic 
plt.show() 


###### Find maximum for g #######

plot_function(xlims=(-1.5,1.5),math_symbol=r"$sin(e^{-x})$",f=calc_g)

x0_g,g_x0 = find_extremum_tengente_approx((-1.5,1.5),calc_g,0.0001)

print("La fonction g(x) a minimum/maximum à x = " + str(round(x0_g,3)))


##################### Verification avec scipy ###################################
################################################################################

#### Evaluate the maximum with optimize function 
### to find the maximum we can use the minium of -f 
def calc_f_neg(x):
    """
    This function returns the opposite of g function
    """
    return -1.0*(np.exp(-x**2 + np.sin(x - (pi/3))))

def calc_g_neg(x):
    """
    This function returns the opposite if g function 
    """
    return -1.0 * np.sin(np.exp(-x))

# Get the maximum by using the negative of calc_f
x0_f_scipy = optimize.fminbound(calc_f_neg,-2.0,2.0)
print("The maximum value of f with scipy is : " + str(x0_f_scipy))
# Get the maximum by using the negative of calc_f
x0_g_scipy = optimize.fminbound(calc_g_neg,-1.5,1.5)
print("The maximum value of g with scipy is : " + str(x0_g_scipy))



########################################
############## Derivee seconde #########
########################################


##### on regarde s'il s'agit d'un maximum ou d un minium avec le calcul de la derivee  ###



def check_min_max(f_derivee,x,delta):
    """
    This function checks if the extremum is minimum or maximum 
    paramters: 
        f_derivee : function f derivate in which we calculate the derivate
        x : point in which we want to see if the extremum is minimum or maximum
        delta : used to applied the formula of tengente approxilmation 
    """
    x_seconde = calc_deriv(f_derivee,x,delta)
    if x_seconde > 0:
        print("Il s'agit d'un minimum car la derivee seconde est > 0")
    elif x_seconde < 0:
        print("Il s'agit d'un maximum car la derivee seconde est < 0")
    

# pour f
check_min_max(calc_f_prime,x0_f,0.0001)
# pour g 
check_min_max(calc_g_prime,x0_g,0.0001)
################################### Fin du script ################################
