import operator
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import random

def step(x):
    if x>0: return 1
    else: return 0

def sigmoid(x, k):
    return 1 / (1 + pow(math.e, -k*x))

def random_vector():
    w=tuple()
    for i in range(19):
        w+=tuple([random.random()*2-1])
    return w

def E(X, ww):
    error=0
    rt=sigmoid(ww[17],1)
    for val in X:
        if circleeqn(val[0],val[1])<=1: y=1
        else: y=0
        if c_percep(val[0],val[1],ww)<=rt:myx=1
        else: myx=0
        error+= math.pow(y-myx,2)
    return error

def magnitude(grad):
    sum=0
    for var in grad:
        sum+=var**2
    return math.sqrt(sum)

def one_d_minimize(f, left, right, tol):
    if right-left<=tol: return (left+right)/2
    one3= left+((right-left)/3)
    two3=left+((right-left)*2/3)
    if f(one3)> f(two3): return one_d_minimize(f, one3, right, tol)
    else: return one_d_minimize(f, left, two3, tol)

def grad_desc_with_line_search(f, df, start, tol):
    location = start
    while magnitude(df(location))>tol:
        direction = df(location)
        closed_f= make_funct(f, location, direction)
        lmda = one_d_minimize(closed_f,0,1,10**-8)
        location = [l - lmda * d for l, d in zip(location, direction)]
    return location

def make_funct(f, loc,dir):
    def funct(a):
        location = [ l - a*d for l, d in zip(loc, dir)]
        return f(location)
    return funct

def sin_func(x):
    return math.sin(x)+math.sin(3*x)+math.sin(4*x)

def my_func(A):
    x,y=A
    return 4*x**2-3*x*y+2*y**2+24*x-20*y
    #return (1-y)**2+100*(x-y**2)**2

def my_df(A): # partial df w/ respect to x, partial df w/ respect to y)
    x,y= A
    return  (8*x-3*y+24,-3*x+4*y-20)
    #return  (200*x-200*y**2, 400*y**3-400*x*y+2*y-2)

print(grad_desc_with_line_search(my_func, my_df, (0,0), 10**-8))

