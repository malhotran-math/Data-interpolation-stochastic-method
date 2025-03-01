#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:11:00 2025

@author: malhotran
"""

import numpy as np 
import random 
import math 
import matplotlib.pylab as plt 
import time 
from scipy.spatial import ConvexHull, distance




def rect_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def polar_dist(r1, theta1, r2, theta2):
    """Compute distance between two polar points"""
    return rect_dist(r1 * math.cos(theta1), r1 * math.sin(theta1), r2 * math.cos(theta2), r2 * math.sin(theta2))

t = time.time() 
#variables
varquant = 500 #number of elements to compute variance over 
n = 3000 #number of rays 
m = np.add(3000,np.zeros(n)) # list of size n, giving the number of points on each ray 
alpha = 0.3 #biasing against randomness 
beta = 0.5 #biasing against neighbour rays 

#data 
rlist = [[i+1 for i in range(int(m[j]))] for j in range(n)]  #radial position of points along rays
Theta_ini = np.random.uniform(low=0.0,high=2*np.pi,size=n)
minang = np.min(Theta_ini)
Thetalist = np.add(Theta_ini,-minang) 
Thetalist = np.array(Thetalist).tolist()
Thetalist.sort() #angles of the rays
H_ini = [[random.random() for _ in range(int(m[j]))] for j in range(n)] #interpolated data within high resolution domain

#print(H_ini[2])
#print(np.shape(H_ini)) 

'''def one_amendment(H,r,Theta): 
    mid_angle = min(Theta, key=lambda x:abs(x-np.pi)) #closest angle to pi 
    pos_midangle = Theta.index(mid_angle) #position of the closest angle in the list
    
    h0 = H[0][-1] - H[0][-2] #gradient of the first boundary point
    var0 = np.var(H[0][-varquant:]) #variance of the first ray
    a0new = H[0][-1] + alpha*h0 + (1-alpha)*(np.random.normal(0,var0)) #new point for the first ray
    H[0].append(a0new)
    
    hmid = H[pos_midangle][-1] - H[pos_midangle][-2] #gradient of the mid boundary point
    varmid = np.var(H[pos_midangle][-varquant:]) #variance of the middle ray
    amidnew = H[pos_midangle][-1] + alpha*(hmid) + (1-alpha)*(np.random.normal(0,varmid)) #new point for the middle ray
    H[pos_midangle].append(amidnew)
    
    for i in np.arange(1,n): 
        if i == pos_midangle:
            waste = 0
        else:
            h_temp = H[i][-1] - H[i][-2] #gradient of the point
            var_temp = np.var(H[i][-10:])  #variance of the ith ray 
            a_temp = H[i][-1] + alpha*h_temp + (1-alpha)*(np.random.normal(0,var_temp)) 
            
            d2first = polar_dist(r[-1],0,r[-1],Theta[i])
            d2mid = polar_dist(r[-1],mid_angle,r[-1],Theta[i]) 
            dtot = d2first+d2mid 
            a_influence = d2first*a0new + d2mid*amidnew 
            a_infl = a_influence/dtot 
        
            a_final = beta1*a_temp + (1-beta1)*a_infl
            H[i].append(a_final)
        
    r.append(r[-1]+1)
    
    return H,r,Theta,pos_midangle''' 

iterations = 3000 #number of points to be extrapolated per ray 

def multi_amendment(H,r,Theta,n,m,iterations): 
    mid_angle = min(Theta, key=lambda x:abs(x-np.pi)) #closest angle to pi 
    pos_midangle = Theta.index(mid_angle) #position of the closest angle in the list
    
    for itern in range(iterations):
        h0 = H[0][-1] - H[0][-2] #last gradient for the first ray 
        var0 = np.var(H[0][-varquant:]) #variance for the first ray
        a0new = H[0][-1] + alpha*h0 + (1-alpha)*(np.random.normal(0,var0)) #new point for the first ray
        H[0].append(a0new) 
        r[0].append(r[0][-1]+1)  
    
        hmid = H[pos_midangle][-1] - H[pos_midangle][-2] #last gradient for the mid ray 
        varmid = np.var(H[pos_midangle][-varquant:]) #variance for the middle ray
        amidnew = H[pos_midangle][-1] + alpha*(hmid) + (1-alpha)*(np.random.normal(0,varmid)) #new point for the middle ray
        H[pos_midangle].append(amidnew)
        r[pos_midangle].append(r[pos_midangle][-1]+1)  
    
    
        for i in np.arange(1,n): 
            if i!=pos_midangle: 
                h_temp = H[i][-1] - H[i][-2] #gradient for the ith ray 
                var_temp = np.var(H[i][-10:])  #variance for the ith ray 
                a_temp = H[i][-1] + alpha*h_temp + (1-alpha)*(np.random.normal(0,var_temp)) 
                
                d2first = polar_dist(r[0][-1],0,r[i][-1],Theta[i]) #distance from the first ray
                d2mid = polar_dist(r[pos_midangle][-1],mid_angle,r[i][-1],Theta[i])  #distance from the mid ray
                dtot = d2first+d2mid 
                a_influence = d2first*a0new + d2mid*amidnew 
                a_infl = a_influence/dtot #weighted influence from the first and middle ray 
        
                a_final = beta1*a_temp + (1-beta1)*a_infl #new point for the ith ray
                H[i].append(a_final)
                r[i].append(r[i][-1]+1) 
    
        
   
    return H,r,Theta,pos_midangle

'''amendment adds iterations many points per ray, where n is the number of rays, 
and m is an n-dimensional list with m[i] being the number of points on the ith ray.

r is a list of lists, where r[i][j] is the radial position of the jth point on the
ith ray.

Theta is an n-dimensional list, where Theta[0]=0, and Theta[i] is the angle from the 
ith ray to the 0th ray. 

H is a list of lists, where H[i][j] is the height of the jth interpolated point on the
ith ray. 
'''
def amendment(H,r,Theta,n,m,iterations):  
    for itern in range(iterations): #number of steps to add for each ray 
        for i in range(n): #looping over all rays  
            '''Stochastic evolution step with bias from previous points on the ith ray''' 
            h_temp = H[i][-1] - H[i][-2] #gradient for the ith ray 
            var_temp = np.var(H[i][-varquant:])  #variance for the ith ray 
            a_temp = H[i][-1] + alpha*h_temp + (1-alpha)*(np.random.normal(0,var_temp)) 
                
            '''interpolation from last points of the neighbouring rays'''
            d2left = polar_dist(r[i-1][-1],Theta[i-1],r[i][-1],Theta[i]) #distance from left ray
            d2right = polar_dist(r[i][-1],Theta[i],r[np.mod(i+1,n)][-1],Theta[np.mod(i+1,n)])  #distance from right ray
            dtot = d2left+d2right 
            a_influence = d2right*H[i-1][-1] + d2left*H[np.mod(i+1,n)][-1]  
            a_infl = a_influence/dtot #weighted influence from the first and middle ray 
        
            '''new point for ith ray'''    
            a_final = beta*a_temp + (1-beta)*a_infl #new point for the ith ray
            H[i].append(a_final)
            r[i].append(r[i][-1]+1)
    
    return H,r,Theta




Hnew,rnew,Theta = amendment(H_ini, rlist, Thetalist,n,m, iterations)


'''for i in range(n):
    if i%100==0:   
        plt.plot(rnew[i][-35:],Hnew[i][-35:])    
    
plt.legend()
        '''
        
print(time.time()-t)     

t = time.time() 

    
    

'''n = 60000
    
H = [[random.random() for i in range(n)], [random.random() for i in range(n)], [random.random() for i in range(n)]]

   
print(time.time()-t)     


'''


