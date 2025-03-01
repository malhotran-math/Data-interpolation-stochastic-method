#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 18:01:06 2025

@author: malhotran
"""

import grid_classes
import read_data
import basic_functions
import numpy as np

import random 
import math 

def import_data(fname_fine, fname_coarse):
    xf, yf, vf = read_data.read_field(fname_fine)
    xc, yc, vc = read_data.read_field(fname_coarse)

    Rf = grid_classes.RectangularGrid(xf, yf, vf)
    Rc = grid_classes.RectangularGrid(xc, yc, vc)

    return Rf, Rc

def radial_slices_to_polar_grid(radial_slices, thetas, rs, center_x, center_y):
    n_rs = len(rs)
    n_thetas = len(thetas)
    v = []
    for rad_slice, start in radial_slices:
        v_rad = [np.nan for k in range(n_rs)]
        v_rad[start:start+len(rad_slice.averaged_values)] = rad_slice.averaged_values
        v.append(v_rad)
    Pa = grid_classes.PolarGrid(thetas, rs, v, center_x, center_y)
    return Pa



def rect_dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)


def polar_dist(r1, theta1, r2, theta2):
    """Compute distance between two polar points"""
    return rect_dist(r1 * math.cos(theta1), r1 * math.sin(theta1), r2 * math.cos(theta2), r2 * math.sin(theta2))



'''amendment adds iterations many points per ray, where n is the number of rays, 
and m is an n-dimensional list with m[i] being the number of points on the ith ray.

r is a list of lists, where r[i][j] is the radial position of the jth point on the
ith ray.

Theta is an n-dimensional list, where Theta[0]=0, and Theta[i] is the angle from the 
ith ray to the 0th ray. 

H is a list of lists, where H[i][j] is the height of the jth interpolated point on the
ith ray. 
'''
def extrapolate(Pf, iterations):
    H = [[v for v in ray] for ray in Pf.v]
    r = Pf.rs
    Theta = Pf.thetas
    n = len(Pf.thetas)
    m = [np.where(np.isnan(Pf.v[k]))[0][0] for k in range(n)]  
    varquant = 10
    alpha =0.5
    beta = 0.5
    for itern in range(iterations): #number of steps to add for each ray 
        for i in range(n): #looping over all rays  
            '''Stochastic evolution step with bias from previous points on the ith ray''' 
            h_temp = H[i][m[i]-1] - H[i][m[i]-2] #gradient for the ith ray 
            #var_temp = np.var(H[i][m[i]-varquant:m[i]-1])  #variance for the ith ray 
            a_temp = H[i][m[i]-1] + alpha*h_temp + (1-alpha)*(np.random.normal(0,2)) 
                
            '''interpolation from last points of the neighbouring rays'''
        
            d2left = polar_dist(1,Theta[i-1],1,Theta[i]) #distance from left ray
            d2right = polar_dist(1,Theta[i],1,Theta[np.mod(i+1,n)])  #distance from right ray
            dtot = d2left+d2right 
            a_influence = d2right*H[i-1][m[i-1]-1] + d2left*H[np.mod(i+1,n)][m[np.mod(i+1,n)]-1]  
            a_infl = a_influence/dtot #weighted influence from the first and middle ray 
        
            ''''new point for ith ray'''    
            #a_final = beta*a_temp + (1-beta)*a_infl #new point for the ith ray
            a_final = a_temp 
            H[i][m[i]] = a_temp 
            m[i] +=1
            #assert not np.isnan(h_temp)
            #assert not np.isnan(var_temp)
            #assert not np.isnan(a_temp)
            #assert not np.isnan(d2left)
            #assert not np.isnan(d2right)
            #assert not np.isnan(dtot)
            #assert not np.isnan(a_influence)
            #assert not np.isnan(a_infl)
            #assert not np.isnan(a_final)
    return H


def construct_polar_grids(Rf, Rc, n_rays, dr, r_min, r_max):
    Pf = grid_classes.polar_grid_from_rectangular(Rf, n_rays, dr, r_min, r_max)
    cx = Pf.center_x
    cy = Pf.center_y
    Pc = grid_classes.polar_grid_from_rectangular(Rc, n_rays, dr, r_min, r_max, center=[cx, cy])
    return Pf, Pc



def construct_radial_slices(Pf,Pc,H,inner_rel_dist, outer_rel_dist):
    inner_rel_steps = int(np.ceil(inner_rel_dist/dr))
    outer_rel_steps = int(np.ceil(outer_rel_dist/dr))
    radial_slices = []
    for i in range(n_rays):
        rayf = Pf.v[i]
        rayc = Pc.v[i]
        cutoff = rayf.index(np.nan)
        #print(cutoff, inner_rel_steps, outer_rel_steps)
        rs = Pf.rs[cutoff-inner_rel_steps: cutoff + outer_rel_steps]
        fine_values = rayf[cutoff-inner_rel_steps: cutoff]
        coarse_values = rayc[cutoff-inner_rel_steps: cutoff + outer_rel_steps]
        extrapolated_values = H[i][cutoff-inner_rel_steps: cutoff + outer_rel_steps] 
        rad_slice = grid_classes.RadialSlice(rs, fine_values, coarse_values, extrapolated_values=extrapolated_values)
        radial_slices.append((rad_slice, cutoff-inner_rel_steps))
    return radial_slices

def merge_fine_average_coarse(Pf, Pa, Pc):
    n_thetas = len(Pf.thetas)
    v_new = []
    for i in range(n_thetas):
        v_new_theta = []
        start = 0
        while np.isnan(Pa.v[i][start]):
            start += 1
        end = start
        while not np.isnan(Pa.v[i][end]):
            end += 1
        for j in range(len(Pc.rs)):
            if j < start:
                v_new_theta.append(Pf.v[i][j])
            elif j < end:
                v_new_theta.append(Pa.v[i][j])
            else:
                v_new_theta.append(Pc.v[i][j])
        v_new.append(v_new_theta)
    return grid_classes.PolarGrid(Pf.thetas, Pc.rs, v_new, Pf.center_x, Pf.center_y)


def merge_fine_coarse(Pf, Pc):
    n_thetas = len(Pf.thetas)
    v_new = []
    for i in range(n_thetas):
        v_new_theta = []
        end = Pf.v[i].index(np.nan)
        for j in range(len(Pc.rs)):
            if j < end:
                v_new_theta.append(Pf.v[i][j])
            else:
                v_new_theta.append(Pc.v[i][j])
        v_new.append(v_new_theta)
    return grid_classes.PolarGrid(Pf.thetas, Pc.rs, v_new, Pf.center_x, Pf.center_y)



if __name__ == "__main__":
    deg = 2
    gamma = 1
    r_min = 0
    r_max = 80000
    dr = 100
    n_rays = 150
    inner_rel_distance = 2000
    outer_rel_distance = 10000
    cmap = 'YlGnBu' 

    Rf, Rc = import_data("high-res-bathymetry", "low-res-bathymetry")
    Pf, Pc = construct_polar_grids(Rf, Rc, n_rays, dr, r_min, r_max)
    
    Pm = merge_fine_coarse(Pf,Pc) 
    Pm.plot(cmap=cmap)
    
    iterations = int(np.ceil(outer_rel_distance/dr))
    Pf.plot(cmap=cmap) 
    H=extrapolate(Pf, iterations)
    Pf.plot(cmap=cmap)
    
    radial_slices = construct_radial_slices(Pf, Pc, H, inner_rel_distance, outer_rel_distance)
    
    
    for rad_slice, cutoff in radial_slices:
        rad_slice.averaged_values = rad_slice.weighted_average(gamma)
    Pa = radial_slices_to_polar_grid(radial_slices, Pf.thetas, Pf.rs, Pf.center_x, Pf.center_y)
    Ptotal = merge_fine_average_coarse(Pf, Pa, Pc)
    Pa.plot(cmap=cmap)
    Ptotal.plot(cmap=cmap)
    radial_slices[int(n_rays/2)][0].plot()





