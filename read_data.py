#!/usr/bin/env python3
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import time
import dill as pickle
from scipy.interpolate import CubicSpline
from ctypes import c_float
import basic_functions
import grid_classes
from PIL import  Image


def read_field(fname:str) :
    """
    Read the gridded field from the csv file
    """

    print(f"read {fname}")
    tstart = time.time()
    pfilename = os.path.join(os.path.dirname(__file__),"..","..","toydata",fname + ".pickle")
    if os.path.isfile(pfilename):
        with open(pfilename,'rb') as f:
            values = np.array(pickle.load(f),dtype=c_float)
        print(f"    read pickle: {time.time()-tstart}"); tstart = time.time()
    else:
        filename = os.path.join(os.path.dirname(__file__),"..","..","toydata",fname + ".csv")
        values =  np.genfromtxt(filename, delimiter=',')
        print(f"    read csv: {time.time()-tstart}"); tstart = time.time()
        with open(pfilename,'wb') as f:
            pickle.dump(values,f)
        print(f"    write pickle: {time.time()-tstart}"); tstart = time.time()

    x = np.unique(values[:,0])
    y = np.unique(values[:,1])
    x_d = { xi: i for i,xi in enumerate(x)}
    y_d = { yi: i for i,yi in enumerate(y)}
    print(f"    coord struct: {time.time()-tstart}"); tstart = time.time()
    i = np.array( [x_d[vi] for vi in values[:,0]], dtype = int)
    j = np.array( [y_d[vi] for vi in values[:,1]], dtype = int)
    ij = i*y.shape[0] + j
    v = np.full((x.shape[0]*y.shape[0]), fill_value = np.nan, dtype=c_float)
    v[ij] = values[:,2]
    v = v.reshape( (x.shape[0], y.shape[0]))
    print(f"    regrid: {time.time()-tstart}"); tstart = time.time()

    return x, y, v

if __name__ == "__main__":
    dtheta = 2*math.pi / 200
    dr = 150
    r_min = 0
    r_max = 30000

    xf, yf, vf = read_field("high-res-bathymetry")
    xc, yc, vc = read_field("low-res-bathymetry")


    Rf = grid_classes.RectangularGrid(xf, yf, vf)
    Rc = grid_classes.RectangularGrid(xc, yc, vc)

    fine_data = basic_functions.generate_data_set(xf, yf, vf)
    fine_center = basic_functions.centerpoint(fine_data)
    print(fine_center)

    Pf = grid_classes.polar_grid_from_rectangular(Rf, dtheta, dr, r_min, r_max)
    # UG = grid_classes.polar_to_unsorted_grid(Pf)

    Pf.plot()
    Rf.plot()





def interpolate_value(xf, yf, vf, x0, y0):
    ''' Compute the interpolated value at a point, taking as input the data (xf, yf, vf) and the point at which
        we want to know the interpolated value (x0, y0)
    '''
    vf_modified = np.where(np.isnan(vf), 0, vf)

    cxf = CubicSpline(xf, vf_modified, bc_type='clamped', extrapolate=False)
    vinterf = cxf(x0)

    cyf = CubicSpline(yf, vinterf, axis=1, bc_type = 'clamped', extrapolate=False)
    vc2f = cyf(y0)
    return vc2f


def laplacian():
    cx = CubicSpline(xc, vc, bc_type='clamped', extrapolate=False)
    vinter = cx(xf)

    cy = CubicSpline(yc, vinter, axis=1, bc_type = 'clamped', extrapolate=False)
    vc2f = cy(yf)
    dvf = vf - vc2f
    dvf[np.isnan(vf)] = 0

    fill_in = np.where(np.isnan(vf))[0]

    for i in range(10):
        if i%100 == 0:
            print(f"iter {i}")
        dvf1 = dvf.copy()
        dvf1[1:-1,1:-1] = (dvf[2:,1:-1] + dvf[:-2,1:-1] + dvf[1:-1,2:] + dvf[1:-1,:-2])/4
        dvf = np.where(np.isnan(vf), dvf1, dvf)


    # https://github.com/abhinavs95/photo-uncrop


    plt.subplot(221)
    plt.contourf(yf,xf,dvf)
    #plt.contourf(yc,xc,vc)
    plt.colorbar()
    plt.title("extension")
    #plt.title("coarse map")

    plt.subplot(222)
    plt.contourf(yf,xf,vf)
    plt.colorbar()
    plt.title("fine map")

    plt.subplot(223)
    vc2f[np.isnan(vf)] = np.nan
    plt.contourf(yf,xf,vc2f)
    plt.title("coarse map, interp")
    plt.colorbar()

    plt.subplot(224)
    plt.contourf(yf,xf,vf-vc2f)
    plt.colorbar()
    plt.title("fine - coarse map")
    # plt.show()