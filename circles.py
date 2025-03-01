#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 16:47:03 2025

@author: malhotran
"""

import numpy as np 
import random 
import math 
import matplotlib.pylab as plt 
import time 
from scipy.spatial import ConvexHull, distance


def point_to_segment_distance(point, seg_start, seg_end):
    """
    Computes the shortest distance from a point to a line segment.
    """
    p = np.array(point)
    a = np.array(seg_start)
    b = np.array(seg_end)
    
    # Project point onto the line (parameterized by t)
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / np.dot(ab, ab)
    
    # Clamp t to the segment [0,1] (ensuring projection stays on segment)
    t = max(0, min(1, t))
    
    # Compute closest point on segment
    closest = a + t * ab
    return np.linalg.norm(p - closest)

'''L is a list of 3 lists, L=[[X],[Y],[Z]]. c is the calculated focal point. '''
def rmax(L,c):
    points = np.array(list(zip(L[0], L[1])))
    cx, cy = c

    # Compute Euclidean distances from center to all points
    distances = np.linalg.norm(points - np.array(c), axis=1)

    # The outcircle's radius is the max distance
    return np.max(distances)

def rmin(L,c):
    points = np.array(list(zip(L[0], L[1])))
    cx, cy = c
    hull = ConvexHull(points)
    hull_vertices = points[hull.vertices]
    if any(np.dot(eq[:-1], [cx, cy]) + eq[-1] > 0 for eq in hull.equations):
        return None 
    min_distance = float('inf')
    for i in range(len(hull_vertices)):
        p1 = hull_vertices[i]
        p2 = hull_vertices[(i + 1) % len(hull_vertices)]  # Wrap around to form edges

        # Find shortest distance from center to edge (p1, p2)
        dist = point_to_segment_distance(c, p1, p2)
        min_distance = min(min_distance, dist)

    return min_distance
    


