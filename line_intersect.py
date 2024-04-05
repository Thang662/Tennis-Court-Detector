from sympy import Line
import sympy
import numpy as np

def line_intersection(line1, line2):
    """
    Find 2 lines intersection point
    """
    l1 = Line((line1[0], line1[1]), (line1[2], line1[3]))
    l2 = Line((line2[0], line2[1]), (line2[2], line2[3]))

    intersection = l1.intersection(l2)
    point = None
    if len(intersection) > 0:
        if isinstance(intersection[0], sympy.geometry.point.Point2D):
            point = intersection[0].coordinates
    return point

def get_intersect(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return None
    return (x/z, y/z)