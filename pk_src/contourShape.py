"""
Module to find the contour shape of an edge set on the plane.
Given a set of connected edges on the plane, calculate the outline of the contour (meaning intersections of edges will create new points and the result will be a set of polygons)

Based on the paper `"Fast tetrahedron-tetrahedron overlap algorithm" <http://vcg.isti.cnr.it/Publications/2003/GPR03/fast_tetrahedron_tetrahedron_overlap_algorithm.pdf>`_ by F. Ganovelli, F. Ponchio and C. Rocchini.

Uses two lists of vertices for each tetrahedra. The class function Collision_tet_tet.check() returns True iff the two given tetrahedra actually intersect (or at least have common vertices).
Before calling 'Collision_tet_tet.check()' the two tetrahedra need to be set with the methods **setV1**, **setV2** or **setV1V2**.
"""


import numpy as np
import operator

dot = lambda x,y: sum(map(operator.mul, x, y))
"""dot product as lambda function to speed up calculations"""
cross = lambda x,y: map(operator.sub, [x[1]*y[2], x[2]*y[0], x[0]*y[1]], [x[2]*y[1], x[0]*y[2], x[1]*y[0]])
"""cross product as lambda function to speed up calculations"""

EPSILON = 2e-15

class findContourShape:
    def getContourShape(pts):
        return pts

    def Turn(p1,p2,p3):
        A = (p3[1]-p1[1])*(p2[0]-p1[0])
        B = (p2[1]-p1[1])*(p3[0]-p1[0])
        return 1 if (A > B+EPSILON) else -1 if (A+EPSILON < B) else 0

    def isIntersect(p1,p2,p3,p4):
        return (Turn(p1, p3, p4) != Turn(p2, p3, p4)) & (Turn(p1, p2, p3) != Turn(p1, p2, p4))

    def getIntersectionPoint(l1p1,l1p2,l2p1,l2p2):
        u1 = ((float(l2p2[0])-l2p1[0])*(l1p1[1]-l2p1[1]) - (l2p2[1]-l2p1[1])*(l1p1[1]-l2p1[1])) / ((l2p2[1]-l2p1[1])*(l1p2[0]-l1p1[0]) - (l2p2[0]-l2p1[0])*(l1p2[1]-l1p1[1]))
        if u1<0 | u1>1:
            return None
        else:
            return l1p1+u1*(l1p2-l1p1)

    