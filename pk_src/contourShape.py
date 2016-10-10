"""
Module to find and process the contour shape of an edge set on the plane.
Given a set of triangles on the plane, calculate the outline of the contour (meaning intersections of edges will create new points and the result will be a set of polygons)

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


class Point:

    # coordinates
    x = 0
    y = 0
    # index in point field
    ind = -1
    # created point (for intersection or boolean)
    created = True
    #neighbor points in order of x coordinates
    prev = None
    next = None

    def __init__(self, x, y, ind, created, prev, next):
        self.x = x
        self.y = y
        self.ind = ind
        self.created = created
        self.prev = prev
        self.next = next


def turn(p1,p2,p3):
    A = (p3[1]-p1[1])*(p2[0]-p1[0])
    B = (p2[1]-p1[1])*(p3[0]-p1[0])
    return 1 if (A > B+EPSILON) else -1 if (A+EPSILON < B) else 0


def isIntersect(p1,p2,p3,p4):
    return (turn(p1, p3, p4) != turn(p2, p3, p4)) & (turn(p1, p2, p3) != turn(p1, p2, p4))


def getIntersectionPoint(l1p1,l1p2,l2p1,l2p2):
    u1 = ((float(l2p2[0])-l2p1[0])*(l1p1[1]-l2p1[1]) - (l2p2[1]-l2p1[1])*(l1p1[1]-l2p1[1])) / ((l2p2[1]-l2p1[1])*(l1p2[0]-l1p1[0]) - (l2p2[0]-l2p1[0])*(l1p2[1]-l1p1[1]))
    if u1<0 | u1>1:
        return None
    else:
        return l1p1+u1*(l1p2-l1p1)


def isOuterEdge(points, edge, trisForEdge):
    # edge is an outer edge if referenced by exactly one face/triangle
    return len(trisForEdge) == 1


def getOuterEdges(points, triEdgesRef):
    outerEdges = set([])
    for edg in triEdgesRef:
        if isOuterEdge(points, edg, triEdgesRef[edg]):
            outerEdges.add(edg)

    return outerEdges


def getPolygon(ptsCord, edges):
    pts = {}
    for edge in edges:
        for pt in edge:
            if pt not in pts:
                pts[pt] = Point(ptsCord[pt][0], ptsCord[pt][1], pt, False, None, None)

        edgeList = list(edge)

        pt0 = pts[edgeList[0]]
        pt1 = pts[edgeList[1]]

        if (pt0.next is not None) & (pt1.next is None):
            pt0, pt1 = pt1, pt0

        if pt0.next is None:
            if pt1.prev is not None:
                #reverse the order of the list
                pt = pt1
                while pt is not None:
                    pt.prev, pt.next = pt.next, pt.prev
                    pt = pt.next

        else:
            # reverse the order of the list
            pt = pt0
            while pt is not None:
                pt.prev, pt.next = pt.next, pt.prev
                pt = pt.prev


        pt1.prev = pt0
        pt0.next = pt1

    poly = []

    pt =pt0

    while pt not in poly:
        poly.append(pt)
        pt = pt.next

    return poly





















