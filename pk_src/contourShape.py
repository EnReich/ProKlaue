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


class SearchTreeNode:
    key = 0
    val = None
    parent = None
    left = None
    right = None

    def __init__(self, key, val, parent, left, right):
        self.key=key
        self.val=val
        self.parent=parent
        self.left =left
        self.right = right


class SearchTree:
    root = None

    def search(self, key):
        node = self.root
        while node is not None:
            if node.key == key:
                return node
            else:
                if key < node.key:
                    node = node.left
                else:
                    node = node.right
        return None

    def insertRec(self, lastNode, nodeToInsert):
        if nodeToInsert.key > lastNode.key:
            if lastNode.right is not None:
                self.insertRec(lastNode.right, nodeToInsert)
            else:
                lastNode.right = nodeToInsert
                nodeToInsert.parent = lastNode
        else:
            if lastNode.left is not None:
                self.insertRec(lastNode.left, nodeToInsert)
            else:
                lastNode.left = nodeToInsert
                nodeToInsert.parent = lastNode

    def insert(self, nodeToInsert):
        if self.root is not None:
            self.insertRec(self.root, nodeToInsert)
        else:
            self.root = nodeToInsert

    def delete(self, nodeToDelete):
        #no children just delete
        if (nodeToDelete.right is None) & (nodeToDelete.left is None):
            if nodeToDelete.parent is not None:
                if nodeToDelete.key < nodeToDelete.parent.key:
                    nodeToDelete.parent.left = None
                else:
                    nodeToDelete.parent.right = None
            if self.root is nodeToDelete:
                self.root=None

        #one children just give it to parent
        elif nodeToDelete.right is None:
            if nodeToDelete is not self.root:
                if nodeToDelete.left.key < nodeToDelete.parent.key:
                    nodeToDelete.parent.left = nodeToDelete.left
                else:
                    nodeToDelete.parent.right = nodeToDelete.left

                nodeToDelete.left.parent = nodeToDelete.parent
            else:
                self.root = nodeToDelete.left
                self.root.parent = None

        elif nodeToDelete.left is None:
            if nodeToDelete is not self.root:
                if nodeToDelete.right.key < nodeToDelete.parent.key:
                    nodeToDelete.parent.left = nodeToDelete.right
                else:
                    nodeToDelete.parent.right = nodeToDelete.right

                nodeToDelete.right.parent = nodeToDelete.parent
            else:
                self.root = nodeToDelete.right
                self.root = None

        #two children
        else:
            #find most right child in left subtree
            maxChild = self.findMaximum(nodeToDelete.left)
            key = maxChild.key
            val = maxChild.val
            self.delete(maxChild)
            nodeToDelete.key = key
            nodeToDelete.val = val

    def findMaximum(self, root):
        if root.right is not None:
            return self.findMaximum(root.right)
        else:
            return root

    def __init__(self):
        self.root = None


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

    def __init__(self, x, y, ind=-1, created=True, prev=None, next=None):
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
                pts[pt] = Point(x = ptsCord[pt][0], y = ptsCord[pt][1], ind = pt, created = False, prev = None, next = None)

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


def signedArea(poly, firstDoubled= False):
    sumArea = 0
    if firstDoubled:
        indices = range(len(poly))
    else:
        indices = range(len(poly)+1)
        indices[len(poly)] = 0
    for i in range(len(indices)-1):
        p1 = poly[indices[i]]
        p2 = poly[indices[i+1]]
        sumArea += p1.x*p2.y - p2.x*p1.y
    return sumArea



















