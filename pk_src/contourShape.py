"""
Module to find and process the contour shape of an edge set on the plane.
Given a set of triangles on the plane, calculate the outline of the contour (meaning intersections of edges will create new points and the result will be a set of polygons)

Based on the paper `"Fast tetrahedron-tetrahedron overlap algorithm" <http://vcg.isti.cnr.it/Publications/2003/GPR03/fast_tetrahedron_tetrahedron_overlap_algorithm.pdf>`_ by F. Ganovelli, F. Ponchio and C. Rocchini.

Uses two lists of vertices for each tetrahedra. The class function Collision_tet_tet.check() returns True iff the two given tetrahedra actually intersect (or at least have common vertices).
Before calling 'Collision_tet_tet.check()' the two tetrahedra need to be set with the methods **setV1**, **setV2** or **setV1V2**.
"""


import numpy as np
import operator
import math
import heapq
import bisect

dot = lambda x,y: sum(map(operator.mul, x, y))
"""dot product as lambda function to speed up calculations"""
cross = lambda x,y: map(operator.sub, [x[1]*y[2], x[2]*y[0], x[0]*y[1]], [x[2]*y[1], x[0]*y[2], x[1]*y[0]])
"""cross product as lambda function to speed up calculations"""

EPSILON = 2e-15
GRAD_TO_RAD = math.pi/180

class SearchTreeNode:
    def __init__(self, key, val, parent, left, right):
        self.left = left #left Node
        self.right = right #right Node
        self.up = up #polygon is up from segment

        self.key = key
        self.val = val

        self.parent = parent
        self.left = left
        self.right = right

    def getKey(self, x):
        if self.right.x == self.left.x:
            return self.right.y
        else:
            return ((x-self.left.x)/(self.right.x-self.left.x))*(self.right.y-self.left.y)+self.left.y


class SearchTree:
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

    def findMinimum(self, root):
        if root.left is not None:
            return self.findMinimum(root.left)
        else:
            return root

    def findNext(self, root):
        if root.right is not None:
            return self.findMinimum(root.right)
        else:
            last = root
            actual = root.parent
            while actual is not None:
                if actual.left is last:
                    return actual
                else:
                    last = actual
                    actual = actual.parent
            return None


    def findPrev(self, root):
        if root.left is not None:
            return self.findMaximum(root.left)
        else:
            last = root
            actual = root.parent
            while actual is not None:
                if actual.right is last:
                    return actual
                else:
                    last = actual
                    actual = actual.parent
            return None

    def __init__(self):
        self.root = None


class Point:
    def __init__(self, x, y, ind=-1, created=True, prev=None, next=None, up=False):
        self.x = x                  #coordinates
        self.y = y
        self.ind = ind              #indice in the point field
        self.created = created      #was point created
        self.prev = prev
        self.next = next
        self.up=up                  #polygon lies upwards to point


class Segment:
    def __init__(self, ind, left, right, up):
        self.ind = ind
        self.left = left
        self.right = right
        self.turn = turn

class SweepEvent:
    def __init__(self, p, other, left, right, crossing, turn):
        self.p = p                  #point of the event
        self.other = other          #other point of the edge
        self.left = left            #is this the left endpoint
        self.right = right          #is this the right endpoint
        self.crossing = crossing    #is this a crossing point
        self.turn = turn            #the turn of the segment to this point (left -> (crossing) -> right)


def turn(p1x, p1y, p2x, p2y, p3x,p3y):
    A = (p3y-p1y)*(p2x-p1x)
    B = (p2y-p1y)*(p3x-p1x)
    return 1 if (A > B+EPSILON) else -1 if (A+EPSILON < B) else 0


def isIntersect(p1,p2,p3,p4):
    return (turn(p1, p3, p4) != turn(p2, p3, p4)) & (turn(p1, p2, p3) != turn(p1, p2, p4))


def getIntersectionPoint(l1p1,l1p2,l2p1,l2p2):
    u1 = ((float(l2p2[0])-l2p1[0])*(l1p1[1]-l2p1[1]) - (l2p2[1]-l2p1[1])*(l1p1[1]-l2p1[1])) / ((l2p2[1]-l2p1[1])*(l1p2[0]-l1p1[0]) - (l2p2[0]-l2p1[0])*(l1p2[1]-l1p1[1]))
    if u1<0 | u1>1:
        return None
    else:
        return l1p1+u1*(l1p2-l1p1)

def getOuterEdges(triEdgesRef):
    outerEdges = set([edg for edg, ref in triEdgesRef.items() if
                      len(ref) == 1])
    return outerEdges


def getPolygon(ptsCord, edges, triRefs, edgRefs):
    h =[]
    pts = {}
    for edge in edges:
        for pt in edge:
            if pt not in pts:
                npt = Point(x=ptsCord[pt][0, 0], y=ptsCord[pt][0, 1], ind=pt, created=False, prev=None, next=None)
                pts[pt] = npt

    for edge in edges:
        edgeList = list(edge)
        ptLeft = pts[edgeList[0]]
        ptRight = pts[edgeList[1]]
        if (ptLeft.x, ptLeft.y) > (ptRight.x, ptRight.y):
            ptLeft, ptRight = ptRight, ptLeft

        tri = triRefs[edge][0]
        for pt in tri:
            if pt not in edge:
                pt3 = pt

        turn = turn(ptLeft.x, ptLeft.y, ptRight.x, ptRight.y, ptsCord[pt3][0, 0], ptsCord[pt3][0, 1])

        evtLeft = SweepEvent(ptLeft, ptRight, True, False, False, turn)
        evtRight = SweepEvent(ptRight, ptLeft, False, False, False, -turn)

        heapq.heappush(h, ((evtLeft.p.x, evtLeft.p.y), evtLeft))
        heapq.heappush(h, ((evtRight.p.x, evtRight.p.y), evtRight))


    t = []
    ind = 0
    while h:
        event = heapq.heappop(h)
        if event.left:
            tri = triRefs[event.p.ind]
            seg = Segment(ind, event.p, event.other, event.turn)
            ind += 1





    pts = {}
    for edge in edges:
        for pt in edge:
            if pt not in pts:
                pts[pt] = Point(x=ptsCord[pt][0, 0], y=ptsCord[pt][0, 1], ind=pt, created=False, prev=None, next=None)

        edgeList = list(edge)

        pt0 = pts[edgeList[0]]
        pt1 = pts[edgeList[1]]

        if pt0.next is not None:
            pt1.next = pt0
            pt0.prev = pt1
        elif pt1.next is not None:
            pt0.next = pt1
            pt1.prev = pt0
        else:
            tri = triRefs[edge]
            for pt in tri:
                if (pt != edgeList[0]) & (pt != edgeList[1]):
                    pt3_ind = pt
            if turn(pt0.x, pt0.y, pt1.x, pt1.y, ptsCord[pt3_ind][0, 0], ptsCord[pt3_ind][0, 1]) > 0:
                pt0.next = pt0
                pt1.prev = pt1
            else:
                pt1.next = pt0
                pt0.prev = pt1




        #if (pt0.next is not None) & (pt1.next is None):
        #     pt0, pt1 = pt1, pt0
        #
        # if pt0.next is None:
        #     if pt1.prev is not None:
        #         #reverse the order of the list
        #         pt = pt1
        #         while pt is not None:
        #             pt.prev, pt.next = pt.next, pt.prev
        #             pt = pt.next
        #
        # else:
        #     # reverse the order of the list
        #     pt = pt0
        #     while pt is not None:
        #         pt.prev, pt.next = pt.next, pt.prev
        #         pt = pt.prev
        # pt1.prev = pt0
        # pt0.next = pt1

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
    return sumArea/2



def rotate(pts, rp, alpha, beta, gamma, rad = False):
    if not rad:
        alpha *= GRAD_TO_RAD
        beta *= GRAD_TO_RAD
        gamma *= GRAD_TO_RAD
    rot = np.array([
                [math.cos(beta)*math.cos(gamma), math.cos(beta)*math.sin(gamma), -math.sin(beta)],
                [-math.cos(alpha)*math.sin(gamma)+math.sin(alpha)*math.sin(beta)*math.cos(gamma), math.cos(alpha)*math.cos(gamma)+math.sin(alpha)*math.sin(beta)*math.sin(gamma), math.sin(alpha)*math.cos(beta)],
                [math.sin(alpha)*math.sin(gamma)+math.cos(alpha)*math.sin(beta)*math.cos(gamma), -math.sin(alpha)*math.cos(gamma)+math.cos(alpha)*math.sin(beta)*math.sin(gamma), math.cos(alpha)*math.cos(beta)]
            ]).transpose()
    rp = np.array(rp)
    npts = [np.mat(pt - rp)*rot + rp for pt in pts]
    return npts






















