"""
Module to find and process the contour shape of segments on the plane.
Given a set of segments on the plane, calculate the outline of the contour (meaning intersections of edges will create new points and the result will be a set of segments)
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
LEFT_ENDPOINT = 0
RIGHT_ENDPOINT = 1
CROSS_POINT = 2
REMOVED = 3

class SearchTreeNode:
    def __init__(self, key, val, parent, left, right):
        self.left = left #left Node
        self.right = right #right Node

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

    def __eq__(self, other):
        return (other.x == self.x and other.y == self.y)


class Segment:
    def __init__(self, left, right, turn, lastCrossing = None, upper=None, lower=None):
        self.left = left                    #left endpoint of segment
        self.right = right                  #right endpoint of segment
        self.turn = turn                    #turn of the segment (left -> right -> face)
        self.lastCrossing = lastCrossing if lastCrossing is not None else self.left    #lastCrossingPoint
        self.upper = upper                  #upper segment
        self.lower = lower                  #lower segment

    def getIntersection(self, other):
        #check bounding box (only y, x is checked by sweep line)
        if min(self.left.y, self.right.y) > max(other.left.y, other.right.y):
            return None
        if max(self.left.y, self.right.y) < min(other.left.y, other.right.y):
            return None
        #a bit more complicated intersection test
        if not isIntersect(self.left, self.right, other.left, other.right):
            return None
        #calculation
        return getIntersectionPoint(self.left, self.right, other.left, other.right)

    def getYForX(self, x):
        if self.right.x == self.left.x:
            return self.left.y
        else:
            return ((x - self.left.x) / (self.right.x - self.left.x)) * (self.right.y - self.left.y) + self.left.y

    def getPartOf(self, l=None, r=None):
        left = l if l is not None else self.lastCrossing
        right = r if r is not None else self.right
        return Segment(left= left,right=right,turn = self.turn, upper = self.upper, lower = self.lower)

    def __str__(self):
        return 'left: '+str((self.left.x, self.left.y))+ ' | right: '+ str((self.right.x, self.right.y))+ ' | turn: ' + str(self.turn)




class SweepEvent:
    def __init__(self, segment, type, point):
        self.point = point          #point for the event (insertion)
        self.segment = segment      #segment for the event
        self.type = type            #type of the event: 0 - left endpoint, 1 - right endpoint, 2 - crossing point, 3 - removed


class SegmentList:
    def __init__(self, firstSegment=None):
        self.first = firstSegment

    def insert(self, segment):
        segment.lower = None
        segment.upper = None
        if self.first is None:
            self.first = segment
        else:
            actSeg = self.first
            x = segment.left.x
            y = segment.left.y
            while (actSeg is not None) and (y>actSeg.getYForX(x)):
                lastSeg = actSeg
                actSeg = actSeg.upper


            if actSeg is self.first:
                self.first.lower = segment
                segment.upper = self.first
                segment.lower = None
                self.first = segment
            elif actSeg is None:
                segment.upper = None
                segment.lower = lastSeg
                lastSeg.upper = segment
            else:
                segment.upper = actSeg
                segment.lower = actSeg.lower
                actSeg.lower.upper = segment
                actSeg.lower = segment

    def delete(self, segment):
        if self.first is segment:
            self.first = segment.upper

        if segment.lower is not None:
            segment.lower.upper = segment.upper
        if segment.upper is not None:
            segment.upper.lower = segment.lower

    def switch(self, segment1, segment2):
        if segment2.lower is segment1:
            segment2.lower = segment1.lower
            if segment2.lower is not None:
                segment2.lower.upper = segment2

            segment1.upper = segment2.upper
            if segment1.upper is not None:
                segment1.upper.lower = segment1

            segment1.lower = segment2
            segment2.upper = segment1

        elif segment1.lower is segment2:
            segment1.lower = segment2.lower
            if segment1.lower is not None:
                segment1.lower.upper = segment1

            segment2.upper = segment1.upper
            if segment2.upper is not None:
                segment2.upper.lower = segment2

            segment2.lower = segment1
            segment1.upper = segment2

        else:
            segment1.upper, segment1.lower, segment2.upper, segment2.lower = segment2.upper, segment2.lower, segment1.upper, segment1.lower
            if segment1.upper is not None:
                segment1.upper.lower = segment1
            if segment1.lower is not None:
                segment1.lower.upper = segment1
            if segment2.upper is not None:
                segment2.upper.lower = segment2
            if segment2.lower is not None:
                segment2.lower.upper = segment2

        if segment1 is self.first:
            self.first = segment2
        elif segment2 is self.first:
            self.first = segment1

    def isOuterSegment(self, seg):
        actSeg = self.first
        sumTurn = 0
        while(actSeg is not seg):
            sumTurn += actSeg.turn
            actSeg = actSeg.upper
        if sumTurn == 0 or sumTurn + seg.turn == 0:
            return True
        else:
            return False



def turn(p1x, p1y, p2x, p2y, p3x,p3y):
    A = (p3y-p1y)*(p2x-p1x)
    B = (p2y-p1y)*(p3x-p1x)
    return 1 if (A > B+EPSILON) else -1 if (A+EPSILON < B) else 0


def isIntersect(p1,p2,p3,p4):
    return (turn(p1.x, p1.y, p3.x, p3.y, p4.x, p4.y) != turn(p2.x, p2.y, p3.x, p3.y, p4.x, p4.y)) & (turn(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) != turn(p1.x, p1.y, p2.x, p2.y, p4.x, p4.y))


def getIntersectionPoint(l1p1,l1p2,l2p1,l2p2):
    div = ((l2p2.y-l2p1.y)*(l1p2.x-l1p1.x) - (l2p2.x-l2p1.x)*(l1p2.y-l1p1.y))
    if div == 0:
        return -1
    else:
        u1 = ((float(l2p2.x)-l2p1.x)*(l1p1.y-l2p1.y) - (l2p2.y-l2p1.y)*(l1p1.x-l2p1.x)) / div
        if (u1 < 0) | (u1 > 1):
            return None
        else:
            ipt = Point(l1p1.x+u1*(l1p2.x-l1p1.x), l1p1.y+u1*(l1p2.y-l1p1.y))
            return ipt


def getOuterEdges(triEdgesRef):
    outerEdges = set([edg for edg, ref in triEdgesRef.items() if
                      len(ref) == 1])
    return outerEdges


def isConnected(edges):
    edgesList = list(edges)
    ptsParts = {}
    edgParts = range(len(edgesList))

    #for edg in edgesList:


def getSegments(ptsCord, edges, triRefs, tris):
    pts = {}
    segments = []

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

        tri = tris[list(triRefs[edge])[0]]
        for pt in tri:
            if pt not in edge:
                pt3 = pt

        edgTurn = turn(ptLeft.x, ptLeft.y, ptRight.x, ptRight.y, ptsCord[pt3][0, 0], ptsCord[pt3][0, 1])

        segment = Segment(left=ptLeft, right=ptRight, turn=edgTurn)

        segments.append(segment)

    return segments


def getPolygon(segmentsInput):
    finalSegments = []
    h =[] #priority queue for sweeping events
    t = SegmentList()
    crossFinder = {} #to look up entries in the queue for deletion

    for segment in segmentsInput:
        evtLeft = SweepEvent(segment, LEFT_ENDPOINT, segment.left)
        evtRight = SweepEvent(segment, RIGHT_ENDPOINT, segment.right)

        heapq.heappush(h, ((segment.left.x, segment.left.y, 1), evtLeft))
        heapq.heappush(h, ((segment.right.x, segment.right.y, 0), evtRight))

    while h:
        event = heapq.heappop(h)[1]
        if event.type == LEFT_ENDPOINT: #left endpoint
            seg = event.segment

            #insert segment
            t.insert(seg)

            #remove old intersection
            if (seg.lower is not None) and (seg.upper is not None):
                if frozenset([seg.upper, seg.lower]) in crossFinder:
                    crossFinder[frozenset([seg.upper, seg.lower])].status = REMOVED

            # check intersections
            if seg.lower is not None:
                if frozenset([seg, seg.lower]) in crossFinder:
                    crossFinder[frozenset([seg, seg.lower])].status = CROSS_POINT
                else:
                    iptLow = seg.getIntersection(seg.lower)
                    if iptLow is not None:
                        iptEvent = SweepEvent([seg.lower, seg], CROSS_POINT, iptLow)
                        crossFinder[frozenset([seg, seg.lower])] = iptEvent
                        heapq.heappush(h, ((iptLow.x, iptLow.y, -1), iptEvent))

            if seg.upper is not None:
                if frozenset([seg, seg.upper]) in crossFinder:
                    crossFinder[frozenset([seg, seg.upper])].status = CROSS_POINT
                else:
                    iptUp = seg.getIntersection(seg.upper)
                    if iptUp is not None:
                        iptEvent = SweepEvent([seg, seg.upper], CROSS_POINT, iptUp)
                        crossFinder[frozenset([seg, seg.upper])] = iptEvent
                        heapq.heappush(h, ((iptUp.x, iptUp.y, -1), iptEvent))

        elif event.type == RIGHT_ENDPOINT: #right endpoint
            seg = event.segment

            lower = seg.lower
            upper = seg.upper

            #check intersection
            if lower is not None and upper is not None:
                if frozenset([lower, upper]) in crossFinder:
                    crossFinder[frozenset([lower, upper])].status = CROSS_POINT
                else:
                    ipt = lower.getIntersection(upper)
                    if ipt is not None:
                        iptEvent = SweepEvent([lower, upper], CROSS_POINT, ipt)
                        crossFinder[frozenset([lower, upper])] = iptEvent
                        heapq.heappush(h, ((ipt.x, ipt.y, -1), iptEvent))

            if t.isOuterSegment(seg) and ((event.point.x != seg.lastCrossing.x) or (event.point.y != seg.lastCrossing.y)):
                finalSegments.append(seg.getPartOf())

            t.delete(seg)

        elif event.type == CROSS_POINT: #crossing point
            seg0 = event.segment[0]
            seg1 = event.segment[1]

            if t.isOuterSegment(seg0) and (event.point != seg0.lastCrossing):
                # print seg0.getPartOf(r=event.point)
                finalSegments.append(seg0.getPartOf(r=event.point))

            if t.isOuterSegment(seg1) and (event.point != seg1.lastCrossing):
                # print seg1.getPartOf(r=event.point)
                finalSegments.append(seg1.getPartOf(r=event.point))

            seg0.lastCrossing = event.point
            seg1.lastCrossing = event.point

            t.switch(seg0, seg1)

            # remove old intersection
            if seg1.lower is not None:
                if frozenset([seg0, seg1.lower]) in crossFinder:
                    crossFinder[frozenset([seg0, seg1.lower])].status = REMOVED

            if seg0.upper is not None:
                if frozenset([seg1, seg0.upper]) in crossFinder:
                    crossFinder[frozenset([seg1, seg0.upper])].status = REMOVED

            # check intersections
            if seg1.lower is not None:
                if frozenset([seg1, seg1.lower]) in crossFinder:
                    crossFinder[frozenset([seg1, seg1.lower])].status = CROSS_POINT
                else:
                    iptLow = seg1.getIntersection(seg1.lower)
                    if iptLow is not None:
                        iptEvent = SweepEvent([seg1, seg1.lower], CROSS_POINT, iptLow)
                        crossFinder[frozenset([seg1, seg1.lower])] = iptEvent
                        heapq.heappush(h, ((iptLow.x, iptLow.y, -1), iptEvent))

            if seg0.upper is not None:
                if frozenset([seg0, seg0.upper]) in crossFinder:
                    crossFinder[frozenset([seg0, seg0.upper])].status = CROSS_POINT
                else:
                    iptUp = seg0.getIntersection(seg0.upper)
                    if iptUp is not None:
                        iptEvent = SweepEvent([seg0.upper, seg0], CROSS_POINT, iptUp)
                        crossFinder[frozenset([seg0, seg0.upper])] = iptEvent
                        heapq.heappush(h, ((iptUp.x, iptUp.y, -1), iptEvent))

    return finalSegments

    #
    # pts = {}
    # for edge in edges:
    #     for pt in edge:
    #         if pt not in pts:
    #             pts[pt] = Point(x=ptsCord[pt][0, 0], y=ptsCord[pt][0, 1], ind=pt, created=False, prev=None, next=None)
    #
    #     edgeList = list(edge)
    #
    #     pt0 = pts[edgeList[0]]
    #     pt1 = pts[edgeList[1]]
    #
    #     if pt0.next is not None:
    #         pt1.next = pt0
    #         pt0.prev = pt1
    #     elif pt1.next is not None:
    #         pt0.next = pt1
    #         pt1.prev = pt0
    #     else:
    #         tri = triRefs[edge]
    #         for pt in tri:
    #             if (pt != edgeList[0]) & (pt != edgeList[1]):
    #                 pt3_ind = pt
    #         if turn(pt0.x, pt0.y, pt1.x, pt1.y, ptsCord[pt3_ind][0, 0], ptsCord[pt3_ind][0, 1]) > 0:
    #             pt0.next = pt0
    #             pt1.prev = pt1
    #         else:
    #             pt1.next = pt0
    #             pt0.prev = pt1
    #
    #
    #
    #
    #     #if (pt0.next is not None) & (pt1.next is None):
    #     #     pt0, pt1 = pt1, pt0
    #     #
    #     # if pt0.next is None:
    #     #     if pt1.prev is not None:
    #     #         #reverse the order of the list
    #     #         pt = pt1
    #     #         while pt is not None:
    #     #             pt.prev, pt.next = pt.next, pt.prev
    #     #             pt = pt.next
    #     #
    #     # else:
    #     #     # reverse the order of the list
    #     #     pt = pt0
    #     #     while pt is not None:
    #     #         pt.prev, pt.next = pt.next, pt.prev
    #     #         pt = pt.prev
    #     # pt1.prev = pt0
    #     # pt0.next = pt1
    #
    # poly = []
    #
    # pt =pt0
    #
    # while pt not in poly:
    #     poly.append(pt)
    #     pt = pt.next
    #
    # return poly


# def signedArea(poly, firstDoubled= False):
#     sumArea = 0
#     if firstDoubled:
#         indices = range(len(poly))
#     else:
#         indices = range(len(poly)+1)
#         indices[len(poly)] = 0
#     for i in range(len(indices)-1):
#         p1 = poly[indices[i]]
#         p2 = poly[indices[i+1]]
#         sumArea += p1.x*p2.y - p2.x*p1.y
#     return sumArea/2

def signedArea(segments):
    sumArea = 0

    for segment in segments:
        if segment.turn == 1:
            p1 = segment.left
            p2 = segment.right
        else:
            p1 = segment.right
            p2 = segment.left
        sumArea += p1.x * p2.y - p2.x * p1.y

    return sumArea / 2


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


segs = []
# segs.append(Segment(left=Point(x=0,y =2), right= Point(x=1, y=1), turn = 1))
# segs.append(Segment(left=Point(x=1,y =1), right= Point(x=5, y=1.5), turn = 1))
# segs.append(Segment(left=Point(x=3, y=3), right=Point(x=5,y =1.5) , turn = -1))
# segs.append(Segment(left=Point(x=1.5, y=3), right= Point(x=3,y =3), turn = -1))
# segs.append(Segment(left=Point(x=0, y=2), right=Point(x=1.5,y =3) , turn = -1))
# segs.append(Segment(left=Point(x=0,y =3.5), right= Point(x=1.5, y=1.5), turn = 1))
# segs.append(Segment(left=Point(x=1.5,y =1.5), right= Point(x=4.5, y=2.5), turn = 1))
# segs.append(Segment(left=Point(x=3.5, y=3.5), right=Point(x=4.5,y =2.5) , turn = -1))
# segs.append(Segment(left=Point(x=3.5,y =3.5), right= Point(x=5, y=4.5), turn = 1))
# segs.append(Segment(left=Point(x=3,y =6), right= Point(x=5, y=4.5), turn = -1))
# segs.append(Segment(left=Point(x=0,y =3.5), right= Point(x=3, y=6), turn = -1))

segs.append(Segment(left=Point(x=0,y =2), right= Point(x=0, y=10), turn = -1))
segs.append(Segment(left=Point(x=0,y =10), right= Point(x=8, y=10), turn = -1))
segs.append(Segment(left=Point(x=8,y =2), right= Point(x=8, y=10), turn = 1))
segs.append(Segment(left=Point(x=0,y =2), right= Point(x=8, y=2), turn = 1))

print signedArea(segs)


segs.append(Segment(left=Point(x=0-7,y =2-7), right= Point(x=0-7, y=10-7), turn = -1))
segs.append(Segment(left=Point(x=0-7,y =10-7), right= Point(x=8-7, y=10-7), turn = -1))
segs.append(Segment(left=Point(x=8-7,y =2-7), right= Point(x=8-7, y=10-7), turn = 1))
segs.append(Segment(left=Point(x=0-7,y =2-7), right= Point(x=8-7, y=2-7), turn = 1))

print signedArea(segs)


segsN = getPolygon(segs)
for i, seg in enumerate(segsN):
    print '----------------------------------------------'
    print i
    print seg
print '--------------------------------------------------------------------------'
print signedArea(segsN)

