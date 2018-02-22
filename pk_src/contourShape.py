"""
BUGGED MODULE, keeping it for possible copy paste.
Module to find and process the contour shape of segments on the plane.
Given a set of segments on the plane, calculate the outline of the contour (meaning intersections of edges will create new points and the result will be a set of segments)
"""



import numpy as np
import operator
import math
import heapq
import bisect
from pk_src import misc

dot = lambda x, y: sum(map(operator.mul, x, y))
"""dot product as lambda function to speed up calculations"""
cross = lambda x, y: map(operator.sub, [x[1]*y[2], x[2]*y[0], x[0]*y[1]], [x[2]*y[1], x[0]*y[2], x[1]*y[0]])
"""cross product as lambda function to speed up calculations"""

EPSILON = 1e-15
EPSILON_COMP = 1e-15
GRAD_TO_RAD = math.pi/180
LEFT_ENDPOINT = 0                           #code for type of a sweep event of a left endpoint
RIGHT_ENDPOINT = 1
CROSS_POINT = 2
REMOVED = 3
TEST_POINT = 4
PRIORITY_LEFT = 1                           #priority for a sweep event of a left endpoint
PRIORITY_RIGHT = 0
COUNTER_CLOCKWISE = 1
CLOCKWISE = -1


class Point:
    """Class to represent a point.
           :param x: x-coordinate
           :type x: float
           :param y: y-coordinate
           :type y: float
           :param z: z-coordinate, will be carried along for implications in 3d, though it wont be used in calculations for
           intersection, etc.
           :type z: float
           :param ind: index in the input array of point coordinates
           :type ind: int
           :param created: is this point created or from the input point array
           :type created: bool
    """

    def __init__(self, x, y, z, ind=-1, created=True):
        self.x = x                  #coordinates
        self.y = y
        self.z = z                  #coordinate irrelevant to position in the plane but for 3d data
        self.ind = ind              #indice in the point field
        self.created = created      #was point created

    def __eq__(self, other):
        return (other.x == self.x and other.y == self.y)

    def __ne__(self, other):
        return (other.x != self.x or other.y != self.y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __cmp__(self, other):
        if (self.x, self.y) < (other.x, other.y):
            return -1
        if (self.x, self.y) > (other.x, other.y):
            return 1
        return 0


class Segment:
    """Class to represent a line segment in the plane, z-coordinate of points will be carried along for implications in 3d.
           :param left: left point of the segment (left in the sense of lower x-coordinates)
           :type left: :ref:`contourShape.Point`
           :param right: right point of the segment
           :type right: :ref:`contourShape.Point`
           :param turn: turn of the segment, left -> right -> face
           :type turn: int
           :param upper: upper segment (w.r.t. y-coordinate) in the current segment list
           :type: :ref:`contourShape.Segment`
           :param lower: lower segment in the current segment list
           :type: :ref:`contourShape.Segment`
    """

    def __init__(self, left, right, turn, upper=None, lower=None):
        self.left = left                    #left endpoint of segment
        self.right = right                  #right endpoint of segment
        self.turn = turn                    #turn of the segment (left -> right -> face)
        self.upper = upper                  #upper segment
        self.lower = lower                  #lower segment
        self.evtLeft = None

    def getIntersection(self, other):

        #check bounding box (only y, x is checked by sweep line)
        if min(self.left.y, self.right.y) > max(other.left.y, other.right.y):
            return None
        if max(self.left.y, self.right.y) < min(other.left.y, other.right.y):
            return None

        #overlapping segments
        if self.getSlope() == other.getSlope() and other.getYForX(self.left.x) == self.left.y:
            if self.left.x < other.left.x:
                return other.left
            elif self.left.x == other.left.x:
                return min(self.right, other.right)
            else:
                return self.left

        #a bit more complicated intersection test
        if not isIntersect(self.left, self.right, other.left, other.right):
            return None

        #calculation
        return getIntersectionPoint(self.left, self.right, other.left, other.right)

    def getYForX(self, x):
        if self.right.x == self.left.x:
            return self.left.y
        else:
            return ((x - float(self.left.x)) / (self.right.x - self.left.x)) * (self.right.y - self.left.y) + self.left.y

    def getSlope(self):
        if self.right.x > self.left.x:
            return (float(self.right.y)-self.left.y) / (self.right.x - self.left.x)
        else:
            return float('inf')

    def getPartOf(self, l=None, r=None):
        leftPt = l if l is not None else self.left
        rightPt = r if r is not None else self.right
        return Segment(left= leftPt, right=rightPt, turn = self.turn, upper = None, lower = None)

    def __str__(self):
        return 'left: '+str((self.left.x, self.left.y))+ ' | right: '+ str((self.right.x, self.right.y))+ ' | turn: ' + str(self.turn)

    def __eq__(self, other):
        return other.right == self.right and other.left == self.left and self.turn == other.turn

    def __ne__(self, other):
        return other.left != self.left or other.right != self.right or self.turn != other.turn

    def __hash__(self):
        return hash((self.left, self.right, self.turn))


class SweepEvent:
    def __init__(self, segment, type, point, other):
        self.point = point          #point for the event (insertion)
        self.segment = segment      #segment for the event
        self.type = type            #type of the event: 0 - left endpoint, 1 - right endpoint, 2 - crossing point, 3 - removed
        self.other = other          #other sweep event

    def getPriority(self):
        p = self.segment.left if self.type == LEFT_ENDPOINT else self.segment.right
        pr = PRIORITY_LEFT if self.type == LEFT_ENDPOINT else PRIORITY_RIGHT
        sl = -self.segment.getSlope() if self.type == RIGHT_ENDPOINT else self.segment.getSlope()
        ps = -1 if self.segment.turn == CLOCKWISE else 1
        result = (p.x, p.y, pr, sl, ps)
        return result


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
            slope = segment.getSlope()

            while (actSeg is not None) and (y, slope, segment.turn == COUNTER_CLOCKWISE) > (actSeg.getYForX(x), actSeg.getSlope(), actSeg.turn == COUNTER_CLOCKWISE):
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
                lastSeg.upper = segment
                segment.lower = lastSeg
                segment.upper = actSeg
                actSeg.lower = segment

    def delete(self, segment):
        if self.first is segment:
            if segment.upper is not None:
                self.first = segment.upper
                self.first.lower = None
            else:
                self.first = None
        else:
            if segment.lower is not None:
                segment.lower.upper = segment.upper
            if segment.upper is not None:
                segment.upper.lower = segment.lower

        # segment.upper = None
        # segment.lower = None

    def isOuterSegment(self, seg):
        actSeg = self.first
        sumTurn = 0
        while(actSeg is not seg):
            # if actSeg is None:
            #     print '--------ERROR SEGMENT NOT IN LIST-----------'
            #     print seg           #that should not happen
            #     print seg.lower
            #     print seg.upper
            #     print seg.evtLeft.getPriority()
            #     print seg.evtLeft.other.getPriority()
            #     return False
            if actSeg.right.x-actSeg.left.x > 0:
                sumTurn += actSeg.turn
            actSeg = actSeg.upper
        if sumTurn == 0 or sumTurn + seg.turn == 0:
            return True
        else:
            return False


def listWalker(root):
    current = root
    while current.upper is not None:
        yield current.upper
        current = current.upper


def turn(p1x, p1y, p2x, p2y, p3x, p3y):
    """Calculates the turn from p1 -> p2 -> p3
        :param p1x: x-coordinate of p1
        :type p1x: float
        :param p1y: y-coordinate of p1
        :type p1y: float
        :param p2x: x-coordinate of p2
        :type p2x: float
        :param p2y: y-coordinate of p2
        :type p2y: float
        :param p3x: x-coordinate of p3
        :type p3x: float
        :param p3y: y-coordinate of p3
        :type p3y: float
        :return: :ref:`contourShape.COUNTER_CLOCKWISE` for counterclockwise turn, :ref:`contourShape.CLOCKWISE` for
        clockwise turn, else 0
        :rtype: int
    """
    A = (p3y-p1y)*(p2x-p1x)
    B = (p2y-p1y)*(p3x-p1x)
    return COUNTER_CLOCKWISE if (A > B+EPSILON) else CLOCKWISE if (A+EPSILON < B) else 0


def isIntersect(p1, p2, p3, p4):
    """Intersection test for two lines.

        :param p1: point1 of line1
        :type p1: :ref:`contourShape.Point`
        :param p2: point2 of line1
        :type p2: :ref:`contourShape.Point`
        :param p3: point1 of line2
        :type p3: :ref:`contourShape.Point`
        :param p4: point2 of line2
        :type p4: :ref:`contourShape.Point`
        :return: False, if the two line dont intersect, True, if there could be an intersection point
    """
    return (turn(p1.x, p1.y, p3.x, p3.y, p4.x, p4.y) != turn(p2.x, p2.y, p3.x, p3.y, p4.x, p4.y)) & (turn(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) != turn(p1.x, p1.y, p2.x, p2.y, p4.x, p4.y))


def getIntersectionPoint(l1p1, l1p2, l2p1, l2p2):
    """Calculates the intersection point of two lines.

    :param l1p1: point1 of line1
    :type l1p1: :ref:`contourShape.Point`
    :param l1p2: point2 of line1
    :type l1p2: :ref:`contourShape.Point`
    :param l2p1: point1 of line2
    :type l2p1: :ref:`contourShape.Point`
    :param l2p2: point2 of line2
    :type l2p2: :ref:`contourShape.Point`
    :return: the intersection point in the plane (with interpolated z-Coordinate), if there is one
    :rtype: :ref:`contourShape.Point` if there is an intersection point, else None
    """
    div = ((l2p2.y-l2p1.y)*(l1p2.x-l1p1.x) - (l2p2.x-l2p1.x)*(l1p2.y-l1p1.y))
    if div == 0:
        return ZeroDivisionError
    else:
        u1 = ((float(l2p2.x)-l2p1.x)*(l1p1.y-l2p1.y) - (l2p2.y-l2p1.y)*(l1p1.x-l2p1.x)) / div
        if (u1 < 0) | (u1 > 1):
            return None
        else:
            nx = l1p1.x+u1*(l1p2.x-l1p1.x)
            ny = l1p1.y+u1*(l1p2.y-l1p1.y)

            nz1 = u1*(l1p2.z-l1p1.z) + l1p1.z
            div2 = l2p2.x-l2p1.x
            if div2 == 0:
                u2 = (ny-l2p1.y)/(l2p2.y - l2p1.y)
            else:
                u2 = (nx-l2p1.x)/div2
            nz2 = u2 * (l2p2.z-l2p1.z) + l2p1.z
            nz = (nz1+nz2)/2

            ipt = Point(x=nx, y=ny, z=nz)
            return ipt


def getOuterEdges(triEdgesRef):
    """Determine the Edges in the outline

    :param triEdgesRef: the triangles where a given edge is part of
    :type triEdgesRef: dict of edge -> triangle references for the edge
    :return: the edges in the outline (the edges who are referenced by exactly one face)
    :rtype: array of edge input type
    """
    outerEdges = set([edg for edg, ref in triEdgesRef.items() if
                      len(ref) == 1])
    return outerEdges


def getSegments(ptsCord, edges, triRefs, tris):
    """Transform the input into an array of contourShape.Segment.

    :param ptsCord: coordinates of the points
    :type ptsCord: array of [x,y,z]
    :param edges: edges each defined by the 2 indices from the point array
    :type edges: array of [p1, p2]
    :param triRefs: the triangles where a given edge is part of
    :type triRefs: dict of frozenset([Index of p1, Index of p2]) -> container with triangle indices with p1 and p2
    :param tris: the triangles each defined by 3 indices from the point array
    :type tris: array of [Index of p1, Index of p2, Index of p3]
    :returns: array of contourShape.Segment, with a segment for each given edge
    """
    pts = {}
    segments = []

    for edge in edges:
        for pt in edge:
            if pt not in pts:
                npt = Point(x=ptsCord[pt][0], y=ptsCord[pt][2], z=ptsCord[pt][1], ind=pt, created=False)
                pts[pt] = npt

    for edge in edges:
        edgeList = list(edge)
        ptLeft = pts[edgeList[0]]
        ptRight = pts[edgeList[1]]
        if ptLeft>ptRight:
            ptLeft, ptRight = ptRight, ptLeft

        tri = tris[list(triRefs[edge])[0]]
        for pt in tri:
            if pt not in edge:
                pt3 = pt

        edgTurn = turn(ptLeft.x, ptLeft.y, ptRight.x, ptRight.y, ptsCord[pt3][0], ptsCord[pt3][2])

        segment = Segment(left=ptLeft, right=ptRight, turn=edgTurn)

        segments.append(segment)

    return segments


def splitSegment(h, seg, p):
    """Given a priority queue, a segment and an intersection point, this method splits the segment at the intersection
    point and inputs the new endpoints events (sweep line events) into the priority queue.

    :param h: priority queue
    :type h: heapq
    :param seg: segment to split
    :type seg: contourShape.Segment
    :param p: intersection point
    :type p: :ref:`contourShape.Point`
    :returns: the new right part of the segment
    :rtype: :ref:`contourShape.Segment`
    """

    if p == seg.right or p == seg.left:
        return None
    if p > seg.right or p<seg.left:
        # print '################# Error while splitting ###########'
        return None
    else:
        rightPart = seg.getPartOf(l=p, r = seg.right)
        seg.right = p
        evtRight = seg.evtLeft.other
        evtCrossLeft = SweepEvent(segment=rightPart, type= LEFT_ENDPOINT, point=p, other=evtRight)
        evtCrossRight = SweepEvent(segment=seg, type= RIGHT_ENDPOINT, point=p, other=seg.evtLeft)
        evtRight.other = evtCrossLeft
        seg.evtLeft.other = evtCrossRight
        evtRight.segment = rightPart
        rightPart.evtLeft = evtCrossLeft

        heapq.heappush(h, (evtCrossLeft.getPriority(), evtCrossLeft))
        heapq.heappush(h, (evtCrossRight.getPriority(), evtCrossRight))

        return rightPart


def getPolygon(segmentsInput):
    """Given segments of a number of polygons, this method calculates the boolean union of these polygons via
    a simpler (probably not optimal) sweep line approach.

    :param segmentsInput: segments of a number of polygons
    the whole polygon turns counterclockwise (or clockwise for the outline of a hole)
    :type segments: iterable container of contourShape.Segment
    :returns: the signed area of the polygon described by the segments
    :rtype: array of contourShape.Segment

    **Example:**
        .. code-block:: python

            from pk_src import contourShape
            s1 = contourShape.Segment(contourShape.Point(0,0), contourShape.Point(1,0), contourShape.COUNTER_CLOCKWISE)     #turns from left to right
            s2 = contourShape.Segment(contourShape.Point(0,0), contourShape.Point(0,1), contourShape.CLOCKWISE)             #turns from right to left
            s3 = contourShape.Segment(contourShape.Point(0,1), contourShape.Point(1,1), contourShape.CLOCKWISE)             #from right to left
            s4 = contourShape.Segment(contourShape.Point(1,0), contourShape.Point(1,1), contourShape.COUNTER_CLOCKWISE)     #from left to right
            contourShape.signedArea([s1,s2,s3,s4])
            # Result: 1.0 #
    """
    finalSegments = []
    h =[] #priority queue for sweeping events
    t = SegmentList()

    for segment in segmentsInput:
        evtLeft = SweepEvent(segment=segment, type=LEFT_ENDPOINT, point=segment.left, other=None)
        evtRight = SweepEvent(segment=segment, type=RIGHT_ENDPOINT, point=segment.right, other=evtLeft)
        evtLeft.other = evtRight
        segment.evtLeft = evtLeft

        heapq.heappush(h, (evtLeft.getPriority(), evtLeft))
        heapq.heappush(h, (evtRight.getPriority(), evtRight))


    last = None
    segmentsToBeDeleted=[]
    while h:
        event = heapq.heappop(h)[1]
        if last is not None:
            if last.point.x < event.point.x or (last.point.x == event.point.x and last.type == RIGHT_ENDPOINT and event.type == LEFT_ENDPOINT):
                for seg in segmentsToBeDeleted:
                    t.delete(seg)
                segmentsToBeDeleted = []

        last = event

        # print ' --------------------------------- '
        # print event.segment
        # print 'left' if event.type == LEFT_ENDPOINT else 'right'

        if event.type == LEFT_ENDPOINT: #left endpoint
            seg = event.segment

            #insert segment
            t.insert(seg)

            ipts = []

            for s in listWalker(t.first):
                if seg is not s:
                    ipt = seg.getIntersection(s)
                    if ipt is not None:
                        ipts.append((ipt, s))

            ipts.sort()
            # for ipt in ipts:
            #     print str(ipt[0].x) + ' | ' + str(ipt[0].y)
            #     print ipt[1]

            right_part = seg
            for pt in ipts:
                other_seg = pt[1]
                p = pt[0]
                splitSegment(h, other_seg, p)
                n_right_part = splitSegment(h, right_part, p)
                if n_right_part is not None:
                    right_part = n_right_part

            # iptLow = None
            # iptUp = None
            #
            # if seg.lower is not None:
            #     iptLow = seg.getIntersection(seg.lower)
            # if seg.upper is not None:
            #     iptUp = seg.getIntersection(seg.upper)
            #
            # # ipt = None
            # # contrarySeg = None
            # # if iptUp is not None and iptLow is not None and iptUp != iptLow:
            # #     print "------FEHLER IPT-----"
            #
            # segRight=None
            # if iptLow is not None:
            #     splitSegment(h, seg.lower, iptLow)
            #     segRight = splitSegment(h, seg, iptLow)
            # if iptUp is not None:
            #     splitSegment(h, seg.upper, iptUp)
            #     if segRight is not None and iptLow < iptUp:
            #         segRight = splitSegment(h, segRight, iptUp)
            #     else:
            #         splitSegment(h, seg, iptUp)

            # print 'ipt: ' + str(contrarySeg) if ipt is not None else 'no ipt'

            # if ipt is not None: # and ((seg.left.x < ipt.x - EPSILON_COMP and seg.right.x > ipt.x + EPSILON_COMP) or (contrarySeg.left.x < ipt.x-EPSILON_COMP and contrarySeg.right.x > ipt.x + EPSILON_COMP)):
            #     splitSegment(h, seg, ipt)
            #     splitSegment(h, contrarySeg, ipt)

        elif event.type == RIGHT_ENDPOINT: #right endpoint
            seg = event.segment
            # lower = seg.lower
            # upper = seg.upper
            #
            # if lower is not None and upper is not None:
            #     ipt = lower.getIntersection(upper)
            #     if ipt is not None:# and (lower.left.x < ipt.x - EPSILON_COMP or upper.left.x < ipt.x - EPSILON_COMP) and (lower.right.x > ipt.x + EPSILON_COMP or upper.right.x > ipt.x + EPSILON_COMP):
            #         splitSegment(h, lower, ipt)
            #         splitSegment(h, upper, ipt)

            if t.isOuterSegment(seg):
                finalSegments.append(seg)

            segmentsToBeDeleted.append(seg)

    return finalSegments


def signedArea(segments):
    """Calculates the signed area of the polygon specified by the given segments
     (area for polygons turning counterclockwise is positive, area for polygons turning clockwise is negative.)
     For polygons with holes, this method can be used by simply adding the segments of the hole with clockwise
     turn into the container.

    :param segments: segments of the polygon, turn for a single segment specified in such a way that
    the whole polygon turns counterclockwise (or clockwise for the outline of a hole)
    :type segments: iterable container of contourShape.Segment
    :returns: the signed area of the polygon described by the segments
    :rtype: float

    **Example:**
        .. code-block:: python

            from pk_src import contourShape
            s1 = contourShape.Segment(contourShape.Point(0,0), contourShape.Point(1,0), contourShape.COUNTER_CLOCKWISE)     #turns from left to right
            s2 = contourShape.Segment(contourShape.Point(0,0), contourShape.Point(0,1), contourShape.CLOCKWISE)             #turns from right to left
            s3 = contourShape.Segment(contourShape.Point(0,1), contourShape.Point(1,1), contourShape.CLOCKWISE)             #from right to left
            s4 = contourShape.Segment(contourShape.Point(1,0), contourShape.Point(1,1), contourShape.COUNTER_CLOCKWISE)     #from left to right
            contourShape.signedArea([s1,s2,s3,s4])
            # Result: 1.0 #
    """
    sumArea = 0
    for segment in segments:
        areaSeg = (segment.right.x-segment.left.x)*(segment.right.y+segment.left.y)/float(2)
        if segment.turn != COUNTER_CLOCKWISE:
            sumArea += areaSeg
        else:
            sumArea -= areaSeg

    return sumArea


    # for connected polygons
    # for segment in segments:
    #     if segment.turn == COUNTER_CLOCKWISE:
    #         p1 = segment.left
    #         p2 = segment.right
    #     else:
    #         p1 = segment.right
    #         p2 = segment.left
    #     sumArea += p1.x * p2.y - p2.x * p1.y
    #
    # return sumArea / float(2)


def rotate(pts, rp, alpha, beta, gamma, order="xyz", rad = False):
    """Rotates the given points around a given rotation pivot point for the given angles.
        :param pts: points to be rotated
        :type pts: Array of points in the form [x,y,z]
        :param rp: rotation pivot
        :type rp: [x,y,z]
        :param alpha: angle for rotation around the x axis
        :type alpha: float
        :param beta: angle for rotation around the y axis
        :type beta: float
        :param gamma: angle for rotation around the z axis
        :type gamma: float
        :param rad: are the angles given in radians (default is degrees)
        :type rad: bool
        :returns: the rotated points
        :rtype: Array of points in the form np.array([x,y,z])
    """
    rot = misc.getRotationMatrix(alpha, beta, gamma, order, rad)[:-1, :-1]
    rp = np.array(rp)
    npts = [((pt - rp)*rot.transpose() + rp).A1 for pt in pts]

    # rp = np.mat(rp).transpose()
    # npts = [rot*(np.mat(pt).transpose() - rp) + rp for pt in pts]

    return npts

# tests
#
# segs = []
#
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
# segs.append(Segment(left=Point(x=4,y =1.75), right= Point(x=5, y=1.75), turn = 1))
# segs.append(Segment(left=Point(x=4.9,y =3), right= Point(x=5, y=1.75), turn = -1))
# segs.append(Segment(left=Point(x=4,y =1.75), right= Point(x=4.9, y=3), turn = -1))
#
#
#
#
# segs.append(Segment(left=Point(x=0.5, y=2), right=Point(x=0.5,y =3) , turn = -1))
# segs.append(Segment(left=Point(x=0.5,y =3), right= Point(x=2.5, y=3), turn = -1))
# segs.append(Segment(left=Point(x=0.5,y =2), right= Point(x=2.5, y=2), turn = 1))
# segs.append(Segment(left=Point(x=2.5,y =2), right= Point(x=2.5, y=3), turn = 1))
#
# segs.append(Segment(left=Point(x=0, y=0), right=Point(x=0,y =1) , turn = -1))
# segs.append(Segment(left=Point(x=0,y =1), right= Point(x=3, y=1), turn = -1))
# segs.append(Segment(left=Point(x=0,y =0), right= Point(x=3, y=0), turn = 1))
# segs.append(Segment(left=Point(x=3,y =0), right= Point(x=3, y=1), turn = 1))
#
# segs.append(Segment(left=Point(x=0, y=0), right=Point(x=0,y =3) , turn = -1))
# segs.append(Segment(left=Point(x=0,y =3), right= Point(x=1, y=3), turn = -1))
# segs.append(Segment(left=Point(x=0,y =0), right= Point(x=1, y=0), turn = 1))
# segs.append(Segment(left=Point(x=1,y =0), right= Point(x=1, y=3), turn = 1))
#
# segs.append(Segment(left=Point(x=2, y=0), right=Point(x=2,y =3) , turn = -1))
# segs.append(Segment(left=Point(x=2,y =3), right= Point(x=3, y=3), turn = -1))
# segs.append(Segment(left=Point(x=2,y =0), right= Point(x=3, y=0), turn = 1))
# segs.append(Segment(left=Point(x=3,y =0), right= Point(x=3, y=3), turn = 1))
#
#
#
# segs.append(Segment(left=Point(x=0,y =2), right= Point(x=0, y=10), turn = -1))
# segs.append(Segment(left=Point(x=0,y =10), right= Point(x=8, y=10), turn = -1))
# segs.append(Segment(left=Point(x=8,y =2), right= Point(x=8, y=10), turn = 1))
# segs.append(Segment(left=Point(x=0,y =2), right= Point(x=8, y=2), turn = 1))
#
# print signedArea(segs)
#
# segs.append(Segment(left=Point(x=0-7,y =2-7), right= Point(x=0-7, y=10-7), turn = -1))
# segs.append(Segment(left=Point(x=0-7,y =10-7), right= Point(x=8-7, y=10-7), turn = -1))
# segs.append(Segment(left=Point(x=8-7,y =2-7), right= Point(x=8-7, y=10-7), turn = 1))
# segs.append(Segment(left=Point(x=0-7,y =2-7), right= Point(x=8-7, y=2-7), turn = 1))
#
# print signedArea(segs)
#
# segsN = getPolygon(segs)
# for i, seg in enumerate(segsN):
#     print '----------------------------------------------'
#     print i
#     print seg
# print '--------------------------------------------------------------------------'
# print signedArea(segsN)
