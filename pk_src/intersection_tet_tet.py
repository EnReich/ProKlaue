"""
Module to to calculate 3D intersection of two tetrahedra.
The second tetrahedron will be clipped against the first tetrahedron with a simplified sutherland-hodgman approach.
"""

import numpy as np
import itertools
from scipy.spatial import ConvexHull
import math
import operator

# a few lambda function to speed up calculation (faster than numpy)
dot = lambda x,y: sum(map(operator.mul, x, y))
"""dot product as lambda function to speed up calculation"""
cross = lambda x,y: map(operator.sub, [x[1]*y[2],x[2]*y[0],x[0]*y[1]], [x[2]*y[1],x[0]*y[2],x[1]*y[0]])
"""cross product as lambda function to speed up calculation"""
normalize = lambda v: map(operator.div, v, [math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])]*3)
"""normalization of 3D-vector as lambda function to speed up calculation"""
centroid_tri = lambda v1, v2, v3: map(operator.div, map(operator.add, map(operator.add, v1, v2), v3), [3.0]*3)
"""centroid of tri (3 3D-points) as lambda function to speed up calculation"""

EPSILON = 10e-10

class intersection_tet_tet:
    """
    Class to initialize tetrahedra and set normals and centroids

    :param tetra1: list of the 4 3D-vertices describing the first tetrahedron
    :param tetra2: list of the 4 3D-vertices describing the second tetrahedron
    """
    def __init__(self, tetra1 = None, tetra2 = None):
        # tetrahedra point definition
        self.V1 = []
        self.V2 = []
        # normal vectors of faces belonging to first tetrahedron (there are always exactly 4 face normals)
        self.Normals = [None]*4
        # 3d points of centroid on each face (there are always exactly 4 face centroids)
        self.Centroids = [None]*4
        if (tetra1 is not None):
            self.setV1(tetra1)
        if (tetra2 is not None):
            self.V2 = tetra2
        self.masks = np.zeros((4,1), dtype = int)
        self.coord_1 = np.zeros((4,4))

    def setV1(self, V1):
        """
        Set list of vertices for first tetrahedron and set normals and centroids.

        :param V1: list of the 4 3D-vertices describing the first tetrahedron
        """
        self.V1 = V1
        # set normals and centroids for current tetrahedron
        self.setNormals()
        self.setCentroid()
        self.D = [-dot(self.Normals[i], self.Centroids[i]) for i in range(4)]
    def setV2(self, V2):
        """
        Set list of vertices for second tetrahedron.

        :param V2: list of the 4 3D-vertices describing the second tetrahedron
        """
        self.V2 = V2
    def setV1V2(self, V1, V2):
        """
        Set list of vertices for first and second tetrahedron and set normals and centroids.

        :param V1: list of the 4 3D-vertices describing the first tetrahedron
        :param V2: list of the 4 3D-vertices describing the second tetrahedron
        """
        self.setV1(V1)
        self.V2 = V2

    def setNormals(self):
        """
        Calculate normals of tetrahedron for each 4 facets.
        """
        self.Normals[0] = normalize(cross(map(operator.sub, self.V1[0], self.V1[2]), map(operator.sub, self.V1[0], self.V1[1])))
        self.Normals[1] = normalize(cross(map(operator.sub, self.V1[0], self.V1[3]), map(operator.sub, self.V1[0], self.V1[2])))
        self.Normals[2] = normalize(cross(map(operator.sub, self.V1[0], self.V1[1]), map(operator.sub, self.V1[0], self.V1[3])))
        self.Normals[3] = normalize(cross(map(operator.sub, self.V1[2], self.V1[3]), map(operator.sub, self.V1[2], self.V1[1])))
    def setCentroid(self):
        """
        Calculate centroids of tetrahedron for each 4 facets.
        """
        self.Centroids[0] = centroid_tri(self.V1[0], self.V1[1], self.V1[2])
        self.Centroids[1] = centroid_tri(self.V1[0], self.V1[2], self.V1[3])
        self.Centroids[2] = centroid_tri(self.V1[0], self.V1[1], self.V1[3])
        self.Centroids[3] = centroid_tri(self.V1[1], self.V1[2], self.V1[3])

    def intersect(self):
        """
        Intersection calculation according to simplified sutherland-hodgman approach. Clip the second tetrahedron against the first one. At the end the triangles of the convex hull of all vertices will be returned

        :returns: list of triangles with 3D-coordinates which form the 3D convex intersection volume
        """
        outputList = self.V2
        # check all 4 faces of first tetrahedron
        for i in range(4):
            inputList = outputList
            outputList = []
            # distance of each vertex to plane
            dist = []
            # remember for each vertex if it is inside or outside plane or part of plane
            vtxIn = []
            vtxOut = []
            vtxPlane = []
            # check if all points of second tetrahedron are 'inside'/'outside' of current face
            for j,vtx in enumerate(inputList):
                dist.append (dot(self.Normals[i], vtx) + self.D[i])
                if (dist[-1] <= -EPSILON):
                    vtxIn.append(j)
                elif (dist[-1] >= EPSILON):
                    vtxOut.append(j)
                else:
                    vtxPlane.append(j)
            # if all points are part of plane then there is no intersection volume
            if (len(vtxPlane) == len(inputList)):
                return []
            # if all vertices are inside plane, then there is nothing to clip --> continue with next plane
            if (len(vtxIn) + len(vtxPlane) == len(inputList)):
                outputList = inputList
                continue
            # if all vertices are outside plane, then intersection is empty
            if (len(vtxOut) + len(vtxPlane) == len(inputList)):
                return []

            # points which are part of plane will be added to outputList
            for vtx in vtxPlane:
                outputList.append(inputList[vtx])
            # points inside plane will be added to outputList (starting point of each edge)
            for vtx in vtxIn:
                outputList.append(inputList[vtx])
            # now 1--3 vertices are inside plane --> use edges from these points to calculate intersection with plane
            # create list with edges (indices to vertices) where first index is always vertex inside plane
            edges = itertools.product(vtxIn, vtxOut)
            # edge really intersects current plane --> add intersection vertex to new list
            for edge in edges:
                # intersection point is just linear interpolation : v1 + (v2 - v1)*d
                d = dist[edge[0]] / (dist[edge[0]] - dist[edge[1]])
                outputList.append(map(operator.add, inputList[edge[0]], map(operator.mul, map(operator.sub, inputList[edge[1]], inputList[edge[0]]) , [d]*3)))
        # convex hull of output points and return triangulated object
        try:
            hull = ConvexHull(outputList)
            # vertices are usually NOT ordered according to implicit normal definition which influences volume calculation
            out = []
            vertices = hull.points[hull.vertices]
            # there should be at least 4 vertices or it cannot be a 3D convex polygon
            if (len(vertices) < 4):
                return []
            # get center of convex polygon to test normal direction
            center = map(operator.div, reduce(lambda x, y: map(operator.add, x, y), vertices), [float(len(vertices))]*3)
            # check order of vertices for each triangle and insert them into output list
            for tri in hull.points[hull.simplices]:
                v1 = map(operator.sub, tri[0], tri[1])
                v2 = map(operator.sub, tri[0], tri[2])
                if (dot(cross(v1, v2), map(operator.sub, tri[0], center) ) > 0 ):
                    out.append([tri[0], tri[1], tri[2]])
                else:
                    out.append([tri[0], tri[2], tri[1]])
            return (out)
        except:
            return []
