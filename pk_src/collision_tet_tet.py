"""
Module to test for collision of two tetrahedra using the axis separating theorem.
Based on the paper `"Fast tetrahedron-tetrahedron overlap algorithm" <http://vcg.isti.cnr.it/Publications/2003/GPR03/fast_tetrahedron_tetrahedron_overlap_algorithm.pdf>`_ by F. Ganovelli, F. Ponchio and C. Rocchini.

Uses two lists of vertices for each tetrahedra. The class function Collision_tet_tet.check() returns True iff the two given tetrahedra actually intersect (or at least have common vertices).
Before calling 'Collision_tet_tet.check()' the two tetrahedra need to be set with the methods **setV1**, **setV2** or **setV1V2**.
"""


import numpy as np
import operator

dot = lambda x,y: sum(map(operator.mul, x, y))
"""dot product as lambda function to speed up calculations"""
cross = lambda x,y: map(operator.sub, [x[1]*y[2],x[2]*y[0],x[0]*y[1]], [x[2]*y[1],x[0]*y[2],x[1]*y[0]])
"""cross product as lambda function to speed up calculations"""

class Collision_tet_tet:
    """
    Initialize tetrahedra definitions.

    :param tetra1: list of the 4 3D-vertices describing the first tetrahedron
    :param tetra2: list of the 4 3D-vertices describing the second tetrahedron

    """
    shifts = [1, 2, 4, 8]
    def __init__(self, tetra1 = None, tetra2 = None):
        # list of 3D-vertices of both tetrahedra (e.g. [[x,y,z], [x,y,z], [x,y,z], [x,y,z]])
        self.V1 = [[]]
        self.V2 = [[]]

        if (tetra1 is not None):
            self.V1 = tetra1
        if (tetra1 is not None):
            self.V2 = tetra2
        self.masks = np.zeros((4,1), dtype = int)
        self.coord_1 = np.zeros((4,4))

    def setV1(self, V1):
        """
        Set list of vertices for first tetrahedron.

        :param V1: list of the 4 3D-vertices describing the first tetrahedron
        """
        self.V1 = V1
    def setV2(self, V2):
        """
        Set list of vertices for second tetrahedron.

        :param V2: list of the 4 3D-vertices describing the second tetrahedron
        """
        self.V2 = V2
    def setV1V2(self, V1, V2):
        """
        Set list of vertices for first and second tetrahedron.

        :param V1: list of the 4 3D-vertices describing the first tetrahedron
        :param V2: list of the 4 3D-vertices describing the second tetrahedron
        """
        self.V1 = V1
        self.V2 = V2

    def separating_plane_faceA_1(self, pv1, n, coord, mask_edges):
        """
        Helper function for tetrahedron-tetrahedron collision test:
        checks if plane pv1 is a separating plane. Stores local coordinates and the mask bit mask_edges.

        :param pv1: vectors between vertices of second tetrahedron and vertex V[0] of first tetrahedron (list of lists [[...],[...],...])
        :param n: normal (list [x,y,z])
        :param coord: reference to list
        :param mask_edges: reference to list with single element [m]
        :returns: True if pv1 is a separating plane
        """
        mask_edges[0] = 0
        for i in range(4):
            coord[i] = dot(pv1[i], n)
            if (coord[i] > 0):
                mask_edges[0] |= Collision_tet_tet.shifts[i]

        return (mask_edges[0] == 15)

    def separating_plane_faceA_2(self, n, coord, mask_edges):
        """
        Helper function for tetrahedron-tetrahedron collision test:
        checks if plane v1,v2 is a separating plane. Stores local coordinates and the mask bit mask_edges.

        :param V1: list with vertices of tetrahedron 1
        :param V2: list with vertices of tetrahedron 2
        :param n: normal
        :param coord: reference to list
        :param mask_edges: reference to list with single element [m]
        """
        mask_edges[0] = 0
        for i in range(4):
            coord[i] = dot(map(operator.sub, self.V2[i], self.V1[1]), n)
            if (coord[i] > 0):
                mask_edges[0] |= Collision_tet_tet.shifts[i]

        return (mask_edges[0] == 15)

    def separating_plane_faceB_1(self, pv2, n):
        return ((dot(pv2[0], n) > 0) and (dot(pv2[1], n) > 0) and (dot(pv2[2], n) > 0) and (dot(pv2[3], n) > 0))

    def separating_plane_faceB_2(self, n):
        return ((dot(map(operator.sub, self.V1[0], self.V2[1]), n) > 0) and (dot(map(operator.sub, self.V1[1], self.V2[1]), n) > 0) and (dot(map(operator.sub, self.V1[2], self.V2[1]), n) > 0) and (dot(map(operator.sub, self.V1[3], self.V2[1]), n) > 0))

    def separating_plane_edge_A(self, f0, f1):
        """
        Helper function for tetrahedron-tetrahedron collision: checks if edge is in the plane separating faces f0 and f1.

        :param coord: 4x4 list of lists
        :param self.masks: 4x1 list of lists with single element
        :param f0: integer
        :param f1: integer
        """

        coord_f0 = self.coord_1[f0]
        coord_f1 = self.coord_1[f1]
        mask_f0 = self.masks[f0]
        mask_f1 = self.masks[f1]

        if ((mask_f0[0] | mask_f1[0] ) != 15):
            return False
        mask_f0[0] &= (mask_f0[0] ^ mask_f1[0])
        mask_f1[0] &= (mask_f0[0] ^ mask_f1[0])

        # edge 0: 0--1
        if ((mask_f0[0] & 1) and (mask_f1[0] & 2)):
            if ((coord_f0[1] * coord_f1[0] - coord_f0[0] * coord_f1[1]) > 0):
                # the edge of b (0,1) intersect (-,-) (see the paper)
                return False
        if ((mask_f0[0] & 2) and (mask_f1[0] & 1)):
            if ((coord_f0[1] * coord_f1[0] - coord_f0[0] * coord_f1[1]) < 0):
                return False
        # edge 1: 0--2
        if ((mask_f0[0] & 1) and (mask_f1[0] & 4)):
            if ((coord_f0[2] * coord_f1[0] - coord_f0[0] * coord_f1[2]) > 0):
                return False

        if ((mask_f0[0] & 4) and (mask_f1[0] & 1)):
            if ((coord_f0[2] * coord_f1[0] - coord_f0[0] * coord_f1[2]) < 0):
                return False

        # edge 2: 0--3
        if ((mask_f0[0] & 1) and (mask_f1[0] & 8)):
            if ((coord_f0[3] * coord_f1[0] - coord_f0[0] * coord_f1[3]) > 0):
                return False

        if ((mask_f0[0] & 8) and (mask_f1[0] & 1)):
            if ((coord_f0[3] * coord_f1[0] - coord_f0[0] * coord_f1[3]) < 0):
                return False

        # edge 3: 1--2
        if ((mask_f0[0] & 2) and (mask_f1[0] & 4)):
            if ((coord_f0[2] * coord_f1[1] - coord_f0[1] * coord_f1[2]) > 0):
                return False

        if ((mask_f0[0] & 4) and (mask_f1[0] & 2)):
            if ((coord_f0[2] * coord_f1[1] - coord_f0[1] * coord_f1[2]) < 0):
                return False

        # edge 4: 1--3
        if ((mask_f0[0] & 2) and (mask_f1[0] & 8)):
            if ((coord_f0[3] * coord_f1[1] - coord_f0[1] * coord_f1[3]) > 0):
                return False

        if ((mask_f0[0] & 8) and (mask_f1[0] & 2)):
            if ((coord_f0[3] * coord_f1[1] - coord_f0[1] * coord_f1[3]) < 0):
                return False

        # edge 5: 2--3
        if ((mask_f0[0] & 4) and (mask_f1[0] & 8)):
            if ((coord_f0[3] * coord_f1[2] - coord_f0[2] * coord_f1[3]) > 0):
                return False

        if ((mask_f0[0] & 8) and (mask_f1[0] & 4)):
            if ((coord_f0[3] * coord_f1[2] - coord_f0[2] * coord_f1[3]) < 0):
                return False

        # Now there exists a separating plane supported by the edge shared by f0 and f1.
        return True

    def check(self):
        """
        Function to check if current two tetrahedra can intersect; optimized for speed.

        :returns:  True, iff the two tetrahedra intersect (or have common vertices).
        """

        # vectors between V2 and V1[0]
        p_V1 = [map(operator.sub, p, self.V1[0]) for p in self.V2]
        # edges of V1 (at this point only edges from V1[0])
        e_V1 = [map(operator.sub, p, self.V1[0]) for p in self.V1[1:]]
        # face normal
        n = cross(e_V1[1], e_V1[0])

        # flip normal if not pointing outwards
        if (dot(n, e_V1[2]) > 0):
            n = map(operator.mul, n, [-1]*3)

        if (self.separating_plane_faceA_1(p_V1, n, self.coord_1[0], self.masks[0])):
            return False
        n = cross(e_V1[0], e_V1[2])

        if (dot(n, e_V1[1]) > 0):
            n = map(operator.mul, n, [-1]*3)
        if (self.separating_plane_faceA_1(p_V1, n, self.coord_1[1], self.masks[1])):
            return False
        if (self.separating_plane_edge_A(0, 1)):
            return False
        n = cross(e_V1[2], e_V1[1])

        if (dot(n, e_V1[0]) > 0):
            n = map(operator.mul, n, [-1]*3)
        if (self.separating_plane_faceA_1(p_V1, n, self.coord_1[2], self.masks[2])):
            return False
        if (self.separating_plane_edge_A(0, 2)):
            return False
        if (self.separating_plane_edge_A(1, 2)):
            return False
        e_V1.append (map(operator.sub, self.V1[2], self.V1[1]))
        e_V1.append (map(operator.sub, self.V1[3], self.V1[1]))
        n = cross(e_V1[3], e_V1[4])

        if (dot(n, e_V1[0]) < 0):
            n = map(operator.mul, n, [-1]*3)
        if (self.separating_plane_faceA_2(n, self.coord_1[3], self.masks[3])):
            return False
        if (self.separating_plane_edge_A(0, 3)):
            return False
        if (self.separating_plane_edge_A(1, 3)):
            return False
        if (self.separating_plane_edge_A(2, 3)):
            return False
        if ((self.masks[0][0] | self.masks[1][0] | self.masks[2][0] | self.masks[3][0] ) != 15):
            return True

        # From now on, if there is a separating plane, it is parallel to a face of b.
        p_V2 = [map(operator.sub, p, self.V2[0]) for p in self.V1]
        e_V2 = [map(operator.sub, p, self.V2[0]) for p in self.V2[1:]]
        n = cross(e_V2[1], e_V2[0])

        # Maybe flip normal
        if (dot(n, e_V2[2]) > 0):
            n = map(operator.mul, n, [-1]*3)
        if (self.separating_plane_faceB_1(p_V2, n)):
            return False
        n = cross(e_V2[0], e_V2[2])

        # Maybe flip normal
        if (dot(n, e_V2[1]) > 0):
            n = map(operator.mul, n, [-1]*3)
        if (self.separating_plane_faceB_1(p_V2, n)):
            return False
        n = cross(e_V2[2], e_V2[1])

        # Maybe flip normal
        if (dot(n, e_V2[0]) > 0):
            n = map(operator.mul, n, [-1]*3)
        if (self.separating_plane_faceB_1(p_V2, n)):
            return False

        e_V2.append (map(operator.sub, self.V2[2], self.V2[1]))
        e_V2.append (map(operator.sub, self.V2[3], self.V2[1]))
        n = cross(e_V2[3], e_V2[4])

        # Maybe flip normal. Note the < since e_v2[0] = V2[1] - V2[0].
        if (dot(n, e_V2[0]) < 0):
            n = map(operator.mul, n, [-1]*3)
        if (self.separating_plane_faceB_2(n)):
            return False

        return True
