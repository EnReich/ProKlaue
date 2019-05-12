"""
Calculates the 3D delaunay triangulation (scipy library) on the vertices of a mesh object to have a real 3D solid representation of a given object model. Because the delaunay triangulation in scipy returns an unsorted list of vertices for each tetrahedron, the vertices are reordered to comply with the implicit normal definition used in Maya Autodesk. The result will be the triangulated convex hull of the given object mesh.

Command currently only returns a list of strings where each string encodes the vertices of one tetrahedron (as flattened 1D list) with its correct vertex order for an implicit normal definition (all normals are pointing away from center point of tetrahedron). That means, that the result needs to be converted to matrices (nested lists) again using 'numpy.fromstring(str, sep=",").reshape(4,3)' for each string 'str' in the returned list. The reason for this behavior is the fact, that one cannot return nested lists in Maya commands and the only workaround is to cast them to strings first.

Command only accepts 'transform' nodes and will only be applied to the first object of the current selection.

**command:** cmds.delaunay([obj])

**Args:**
    :obj: string with object's name inside maya

:returns: list of tetrahedrons as strings (each string represents 4 vertices with 3 points each separated by ','. Numbers [0:3] are first vertex, [3:6] second vertex, ...)

**Example:**
    .. code-block:: python

        cmds.polyCube()
        # Result: [u'pCube1', u'polyCube1'] #
        cmds.delaunay()
        # Result: [u'0.5, -0.5, 0.5, -0.5, 0.5, 0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5', u'-0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5', u'0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, -0.5, 0.5, -0.5, -0.5, 0.5', u'-0.5, 0.5, -0.5, -0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, -0.5, 0.5', u'0.5, 0.5, -0.5, 0.5, 0.5, 0.5, -0.5, 0.5, 0.5, 0.5, -0.5, 0.5', u'-0.5, -0.5, -0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, 0.5, -0.5, 0.5'] #

"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
import numpy as np
from scipy.spatial import Delaunay
import operator
from pk_src import misc

class delaunay(OpenMayaMPx.MPxCommand):
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def doIt(self, argList):
        # get only the first object from argument list
        try:
            obj = misc.getArgObj(self.syntax(), argList)[0]
        except:
            cmds.warning("No object selected!")
            return
        if (cmds.objectType(obj) != 'transform'):
            cmds.error("Object is not of type transform!")
            return

        # get and cast array of MPoint-Objects to list structure (needed for ConvexHull)
        pSet = list([p.x, p.y, p.z] for p in misc.getPoints(obj))
        # execute delaunay triangulation algorithm on point set of current object
        hull = Delaunay(pSet)

        # delaunay-triangulated tetrahedra are not ordered --> make sure implicit normal points outwards of each tetrahedra
        for tetra in hull.points[hull.simplices]:
            center = map(operator.add, map(operator.add, tetra[0], tetra[1]), map(operator.add, tetra[2], tetra[3]))
            center = map(operator.div, center, [4.0]*3)
            # check if tetrahedron is oriented in wrong direction (normals should point away from center)
            v1 = map(operator.sub, tetra[0], tetra[1])
            v2 = map(operator.sub, tetra[0], tetra[2])
            # check dot product: if smaller 0, swap vertices 0 and 2
            if (np.dot(np.cross(v1, v2), map(operator.sub, tetra[0], center) ) > 0):
                # swap 0 and 2
                tetra[[0,2]] = tetra[[2,0]]
            self.appendToResult(str(tetra.flatten().tolist())[1:-1])

# creator function
def delaunayCreator():
    return OpenMayaMPx.asMPxPtr( delaunay() )

# syntax creator function
def delaunaySyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
