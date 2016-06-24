"""
Calculates the weighted average of all triangle centroids of current selection. This command is useful to define the origin of an arbitrary object's local coordinate frame in case of irregular triangulation of the object's mesh.
Command only accepts 'transform' nodes and will only be applied to the first object of the current selection.

**see also:** :ref:`centerPoint`

**command:** cmds.centroidPoint([obj])

**Args:**
    :obj: string with object's name inside maya

**Example:**
    .. code-block:: python

        cmds.polyTorus()
        # Result: [u'pTorus1', u'polyTorus1'] #
        cmds.centroidPoint()
        # Result: [8.981527616005896e-10, -9.250593323493368e-08, -3.1820641749655423e-07] #
        cmds.polyPyramid()
        # Result: [u'pPyramid1', u'polyPyramid1'] #
        cmds.centroidPoint()
        # Result: [8.125617691761175e-09, -0.20412414173849558, -8.125620434757627e-09] #
        cmds.centerPoint()
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.api.OpenMaya as om2     # needed for easier access to dag graph and vertices of object
import maya.OpenMaya as om
import maya.cmds as cmds
import operator
import numpy as np
import misc

class centroidPoint(OpenMayaMPx.MPxCommand):
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

        # get points and triangles of current object
        pSet = np.squeeze(np.asarray([[p.x, p.y, p.z] for p in misc.getPoints(obj)]))
        simplices = misc.getTriangles(obj)

        # area of each triangle
        area_t = [0] * len(simplices)
        # centroid of each triangle
        centroid_t = [[0,0,0]] * len(simplices)
        # area of whole object
        area = 0.0
        # centroid of whole object
        centroid = [0,0,0]
        # calculate area and centroid for each triangle
        for i, tri in enumerate(pSet[simplices]):
            area_t[i] = misc.areaTriangle(tri)
            area += area_t[i]
            centroid_t[i] = misc.centroidTriangle(tri)
            # centroid of hull is weighted mean of triangle centroids
            t = map(operator.mul, centroid_t[i], [area_t[i]]*3)
            centroid = map(operator.add, centroid, t)

        centroid = map(operator.div, centroid, [area]*3)

        # set result of command
        self.appendToResult(centroid[0])
        self.appendToResult(centroid[1])
        self.appendToResult(centroid[2])

# creator function
def centroidPointCreator():
    return OpenMayaMPx.asMPxPtr( centroidPoint() )

# syntax creator function
def centroidPointSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
