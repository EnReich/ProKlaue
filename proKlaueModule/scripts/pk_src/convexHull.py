"""
Calculates the convex hull using the scipy-library (based on *Qhull*) of the current object and creates a new
transform node. The name of the new transform node will be the object's name with the suffix '_ch'
Command only accepts 'transform' nodes and will only be applied to the first object of the current selection.

**command:** cmds.convexHull([obj])

**Args:**
    :obj: string with object's name inside maya

:returns: name of transform node with convex hull mesh

**Example:**
    .. code-block:: python

        cmds.polyTorus()
        # Result: [u'pTorus1', u'polyTorus1'] #
        cmds.convexHull()
        # Result: pTorus1_ch #
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.api.OpenMaya as om2
import maya.cmds as cmds
import numpy as np
from scipy.spatial import ConvexHull
import operator
from pk_src import misc

class convexHull(OpenMayaMPx.MPxCommand):
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
        # execute convex hull algorithm on point set of current object
        hull = ConvexHull(pSet)
        # get empty mesh object
        mesh = om2.MFnMesh()
        # add each polygon (triangle) in incremental process to mesh
        center = cmds.centerPoint(obj)
        for i,tri in enumerate(hull.points[hull.simplices]):
            # triangle vertices are not ordered --> make sure all normals point away from object's center
            v1 = map(operator.sub, tri[0], tri[1])
            v2 = map(operator.sub, tri[0], tri[2])
            # cross product of v1 and v2 is the current face normal; if dot product with vector from origin is > 0 use vertex order 0,1,2 and if dot product < 0 reverse order (all triangles are defined clockwise when looking in normal direction -- counter-clockwise when looking onto front of face)
            if (np.dot(np.cross(v1, v2), map(operator.sub, tri[0], center) ) > 0 ):
                mesh.addPolygon( ( om2.MPoint(tri[0]), om2.MPoint(tri[1]), om2.MPoint(tri[2]) ) )
            else:
                mesh.addPolygon( ( om2.MPoint(tri[0]), om2.MPoint(tri[2]), om2.MPoint(tri[1]) ) )
        # get transform node of shapeNode and rename it to match object's name
        transformNode = cmds.listRelatives(mesh.name(), p = 1)
        transformNode = cmds.rename(transformNode, obj + "_ch")
        self.setResult(transformNode)

# creator function
def convexHullCreator():
    return OpenMayaMPx.asMPxPtr( convexHull() )

# syntax creator function
def convexHullSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
