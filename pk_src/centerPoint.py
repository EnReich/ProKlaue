"""
Calculates the average of all points (vertices of mesh) of current selection. This command is useful to define the origin of an arbitrary object's local coordinate frame.
Command only accepts 'transform' nodes and will only be applied to the first object of the current selection.

**see also:** :ref:`centroidPoint`, :ref:`axisParallelPlane`, :ref:`normalize`, :ref:`exportData`

**command:** cmds.centerPoint([obj])

**Args:**
    :obj: string with object's name inside maya

**Example:**
    .. code-block:: python

        cmds.polyTorus()
        # Result: [u'pTorus1', u'polyTorus1'] #
        cmds.centerPoint()
        # Result: [-8.046627044677735e-09, -5.513429641723633e-08, -1.1246651411056519e-07] #
        cmds.polyPyramid()
        # Result: [u'pPyramid1', u'polyPyramid1'] #
        cmds.centerPoint()
        # Result: [1.2363445733853951e-08, -0.21213203072547912, -1.2363447865482158e-08] #
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.api.OpenMaya as om2     # needed for easier access to dag graph and vertices of object
import maya.OpenMaya as om
import maya.cmds as cmds
import operator
import misc

class centerPoint(OpenMayaMPx.MPxCommand):
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

        # calculate average of all points
        points = [[p.x, p.y, p.z] for p in misc.getPoints(obj)]
        mean = reduce(lambda x,y: map(operator.add, x, y), points)
        mean = map(operator.div, mean, [len(points)]*3)

        # set result of command
        self.appendToResult(mean[0])
        self.appendToResult(mean[1])
        self.appendToResult(mean[2])

# creator function
def centerPointCreator():
    return OpenMayaMPx.asMPxPtr( centerPoint() )

# syntax creator function
def centerPointSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
