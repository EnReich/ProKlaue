"""
Calculates the projection of all points in currently selected mesh onto a given axis. Returns a list of 3 values with min[p.x + p.y + p.z], max[p.x + p.y + p.z] and abs(max - min). Function's main purpose is to calculate the range of an object with respect to the x,y,z-axes.
Command only accepts 'transform' nodes.

**command:** cmds.rangeObj([obj], a)

**Args:**
    :obj: string with object's name inside maya
    :axis(a): vector as list of 3 values where object's point are projected to

**Example:**
    .. code-block:: python

        cmds.polyPyramid()
        # Result: [u'pPyramid1', u'polyPyramid1'] #
        cmds.rangeObj(a = [0,1,0])
        # Result: [-0.3535533845424652, 0.3535533845424652, 0.7071067690849304] #
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.api.OpenMaya as om2     # needed for easier access to dag graph and vertices of object
import maya.OpenMaya as om
import maya.cmds as cmds
import operator
import misc
import math
import numpy as np

class rangeObj(OpenMayaMPx.MPxCommand):
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

        argData = om.MArgParser (self.syntax(), argList)
        try:
            axis = [argData.flagArgumentDouble('axis', 0), argData.flagArgumentDouble('axis', 1), argData.flagArgumentDouble('axis', 2)]
        except:
            cmds.warning("Error in rangeObj(): no axis to project points onto specified!")
            return

        # get points of object
        points = misc.getPoints(obj)
        # projection of points
        t = [misc.project([p.x, p.y, p.z], axis) for p in points]
        # get sum of each point to have a single value for comparison
        s = [sum(p) for p in t]
        # min and max
        minV = reduce(lambda x,y: min(x,y), s)
        maxV = reduce(lambda x,y: max(x,y), s)

        self.appendToResult(minV)
        self.appendToResult(maxV)
        self.appendToResult(abs(maxV - minV))

# creator function
def rangeObjCreator():
    return OpenMayaMPx.asMPxPtr( rangeObj() )

# syntax creator function
def rangeObjSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("a", "axis", om.MSyntax.kDouble, om.MSyntax.kDouble, om.MSyntax.kDouble)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
