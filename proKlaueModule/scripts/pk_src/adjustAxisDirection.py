"""
Calculates the projection of all points in currently selected mesh onto a given axis. Returns the given axis or its inverse depending on the distribution of projected points: if more points lie 'above' the median (based on the range of all point) then axis itself will be returned, if more points lie 'below' median then inverse of axis is returned. Main purpose is to use this function on one of the main axes.
Command only accepts 'transform' nodes. Command will only use first object in current selection.

**command:** cmds.adjustAxisDirection([obj], axis)

**Args:**
    :obj: string with object's name inside maya
    :axis(a): vector as list of 3 values where object's points are projected to (e.g. [1,0,0])

**Example:**
    .. code-block:: python

        cmds.polyPyramid()
        # [u'pPyramid1', u'polyPyramid1'] #
        cmds.adjustAxisDirection(a = [0,1,0])
        # Result: [-0.0, -1.0, -0.0] #
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
import operator
from pk_src import misc


class adjustAxisDirection(OpenMayaMPx.MPxCommand):
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

        # get points of object (as list of lists) and project them on given axis
        points = [misc.project([p.x, p.y, p.z], axis) for p in misc.getPoints(obj)]
        # get sum of each point (just to get one single value for each projected point)
        # one could use the euclidean distance but this will lead to errors when some points are negative
        s = [sum(p) for p in points]
        # min and max
        minV = reduce(lambda x,y: min(x,y), s)
        maxV = reduce(lambda x,y: max(x,y), s)
        # get median of range
        median = (minV + maxV)/2.0
        # number of projected point above median
        above = len([v for v in s if (v > median)])
        below = len(s) - above

        # if more points are 'below' median --> use inverse axis
        if (above < below):
            axis = map(operator.mul, axis, [-1]*3)

        self.appendToResult(axis[0])
        self.appendToResult(axis[1])
        self.appendToResult(axis[2])

# creator function
def adjustAxisDirectionCreator():
    return OpenMayaMPx.asMPxPtr( adjustAxisDirection() )

# syntax creator function
def adjustAxisDirectionSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("a", "axis", om.MSyntax.kDouble, om.MSyntax.kDouble, om.MSyntax.kDouble)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
