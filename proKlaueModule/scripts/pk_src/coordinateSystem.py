"""
Initializes a local coordinate system for the given object consisting of 3 colored axes. Will be grouped under the given transform object. Can be used in the export-command instead of the normalized position.
Command only accepts 'transform' nodes and will only be applied to the first object of the current selection.

**see also:** :ref:`normalize`, :ref:`exportData`

**command:** cmds.coordinateSystem([obj], ao = 'yzx', f = False)

**Args:**
    :obj: string with object's name inside maya
    :axisOrder(ao): string with axis ordering of eigenvectors (default 'yzx')
    :fast(f): flag to indicate if covariance matrix should use the convex hull (True) or all points (False) (default True)
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
import numpy as np
from pk_src import misc

class coordinateSystem(OpenMayaMPx.MPxCommand):
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def callback (self, *pArgs):
        cmds.coordinateSystem()

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

        # parameter list
        argData = om.MArgParser (self.syntax(), argList)
        axisOrder = argData.flagArgumentString('axisOrder', 0) if (argData.isFlagSet('axisOrder')) else "yzx"
        fast = argData.flagArgumentBool('fast', 0) if (argData.isFlagSet('fast')) else True

        # build x-axis from cylinder and cone
        xCyl = cmds.polyCylinder(n = "xCyl", ax = [1,0,0], r = 0.05, h = 10, sx = 10)
        xCone = cmds.polyCone(n = "xCone", ax = [1,0,0], r = 0.15, h = 0.4)
        cmds.xform(xCone, t = [5,0,0])
        # combine both object
        xAxis = cmds.polyUnite(xCyl, xCone, ch = 0, n = "x")
        # create y and z axes by duplicating x axis
        yAxis = cmds.duplicate(xAxis, n = "y")
        zAxis = cmds.duplicate(xAxis, n = "z")
        cmds.xform(yAxis, ro = [0,0,90])
        cmds.xform(zAxis, ro = [0,-90,0])
        # freeze rotation of y-axis and z-axis
        cmds.makeIdentity(yAxis, a = 1, r = 1)
        cmds.makeIdentity(zAxis, a = 1, r = 1)
        # create space locator at the origin point (NOT necessary any more because of grouping)
        #origin = cmds.spaceLocator(n = "origin")

        # create shading nodes for each axis (x = red, y = green, z = blue)
        if (len(cmds.ls('xAxis', type = 'lambert'))):
            lx = cmds.ls('xAxis', type = 'lambert')[0]
        else:
            lx = cmds.shadingNode('lambert', asShader = 1, n = 'xAxis')
            cmds.setAttr(lx + '.color', 1, 0, 0, type = 'double3')

        if (len(cmds.ls('yAxis', type = 'lambert'))):
            ly = cmds.ls('yAxis', type = 'lambert')[0]
        else:
            ly = cmds.shadingNode('lambert', asShader = 1, n = 'yAxis')
            cmds.setAttr(ly + '.color', 0, 1, 0, type = 'double3')

        if (len(cmds.ls('zAxis', type = 'lambert'))):
            lz = cmds.ls('zAxis', type = 'lambert')[0]
        else:
            lz = cmds.shadingNode('lambert', asShader = 1, n = 'zAxis')
            cmds.setAttr(lz + '.color', 0, 0, 1, type = 'double3')

        # select each axis and assign shader node
        cmds.select(xAxis)
        cmds.hyperShade(assign = lx)
        cmds.select(yAxis)
        cmds.hyperShade(assign = ly)
        cmds.select(zAxis)
        cmds.hyperShade(assign = lz)

        # group all object
        #cs = cmds.group(xAxis, yAxis, zAxis, n = "CS")
        cs = cmds.polyUnite(xAxis, yAxis, zAxis, ch = 0, n = "CS")
        # make sure pivots are in origin
        cmds.xform(cs, piv = [0,0,0])

        # calculate alignment of given object to set coordinate system inside transform node
        rotM = np.matrix(cmds.alignObj(obj, ao = axisOrder, f = fast)).reshape(4,4)
        # transpose matrix and set center point for translation
        rotM = rotM.transpose().getA1().tolist()
        rotM[12:15] = cmds.centerPoint(obj)
        cmds.xform(cs, m = rotM)

        # set transform node as parent of the new coordinate system
        cmds.parent(cs, obj)
        # activate rotate tool
        cmds.RotateTool()
        self.setResult(cs)

# creator function
def coordinateSystemCreator():
    return OpenMayaMPx.asMPxPtr( coordinateSystem() )

# syntax creator function
def coordinateSystemSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("ao", "axisOrder", om.MSyntax.kString)
    syntax.addFlag("f", "fast", om.MSyntax.kBoolean)
    return syntax

# create button for shelf
def addButton(parentShelf):
    cmds.shelfButton(parent = parentShelf, i = 'pythonFamily.png', c=coordinateSystem().callback, imageOverlayLabel = 'lcs', ann='creates a local coordinate system inside the currently selected object')
