"""
Normalize selected objects by aligning the eigenvectors of the covariance matrix to the world cordinate system and using the center point of each object as the origin of its local coordinate system.

**see also:** :ref:`alignObj`, :ref:`eigenvector`, :ref:`centerPoint`, :ref:`centroidPoint`

**command:** cmds.normalize([obj], s = 1.0, tZ = True, pZ = True, a = True, ao = 'yzx', f = False, cm = 'centerPoint')

**Args:**
    :scale(s): float value to scale all objects and their distance to the origin
    :transToZero(tZ): flag to indicate if objects shall be translated to world space origin (0,0,0) or remain at their position (default True)
    :pivToZero(pZ): flag to indicate if objects pivots shall be translated to world space origin (0,0,0) or remain at their position (default True)
    :align(a): flag to indicate if object shall be aligned according to eigenvectors (default True)
    :axisOrder(ao): string of length 3 with axis order (default 'yzx'). Will only be considered if flag 'align' is True
    :fast(f): flag to indicate if covariance matrix shall be computing using all points (False) or only the points in the convex hull (True). Will only be considered if flag 'align' is True (default False)
    :centerMethod(cm): string 'centerPoint' (default) to use average of all points as position, 'centroidPoint' to use weighted mean of triangle centroid and area as position or 'centerOBB' to use center of oriented bounding box as position (only when 'align' is True)
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
from functools import partial
import operator
import misc
import numpy as np

class normalize(OpenMayaMPx.MPxCommand):
    windowID = 'wNormalize'
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def __cancelCallback(*pArgs):
        if cmds.window(normalize.windowID, exists = True):
            cmds.deleteUI(normalize.windowID)

    def __applyCallback(self, scaleField, CBtToZero, CBpToZero, CBalign, OMaxisOrder, CBfast, RCcenterMethod, *pArgs):
        options = {'s':cmds.floatField(scaleField, q = 1, v = 1), 'tZ':cmds.checkBox(CBtToZero, q = 1, v = 1), 'pZ':cmds.checkBox(CBpToZero, q = 1, v = 1), 'a':cmds.checkBox(CBalign, q = 1, v = 1), 'ao':cmds.optionMenu(OMaxisOrder, q = 1, v = 1), 'f':cmds.checkBox(CBfast, q = 1, v = 1), 'cm':cmds.radioCollection(RCcenterMethod, q = 1, sl = 1)}
        cmds.normalize(**options)

    def createUI (self, *pArgs):
        if cmds.window(self.windowID, exists = True):
            cmds.deleteUI(self.windowID)
        cmds.window(self.windowID, title = 'normalizeUI', sizeable = True, resizeToFitChildren = True)
        cmds.columnLayout(columnAttach = ('both', 5), columnWidth = 250)

        #cmds.rowColumnLayout(numberOfColumns = 2, columnWidth = [(1,120), (2,120)], columnOffset = [(1, 'left', 3), (2, 'left', 3)])
        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (125, 125), columnAlign2 = ("right", "center") )
        cmds.text(label = 'scaling factor:', width = 100, align = 'right')
        scalingFactorField = cmds.floatField(value = 1.0, minValue = 0.1, maxValue = 10, step = 0.1, precision = 2, width = 100)
        cmds.setParent('..')

        #cmds.rowLayout(numberOfColumns = 1, columnWidth1 = (250), columnAlign1 = "center")
        cmds.text(label = '---------- Position ----------')

        cmds.rowLayout(numberOfColumns = 3, columnWidth3 = (80, 80, 80))
        centerMethod = cmds.radioCollection()
        avPoint = cmds.radioButton("centerPoint", label = 'center', select = 1)
        centroid = cmds.radioButton("centroidPoint", label = 'centroid')
        centerOBB = cmds.radioButton("centerOBB", label = 'OBB')

        cmds.setParent('..')
        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (125, 125))
        transToZero = cmds.checkBox(label = 'set T = (0,0,0)', value = True)
        pivToZero = cmds.checkBox(label = 'set piv = (0,0,0)', value = True)
        cmds.setParent('..')


        cmds.text(label = '---------- Orientation ----------')

        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (125, 125))
        alignToAxis = cmds.checkBox(label = 'align obj', value = True)
        # create dropdown menu for different alignment obtions
        axisOrder = cmds.optionMenu( label='axis order:')
        cmds.menuItem( label='xyz' , p = axisOrder)
        cmds.menuItem( label='xzy' , p = axisOrder)
        cmds.menuItem( label='yxz' , p = axisOrder)
        cmds.menuItem( label='yzx' , p = axisOrder)
        cmds.menuItem( label='zxy' , p = axisOrder)
        cmds.menuItem( label='zyx' , p = axisOrder)
        cmds.optionMenu(axisOrder, e = 1, select = 4)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (125, 125))
        fast = cmds.checkBox(label = 'fast calculation', value = False)
        cmds.separator(h = 10, style = 'none')
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (125, 125), columnAlign2 = ("center", "center"))
        cmds.button(label = 'apply', command = partial(self.__applyCallback, scalingFactorField, transToZero, pivToZero, alignToAxis, axisOrder, fast, centerMethod) , width = 80)
        cmds.button(label = 'cancel', command = self.__cancelCallback, width = 80)
        #cmds.window("wNormalize", e = 1, wh = [250,120])
        cmds.showWindow()

    def doIt(self, argList):
        # get objects from argument list
        try:
            selList = misc.getArgObj(self.syntax(), argList)
        except:
            cmds.warning("No object selected!")
            return

        argData = om.MArgParser (self.syntax(), argList)

        tToZ = argData.flagArgumentBool('transToZero', 0) if (argData.isFlagSet('transToZero')) else True
        pToZ = argData.flagArgumentBool('pivToZero', 0) if (argData.isFlagSet('pivToZero')) else True
        align = argData.flagArgumentBool('align', 0) if (argData.isFlagSet('align')) else True
        axisOrder = argData.flagArgumentString('axisOrder', 0) if (argData.isFlagSet('axisOrder')) else 'yzx'
        centerMethod = argData.flagArgumentString('centerMethod', 0) if (argData.isFlagSet('centerMethod')) else "centerPoint"
        fast = argData.flagArgumentBool('fast', 0) if (argData.isFlagSet('fast')) else False

        scale = argData.flagArgumentDouble('scale', 0) if (argData.isFlagSet('scale')) else 1.0
        if (scale <= 0.0):
            cmds.warning("Scaling factor should not be zero or negativ!")
            scale = 1.0

        for obj in selList:
            # get current position and rotation
            pos = cmds.xform(obj, query = True, translation = True)
            rot = cmds.xform(obj, query = True, rotation = True)
            # set translation and rotation to zero
            cmds.xform(obj, translation = [0,0,0], rotation = [0,0,0])
            # calculate center point of objects vertices
            if (centerMethod == "centroidPoint"):
                cPos = cmds.centroidPoint(obj)
            else:
                cPos = cmds.centerPoint(obj)

            # scaling option (effects position of center point and actual object size)
            if (scale != 1.0):
                cPos = map(operator.mul, cPos, [scale]*3)
                cmds.scale(scale, scale, scale, obj)
            # move object by negative center point position
            cmds.move(-cPos[0], -cPos[1], -cPos[2], obj)
            # reset transformations and scale
            cmds.makeIdentity(obj, apply = True, t = 1, s = 1)
            # if object shall remain in [0,0,0] do nothing, else set new translation
            if (not tToZ):
                cmds.xform(obj, translation = map(operator.add, pos, cPos), ws = 1)
            # if pivot shall be in [0,0,0], set it
            if (pToZ):
                cmds.xform(obj, pivots = [0,0,0], ws = 1)
            # set new attribute to remember import position
            if (cmds.attributeQuery('importPosition', n = obj, exists=1) == False):
                cmds.addAttr(obj, longName = 'importPosition', dataType = 'float3')
            cmds.setAttr(obj + ".importPosition", cPos[0], cPos[1], cPos[2], type = 'float3')
            # set rotation of object back to its initial value
            cmds.xform(obj, rotation = rot)
            # finally align object according to its eigenvectors
            if (align):
                rot = cmds.alignObj(obj, ao = axisOrder, f = fast)
                # set new attribute to remember initial orientation
                if (cmds.attributeQuery('rotM', n = obj, exists = 1) == False):
                    cmds.addAttr(obj, longName = 'rotM', dataType = 'matrix')
                cmds.setAttr(obj + ".rotM", rot, type = 'matrix')
                # maintain current position of object
                rot[12:15] = cmds.xform(obj, q = 1, translation = 1)

                cmds.xform(obj, m = rot)
                cmds.makeIdentity(obj, apply = True, r = 1)

                if (centerMethod == "centerOBB"):
                    rangeX = cmds.rangeObj(obj, axis = [1,0,0])
                    rangeY = cmds.rangeObj(obj, axis = [0,1,0])
                    rangeZ = cmds.rangeObj(obj, axis = [0,0,1])
                    center = [0.5 * (rangeX[0]+rangeX[1]), 0.5 * (rangeY[0]+rangeY[1]), 0.5 * (rangeZ[0]+rangeZ[1])]
                    cmds.move(-center[0], -center[1], -center[2], obj)

                cmds.makeIdentity(obj, apply = True, t = 1)
            cmds.refresh(f = 1)

# creator function
def normalizeCreator():
    return OpenMayaMPx.asMPxPtr( normalize() )

# syntax creator function
def normalizeSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("s", "scale", om.MSyntax.kDouble)
    syntax.addFlag("tZ", "transToZero", om.MSyntax.kBoolean)
    syntax.addFlag("pZ", "pivToZero", om.MSyntax.kBoolean)
    syntax.addFlag("a", "align", om.MSyntax.kBoolean)
    syntax.addFlag("ao", "axisOrder", om.MSyntax.kString)
    syntax.addFlag("f", "fast", om.MSyntax.kBoolean)
    syntax.addFlag("cm", "centerMethod", om.MSyntax.kString)
    return syntax

# create button for shelf
def addButton(parentShelf):
    cmds.shelfButton(parent = parentShelf, i = 'pythonFamily.png', c= normalize().createUI, imageOverlayLabel = 'norm', ann='Normalize current selected object by changing its coordinates to world space coordinates and align object according to its eigenvectors')
