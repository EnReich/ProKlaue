"""
Insert an axis parallel plane (app) with normal vector along x-,y- or z-axis to be tangential to the minimum/maximum vertex of a selected object (in the specified axis direction), i.e. the plane has exactly one contact point with the selected object in its minimum/maximum x, y or z direction. Additionally the position of the plane is set to be below/above the mesh object's center point.

**see also:** :ref:`centerPoint`

**command:** cmds.axisParallelPlane([obj], plane = 'yz', position = 'min')

**Args:**
    :plane(p): string ('xy', 'xz', 'yz') to specifiy plane direction (default 'xz')
    :position(pos): string ('min', 'max') to specify if plane should be positioned at the minimal/maximal vertex position concerning the plane orientation (default 'min')
    :animation(anim): boolean flag to indicate if plane should be calculated for whole animation (True) or only for the current time step (default False)

:returns: name of created polyPlane object (name will be object name plus '_app')

**Example:**
    .. code-block:: python

        cmds.polyTorus()
        # Result: [u'pTorus1', u'polyTorus1'] #
        cmds.axisParallelPlane(p = 'xz', pos = 'min')
        # Result: [u'pTorus1_app', u'polyPlane1'] #
        # A new plane 'pTorus1_app' is inserted which is
        # positioned under the torus (-0.5 units below world origin)
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
from functools import partial
import operator
import misc
import numpy as np
import itertools

dot = lambda x,y: sum(map(operator.mul, x, y))
"""dot product as lambda function to speed up calculation"""

class axisParallelPlane(OpenMayaMPx.MPxCommand):
    windowID = 'wAxisParallelPlane'
    tf_obj = ""
    job = -1

    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def __cancelCallback(*pArgs):
        if cmds.window(axisParallelPlane.windowID, exists = True):
            cmds.deleteUI(axisParallelPlane.windowID)

        if (cmds.scriptJob(ex = axisParallelPlane.job)):
            cmds.scriptJob(kill = axisParallelPlane.job, force = 1)
        axisParallelPlane.job = -1

    def __createCallback(self, OMplane, OMposition, CBanimation, *pArgs):
        options = {'plane':cmds.optionMenu(OMplane, q = 1, v = 1), 'position':cmds.optionMenu(OMposition, q = 1, v = 1), 'anim':cmds.checkBox(CBanimation, q = 1, v = 1)}
        cmds.axisParallelPlane(**options)

    def __onSelectionChange(*pArgs):
        selList = cmds.ls(orderedSelection = 1)

        if len(selList) > 0:
            cmds.textField(axisParallelPlane.tf_obj, e = 1, text = selList[0])
        else:
            cmds.textField(axisParallelPlane.tf_obj, e = 1, text = "no object selected!")

    def createUI (self, *pArgs):
        if cmds.window(self.windowID, exists = True):
            cmds.deleteUI(self.windowID)
        if cmds.scriptJob(ex = self.job):
            cmds.scriptJob(kill = self.job, force = 1)

        cmds.window(self.windowID, title = 'axisParallelPlaneUI', sizeable = True, resizeToFitChildren = True)
        cmds.columnLayout(columnAttach = ('both', 5), columnWidth = 260)

        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (90, 170), columnAlign2 = ("right", "center") )

        cmds.text(label = 'current object:', width = 80, align = 'right')
        axisParallelPlane.tf_obj = cmds.textField(text = 'no object selected!', ed = 0, width = 150)
        selList = cmds.ls(orderedSelection = 1)
        if (len (selList) > 0):
            cmds.textField(axisParallelPlane.tf_obj, e = 1, text = selList[0])

        if (axisParallelPlane.job == -1):
            axisParallelPlane.job = cmds.scriptJob(event = ['SelectionChanged', self.__onSelectionChange], protected = 1)

        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (125, 125))
        # create dropdown menu for different planes
        plane = cmds.optionMenu( label = 'plane:' )
        cmds.menuItem( label='xy' , p = plane)
        cmds.menuItem( label='xz' , p = plane)
        cmds.menuItem( label='yz' , p = plane)
        cmds.optionMenu(plane, e = 1, select = 2)

        position = cmds.optionMenu( label = 'position:')
        cmds.menuItem ( label = 'min', p = position)
        cmds.menuItem ( label = 'max', p = position)

        cmds.setParent('..')

        animation = cmds.checkBox(label = 'animation', width = 250)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (125, 125), columnAlign2 = ("center", "center"))
        cmds.button(label = 'create', command = partial(self.__createCallback, plane, position, animation) , width = 80)
        cmds.button(label = 'cancel', command = self.__cancelCallback, width = 80)
        #cmds.window("wNormalize", e = 1, wh = [250,120])
        cmds.showWindow()

    def doIt(self, argList):
        # get objects from argument list
        try:
            obj = misc.getArgObj(self.syntax(), argList)[0]
        except:
            cmds.warning("No object selected!")
            return

        argData = om.MArgParser (self.syntax(), argList)

        plane = argData.flagArgumentString('plane', 0) if (argData.isFlagSet('plane')) else 'xz'
        if (plane != 'xy' and plane != 'xz' and plane != 'yz'):
            cmds.error("Invalid parameter defintion of 'plane'! Expected 'xy', 'xz' or 'yz'.")
            return
        position = argData.flagArgumentString('position', 0) if (argData.isFlagSet('position')) else 'min'
        if (position != 'min' and position != 'max'):
            cmds.error("Invalid parameter definition of position! Expected 'min' or 'max'.")
            return
        # get animation flag
        animation = argData.flagArgumentBool('anim', 0) if (argData.isFlagSet('anim')) else False
        # if animation flag is set, get list of all keyframes (else set to current keyframe)
        keyframes = list(set(cmds.keyframe(q = 1))) if (animation) else [cmds.currentTime(q = 1)]
        # dictionary to link string of plane orientation to normal used for maya object definition
        pToN = {'xy' : [0,0,1], 'xz' : [0,1,0], 'yz' : [1,0,0]}
        # create plane
        polyPlane = cmds.polyPlane(ax = pToN[plane], w = 10, h = 10, sx = 1, sy = 1, n = obj + "_app")
        # initialize extrem value (for one dimension) with +-infinity
        extrema = float('inf') if (position == 'min') else -float('inf')
        # initialize center variable where plane will be move to
        center = []
        extremeKey = []
        # access correct column of points list via index (column 0 are x-values, 1 y-values, 2 z-values; by scalar multiplying the normal vector from dictionary pToN with [0,1,2] the correct index will be calculated)
        index = sum(map(operator.mul, pToN[plane], [0,1,2]))

        # get local vertex positions to calculate bounding box (AABB) for possibly faster calculations
        # TODO: get minimum bounding box instead (transformation matrix from cmds.alignObj(), multiply with local vertices list, get box minMax box vertices, transform back with inverse transformation matrix)
        points = [[p.x, p.y, p.z] for p in misc.getPoints(obj, worldSpace = 0)]
        minMax = []
        for i in range(3):
            tmp = [p[i] for p in points]
            minMax.append([min(tmp), max(tmp)])
        # get vertices of bounding box
        box = np.array(list(itertools.product(minMax[0], minMax[1], minMax[2], [1])))

        # loop over all keyframe to search minimal/maximal position in one dimension
        for key in keyframes:
            cmds.currentTime(key)

            # only use heuristic if center position is set
            if center != []:
                # get transform matrix of current position
                transform = np.array(cmds.xform(q = 1, m = 1)).reshape(4,4)
                # if all transformed box vertices are 'above'/'below' normal, there can be no new minimum/maximum
                # matrix multiplication of box vertices with transform matrix and check if dot product of vector from each vertex to center of plane and plane normal is less 0 --> box vertex is above normal
                aboveN = [p for p in box * transform if dot(center - np.array(p)[0][:-1], pToN[plane]) < 0.0]
                # if minimum is searched and ALL vertices are above normal, there can be no new minimum
                # if maximum is searched and NO vertices are above normal, there can be no new maximum
                if (position == 'min' and len(aboveN) == len(box)) or (position == 'max' and len(aboveN) == 0):
                    continue
            # get points of current object in current frame
            #points = np.array([[p.x, p.y, p.z] for p in misc.getPoints(obj)])
            #points = points[:,index]
            points = [p[index] for p in misc.getPoints(obj)]
            # temp value store to evaluate if extreme value was changed
            tmp = extrema
            # get minimum/maximum value over all values until current key frame
            extrema = min(extrema, min(points)) if (position == 'min') else max(extrema, max(points))

            # check if extreme value was changed and save current key frame
            if (tmp != extrema):
                extremeKey = key
        # calculate center point of object for keyframe with minimal/maximal vertex
        cmds.currentTime(extremeKey)
        center = cmds.centerPoint(obj)
        # adjust minimum value in appropriate column so that object is tangential to plane
        center[sum(map(operator.mul, pToN[plane], [0,1,2]))] = extrema

        cmds.xform(polyPlane, t = center)
        cmds.makeIdentity(polyPlane, a = 1, t = 1)

        self.setResult(polyPlane)

# creator function
def axisParallelPlaneCreator():
    return OpenMayaMPx.asMPxPtr( axisParallelPlane() )

# syntax creator function
def axisParallelPlaneSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("p", "plane", om.MSyntax.kString)
    syntax.addFlag("pos", "position", om.MSyntax.kString)
    syntax.addFlag("a", "anim", om.MSyntax.kBoolean)
    return syntax

# create button for shelf
def addButton(parentShelf):
    cmds.shelfButton(parent = parentShelf, i = 'pythonFamily.png', c= axisParallelPlane().createUI, imageOverlayLabel = 'APP', ann='Create axis parallel plane tangential to minimum/maximum x, y or z coordinate of all vertices in object')
