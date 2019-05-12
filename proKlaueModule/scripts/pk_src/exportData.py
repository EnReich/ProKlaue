"""
Exports the specified information for a list of objects for a whole animation. Information will be written to separate files. The file names will be the object names with a possible prefix. The files itself will contain a tab-separated table with one entry/row for each animation time step. Because the whole export runs in Maya's main thread, there is currently no possibility to cancel already started export commands and one needs to wait until the export is finished which, depending on the length of the animation and number of selected objects, can take a few minutes (to speed up the export with multiple models, one can move the camera away from the scene and all objects, so the rendering step runs much faster). The current progress of the export always shows the active time frame and for each selected object the whole animation will run through once.

All the information will be tracked via maya's space locators which sometimes may show unexpected behaviour. The space locators is not directly set inside the bone model but at the normalized position, where the bone model would be after the execution of the command :ref:`normalize`. Alternatively the flag *localCoordinateSystem* allows to define an arbitrary coordinate system which will be used instead of the automatically calculated one. This coordinate system needs to be grouped under the transform node of the current object. Additionally the normalized position of each bone model is taken in world coordinates from the first keyframe > 0, which may affect the overall orientation. Lastly, in combination with joint-hierarchies it does not matter, which object is selected (the transform bone model or the parent joint node), the script will automatically combine the keyframes of both transform node and joint node, find the transform node as child of the joint node (respectively the joint node as parent of the transform node) and apply all necessary transformations to the mesh object. Requirement for this behaviour to work is the explicit hierarchical order (joint nodes should have only ONE direct transform node as child and each transform node must have an unique parent joint node; to attach multiple transform nodes under a common ancestor, one needs to use intermediate joint nodes for each level of the hierarchy).

**see also:** :ref:`normalize`, :ref:`coordinateSystem`

**command:** cmds.exportData([obj], p = "", fp = "", cm = "centerPoint", lcs = False, jh = False, f = False, tm = True, wt = True, wa = True, wr = True, ao = 'yzx')

**Args:**
    :path(p): path to target directory
    :filePrefix(fp): prefix of all files
    :centerMethod(cm): string 'centerPoint' (default) to use average of all points as position, 'centroidPoint' to use weighted mean of triangle centroid and area as position or 'centerOBB' to use center of oriented bounding box as position (only when 'align' is True)
    :localCoordinateSystem(lcs): flag to indicate if objects have their own local coordinate system (True) or the normalized orientation (False) shall be used for export data (default False)
    :jointHierarchy(jh): flag to indicate if objects are organized in an hierarchy (True) or are completely independent of each other (default False)
    :fast(f): flag to indicate if covariance matrix should use the convex hull (True) or all points (False) (default FALSE)
    :writeTransformM(tm): flag to indicate if transform matrix shall be written to file
    :writeTranslation(wt): flag to indicate if translation shall be written to file
    :writeAngles(wa): flag to indicate if projected angles shall be written to file (deprecated)
    :writeRotations(wr): flag to indicate if rotation values shall be written to file (deprecated)
    :axisOrder(ao): string with axis ordering of eigenvectors (default 'yzx')
    :animationStart(as): first time step where information shall be exported (default *animationStartTime* of **playbackOptions**)
    :animationEnd(ae): last time step where information shall be exported (default *animationEndTime* of **playbackOptions**)
    :animationStep(by): time difference between two animation frames (default *by* of **playbackOptions**)
"""

import maya.OpenMaya as om
import maya.OpenMayaMPx as OpenMayaMPx
import maya.cmds as cmds
from functools import partial
import numpy as np
import operator
import math
import string
from pk_src import misc
import sys

class exportData(OpenMayaMPx.MPxCommand):
    windowID = 'wExportData'

    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def __cancelCallback(*pArgs):
        if cmds.window(exportData.windowID, exists = True):
            cmds.deleteUI(exportData.windowID)

    # wrapper to handle UI-interaction and execute command with specified flags
    def __applyCallback(self, path, prefix, FFstart, FFend, FFstep, RCcenterMethod, CHjointHierarchy, OMaxisOrder, CBfast, CBcenterPoint, CBwr, CBtm, CBwa, *pArgs):
        options = {'p':cmds.textField(path, q = 1, text = 1), 'fp':cmds.textField(prefix, q = 1, text = 1), 'as':cmds.floatField(FFstart, q = 1, value = 1), 'ae':cmds.floatField(FFend, q = 1, value = 1), 'by':cmds.floatField(FFstep, q = 1, value = 1), 'f':cmds.checkBox(CBfast, q = 1, v = 1), 'cm':cmds.radioCollection(RCcenterMethod, q = 1, sl = 1), 'wt':cmds.checkBox(CBcenterPoint, q = 1, v = 1), 'jh': cmds.checkBox(CHjointHierarchy, q = 1, v = 1), 'wr':cmds.checkBox(CBwr, q = 1, v = 1), 'tm':cmds.checkBox(CBtm, q = 1, v = 1), 'wa':cmds.checkBox(CBwa, q = 1, v = 1), 'ao':cmds.optionMenu(OMaxisOrder, q = 1, v = 1)}
        cmds.exportData(**options)

    # executed when state of lcs-checkBox is changed; will grey-out optionMenu 'axisOrder' and checkBox 'fast'
    def __changeLcsState(self, CBlcs, OMaxisOrder, CBfast, *pArgs):
        lcs = cmds.checkBox(CBlcs, q = 1, v = 1)
        if (lcs):
            cmds.optionMenu(OMaxisOrder, e = 1, enable = False)
            cmds.checkBox(CBfast, e = 1, enable = False)
        else:
            cmds.optionMenu(OMaxisOrder, e = 1, enable = True)
            cmds.checkBox(CBfast, e = 1, enable = True)

    # function to handle file dialogs and update corresponding text field
    def __fileDialog(self, field, *pArgs):
        selFile = cmds.fileDialog2(fm = 3, cap = 'specify export directory')
        if (selFile != None):
            cmds.textField(field, edit = True, text = selFile[0])

    def createUI (self, *pArgs):
        if cmds.window(self.windowID, exists = True):
            cmds.deleteUI(self.windowID)
        # initialize window properties
        cmds.window(self.windowID, title = 'exportDataUI', sizeable = True, resizeToFitChildren = True)
        cmds.columnLayout(columnAttach = ('both', 5), columnWidth = 265)

        cmds.rowLayout(numberOfColumns = 3, columnWidth3 = (100, 130, 20) )
        cmds.text(label = 'export directory:', width = 100, align = 'right')
        path = cmds.textField(visible = True, width = 130)
        cmds.button(label='...', command = partial(self.__fileDialog, path), width = 20, height = 20)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns = 3, columnWidth3 = (100, 130, 20) )
        cmds.text(label = 'file prefix:', width = 100, align = 'right')
        filePrefix = cmds.textField(visible = True, width = 130)
        cmds.separator(h = 10, style = 'none')
        cmds.setParent('..')

        cmds.text(label = '---------- Animation Time ----------')
        cmds.rowLayout(numberOfColumns = 6, columnWidth6 = (30, 50, 30, 65, 30, 40))
        cmds.text(label = 'start:', align = 'right', width = 30)
        start = cmds.floatField(visible = True, value = cmds.playbackOptions(q = 1, animationStartTime = 1), width = 50, pre = 1)
        cmds.text(label = 'end:', align = 'right', width = 30)
        end = cmds.floatField(visible = True, value = cmds.playbackOptions(q = 1, animationEndTime = 1), width = 50, pre = 1)
        cmds.text(label = 'step:', align = 'right', width = 30)
        step = cmds.floatField(visible = True, value = cmds.playbackOptions(q = 1, by = 1), width = 40, pre = 1)
        cmds.setParent('..')

        cmds.text(label = '---------- Position ----------')
        cmds.rowLayout(numberOfColumns = 3, columnWidth3 = (80, 80, 80))
        centerMethod = cmds.radioCollection()
        avPoint = cmds.radioButton("centerPoint", label = 'center', select = 1)
        centroid = cmds.radioButton("centroidPoint", label = 'centroid')
        centerOBB = cmds.radioButton("centerOBB", label = 'OBB')
        cmds.setParent('..')
        cmds.rowLayout(numberOfColumns = 1, columnWidth1 = 80)
        jointHierarchy = cmds.checkBox(label = 'Joint Hierarchy', value = False)
        cmds.setParent('..')

        cmds.text(label = '---------- Orientation ----------')
        # create checkbox to switch between local and automatic coordinate system
        cmds.rowLayout(numberOfColumns = 1, columnWidth1 = 80)
        # cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (80, 80))
        # cs = cmds.radioCollection()
        # automatic = cmds.radioButton("Automatic CS", label = 'automatic', select = 1)
        # manual = cmds.radioButton("Manual CS", label = 'manual')
        lcs = cmds.checkBox(label = 'Use Local Coordinate System', value = False, ann = 'If activated a local coordinate system must be grouped under current transform node')
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (130, 130))
        # create dropdown menu for different alignment options
        axisOrder = cmds.optionMenu( label='axis order:')
        cmds.menuItem( label='xyz' , p = axisOrder)
        cmds.menuItem( label='xzy' , p = axisOrder)
        cmds.menuItem( label='yxz' , p = axisOrder)
        cmds.menuItem( label='yzx' , p = axisOrder)
        cmds.menuItem( label='zxy' , p = axisOrder)
        cmds.menuItem( label='zyx' , p = axisOrder)
        cmds.optionMenu(axisOrder, e = 1, select = 4)
        fast = cmds.checkBox(label = 'fast calculation', value = False)
        cmds.setParent('..')

        # edit lcs checkbox and add changeCommand flag which is triggered when state is changed ('axisOrder' and 'fast' are disabled/enabled)
        # cmds.radioButton(automatic, e = 1, cc = partial(self.__changeLcsState, cs, axisOrder, fast))
        cmds.checkBox(lcs, e = 1, cc = partial(self.__changeLcsState, lcs, axisOrder, fast))

        cmds.text(label = '---------- Export ----------')
        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (130, 130))
        cbCenterPoint = cmds.checkBox(label = 'Position', value = True)
        cbWR = cmds.checkBox(label = 'Rotations', value = True)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (130, 130))
        cbTM = cmds.checkBox(label = 'Transformation matrix', value = True)
        cbWA = cmds.checkBox(label = 'Axis angles', value = True)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns = 2, columnWidth2 = (130, 130))
        cmds.button(label = 'apply', command = partial(self.__applyCallback, path, filePrefix, start, end, step, centerMethod, jointHierarchy, axisOrder, fast, cbCenterPoint, cbWR, cbTM, cbWA), width = 80 , align = 'center')
        cmds.button(label = 'cancel', command = self.__cancelCallback, width = 80, align = 'center')
        #cmds.window("wExportData", e = 1, wh = [270,125])
        cmds.showWindow()

    # Invoked when the command is run.
    def doIt(self,argList):
        # get objects from argument list
        try:
            selList = misc.getArgObj(self.syntax(), argList)
        except:
            cmds.warning("No objects selected!")
            return
        # first parse all the given arguments as flags
        argData = om.MArgParser (self.syntax(), argList)

        path = argData.flagArgumentString('path', 0) if (argData.isFlagSet('path')) else ""
        if (path == ""):
            cmds.warning("Please specify output path!")
            return
        # check if path directory ends with slash/backslash (depending on operating system)
        if (path[-1] != "/" and path[-1] != "\\"):
            path += "/" if (sys.platform == "linux" or sys.platform == "linux2") else "\\"
        # add prefix (if specified) after directory path
        prefix = argData.flagArgumentString('filePrefix', 0) if (argData.isFlagSet('filePrefix')) else ""
        path += prefix
        # get flags
        wt = argData.flagArgumentBool('writeTranslation', 0) if (argData.isFlagSet('writeTranslation')) else True
        tm = argData.flagArgumentBool('writeTransformM', 0) if (argData.isFlagSet('writeTransformM')) else True
        wa = argData.flagArgumentBool('writeAngles', 0) if (argData.isFlagSet('writeAngles')) else True
        axisOrder = argData.flagArgumentString('axisOrder', 0) if (argData.isFlagSet('axisOrder')) else "yzx"
        wr = argData.flagArgumentBool('writeRotations', 0) if (argData.isFlagSet('writeRotations')) else True
        centerMethod = argData.flagArgumentString('centerMethod', 0) if (argData.isFlagSet('centerMethod')) else "centerPoint"
        fast = argData.flagArgumentBool('fast', 0) if (argData.isFlagSet('fast')) else False
        lcs = argData.flagArgumentBool('localCoordinateSystem', 0) if (argData.isFlagSet('localCoordinateSystem')) else False
        jointHierarchy = argData.flagArgumentBool('jointHierarchy', 0) if (argData.isFlagSet('jointHierarchy')) else False

        # get playback options and set current time step to beginning of animation
        start = argData.flagArgumentDouble('animationStart', 0) if argData.isFlagSet('animationStart') else cmds.playbackOptions(q = 1, animationStartTime = 1)
        end = argData.flagArgumentDouble('animationEnd', 0) if argData.isFlagSet('animationEnd') else cmds.playbackOptions(q = 1, animationEndTime = 1)
        step = argData.flagArgumentDouble('by', 0) if argData.isFlagSet('by') else cmds.playbackOptions(q = 1, by = 1)

        for obj in selList:
            # user can call command with joint as argument iff actual object is direct child of joint
            if (cmds.objectType(obj) == "joint"):
                # find transform node(s) in joint
                obj = [x for x in cmds.listRelatives(obj) if cmds.objectType(x) == "transform"]
                if (len(obj) == 1):
                    obj = obj[0]
                else:
                    cmds.warning("There are multiple transform nodes grouped under joint! skip object ", obj)
                    continue
                # save object and its parent joint
                tmp = [obj] + cmds.listRelatives(obj, p = 1)
            # if object is NO joint (that means it should be an transform object), it still COULD HAVE a parent joint
            else:
                tmp = [obj]
                # check if object has some kind of parent
                parent = cmds.listRelatives(obj, p = 1)
                # if parent exists, add it to list (tmp)
                if (parent != None):
                    tmp += parent
            
            # set keyframe
            try:
                cmds.currentTime(min([key for key in cmds.keyframe(tmp, q = 1) if key > 0]))
            except:
                cmds.warning("No keyframes for object or its possible parent are found!", tmp)
                continue

            # if local coordinate system shall be used, check for transform node grouped under object
            if (lcs):
                cs = cmds.listRelatives(obj, type = 'transform')
                if (len(cs) == 1):
                    cs = cs[0]
                elif (len(cs) == 0):
                    cmds.warning("Could not find a local coordinate system grouped under object as transform node! skip object ", obj)
                    continue
                else:
                    cmds.warning("Multiple transform nodes are grouped under object and it is not clear which one is the local coordinate system! skip object ", obj)

                # get transformation matrix from local coordinate system
                rotM = cmds.xform(cs, q = 1, m = 1, ws = 1)
            # else use normalized position and get transformation matrix from alignment of object
            else:
                # get rotation matrix to align object
                rotM = np.matrix(cmds.alignObj(obj, ao = axisOrder, f = fast)).reshape(4,4)
                # transpose matrix and set center point for translation
                rotM = rotM.transpose().getA1().tolist()

                if (centerMethod == "centerPoint"):
                    rotM[12:15] = cmds.centerPoint(obj)
                elif (centerMethod == "centroidPoint"):
                    rotM[12:15] = cmds.centroidPoint(obj)
                else:
                    rangeX = cmds.rangeObj(obj, axis = rotM[0:3])
                    rangeY = cmds.rangeObj(obj, axis = rotM[4:7])
                    rangeZ = cmds.rangeObj(obj, axis = rotM[8:11])
                    rotM[12:15] = [0.5 * (rangeX[0] + rangeX[1]), 0.5 * (rangeY[0] + rangeY[1]), 0.5 * (rangeZ[0] + rangeZ[1])]
            
            # initialize locator and set its transformation matrix
            loc = cmds.spaceLocator()
            cmds.xform(loc, m = rotM)
            # set parent constraint with 'maintainOffset' flag (locator will follow objects transformations)
            # 'maintainOffset' is important to ensure correct position of locator
            cmds.parentConstraint(obj, loc, mo = not jointHierarchy)

            # open output stream to files and write header information
            if (wt):
                fileMean = open(path + string.replace(obj, ":", "") + ".mean", 'w')
                fileMean.write ("time\t\tTx\tTy\tTz\n")
            if (tm):
                fileMatrix = open(path + string.replace(obj, ":", "") + ".matrix", 'w')
                fileMatrix.write ("time\t\tM00\tM01\tM02\tM03\tM10\tM11\tM12\tM13\tM20\tM21\tM22\tM23\tM30\tM31\tM32\tM33\n")
            if (wa):
                fileAngles = open(path + string.replace(obj, ":", "") + "." + axisOrder, 'w')
                fileAngles.write("time\t\tAx\tAy\tAz\n")
            if (wr):
                fileRot = open(path + string.replace(obj, ":", "") + ".rot", 'w')
                fileRot.write ("time\t\tRx\tRy\tRz\n")

            # for each frame in animation
            for t in np.arange(start, end + step, step):
                options = {'x':0, 'y':1, 'z':2}
                cmds.currentTime(t)
                if (wt):
                    fileMean.write (str(t))
                    # parse python list of values to tab separated strings
                    fileMean.write ("\t" + "\t".join([ s.strip("[ ]") for s in str(cmds.xform(loc, q = 1, translation = 1)).split(",")]) + "\n")
                if (tm):
                    fileMatrix.write (str(t))
                    # parse python list of values to tab separated strings
                    fileMatrix.write("\t" + "\t".join([ s.strip("[ ]") for s in str(cmds.xform(loc, q = 1, m = 1)).split(",") ]) + "\n")
                if (wa):
                    fileAngles.write(str(t))
                    # write projected angles to file (acos of each value in main diagonal)
                    m = cmds.xform(loc, q = 1, m = 1)
                    # loop over all 3 eigenvectors
                    for e in range(3):
                        # slice over column
                        v = m[e:12:4]
                        # project onto planes (x vector to xy plane, y to yz plane, z to zx plane)
                        v[e-1] = 0
                        # normalize projected vector
                        v = map(operator.div, v, [np.sqrt(v[e] * v[e] + v[e-2] * v[e-2] )] * 3)
                        # angle between current local and world space axis
                        fileAngles.write("\t%d" % (math.acos(v[e]) * 180 / math.pi) )
                    fileAngles.write("\n")
                if (wr):
                    fileRot.write (str(t))
                    # parse python list of values to tab separated strings
                    fileRot.write ("\t" + "\t".join([ s.strip("[ ]") for s in str(cmds.xform(loc, q = 1, rotation = 1)).split(",")]) + "\n")
            if (wt):
                fileMean.close()
            if (tm):
                fileMatrix.close()
            if (wa):
                fileAngles.close()
            if (wr):
                fileRot.close()
            cmds.delete(loc)
        #self.__cancelCallback()

# creator function
def exportDataCreator():
    return OpenMayaMPx.asMPxPtr( exportData() )

# syntax creator function
def exportDataSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("p", "path", om.MSyntax.kString)
    syntax.addFlag("fp", "filePrefix", om.MSyntax.kString)
    syntax.addFlag("cm", "centerMethod", om.MSyntax.kString)
    syntax.addFlag("lcs", "localCoordinateSystem", om.MSyntax.kBoolean)
    syntax.addFlag("jh", "jointHierarchy", om.MSyntax.kBoolean)
    syntax.addFlag("f", "fast", om.MSyntax.kBoolean)
    syntax.addFlag("tm", "writeTransformM", om.MSyntax.kBoolean)
    syntax.addFlag("wt", "writeTranslation", om.MSyntax.kBoolean)
    syntax.addFlag("wa", "writeAngles", om.MSyntax.kBoolean)
    syntax.addFlag("ao", "axisOrder", om.MSyntax.kString)
    syntax.addFlag("wr", "writeRotations", om.MSyntax.kBoolean)
    syntax.addFlag("as", "animationStart", om.MSyntax.kDouble)
    syntax.addFlag("ae", "animationEnd", om.MSyntax.kDouble)
    syntax.addFlag("by", "animationStep", om.MSyntax.kDouble)
    return syntax

# create button for shelf
def addButton(parentShelf):
    cmds.shelfButton(parent = parentShelf, i = 'pythonFamily.png', c=exportData().createUI, imageOverlayLabel = 'exp', ann='export absolute coordinates of each selected object for each time frame')
