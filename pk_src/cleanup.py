"""
Main purpose of this command is the extraction of the outer shell of an arbitrary object. Natural bones may have
natural inclusions due to air bubbles, blood vessels or imaging errors. During animation these inclusions can obstruct a clear view when the bone model is set half-transparent and inflate computation time. To extract only the outer bone shell one needs to delete all separate inclusions and possibly cut through 'connecting tubes', i.e. small blood vessels connecting an outer shell with an inner shell. A threshold can be set to determine and delete all triangles of the bone model whose normal's distance (orthogonal distance from center point of triangle) to the next triangle is smaller than this threshold.

The command implements a button *clean* under the shelf tab 'ProKlaue', where one can find and list all shells of a currently selected transform node. The shells can then be selected to show which part of the bone model is part of this shell. A number behind each shell indicates the amount of triangles inside each set and finally one shell can be extracted, so that all other shells and their triangles will be deleted from the mesh.

The command primarily handles all the user interaction inside maya and processes output of commands :ref:`getShells` and :ref:`findTubeFaces`.

**Command should only be used over the button interface**

"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.api.OpenMaya as om2
import maya.cmds as cmds
from functools import partial
import numpy as np
import operator
import misc

class cleanup(OpenMayaMPx.MPxCommand):
    windowID = 'wCleanup'
    obj = ""
    shells = []
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def __cancelCallback(*pArgs):
        if cmds.window(cleanup.windowID, exists = True):
            cmds.deleteUI(cleanup.windowID)

    def __updateProgress(self, Tprogress, case):
        """Handle different states of progress text element of GUI to inform about current calculations and interaction possibilities.

        @param Tprogress: Maya cmds.text() object pointing to GUI-element
        @param case: integer value specifying the text to be printed
        """
        cmds.text(Tprogress, e = 1, label = '')
        if (case == 0):
            cmds.text(Tprogress, e = 1, label = 'Processing...')
        if (case == 1):
            cmds.text(Tprogress, e = 1, label = "Press 'Find' to calculate shells")
        if (case == 2):
            cmds.text(Tprogress, e = 1, label = 'Click on a shell to view selection')
        if (case == 3):
            cmds.text(Tprogress, e = 1, label = "Press 'Select' to remove all other shells")
        if (case == 4):
            cmds.text(Tprogress, e = 1, label = 'Finished!')
        if (case == 5):
            cmds.text(Tprogress, e = 1, label = 'No object selected!')
        if (case == 6):
            cmds.text(Tprogress, e = 1, label = 'Faces selected! Delete to remove.')
        if (case == 7):
            cmds.text(Tprogress, e = 1, label = 'Fill holes and triangulate...')
        cmds.refresh(f = 1)

    def __findFacesCallback(self, FFthreshold, Tprogress, *pArgs):
        """Callback function for GUI to find faces in tubes with current threshold parameter.

        @param FFthreshold: Maya cmds.floatField() object pointing to GUI-element with threshold value
        @param Tprogress: Maya cmds.text() object pointing to GUI-element
        """
        self.__updateProgress(Tprogress, 0)
        options = {'t':cmds.floatField(FFthreshold, q = 1, value = 1), 'p':Tprogress}
        cmds.findTubeFaces(**options)
        self.__updateProgress(Tprogress, 6)

    def __applySelectCallback(self, SLlist, Tprogress, *pArgs):
        """Callback function for select button in GUI to extract currently selected shell.

        @param SLlist: Maya cmds.textScrollsList() object pointing to GUI-element with listed shells
        @param Tprogress: Maya cmds.text() object pointing to GUI-element
        """
        self.__updateProgress(Tprogress, 0)

        try:
            item = str(cmds.textScrollList(SLlist, q = 1, selectItem = 1)[0])
            # split given string into shells index to reference shell nested list
            index = int(str.split(item)[0][5:])
        except:
            cmds.warning("No shell selected!")
            self.__updateProgress(Tprogress, 2)
            return
        shell_indices = []
        # save all indices of not selected shell in list
        for i,s in enumerate(cleanup.shells):
            if (i != index):
                shell_indices.extend(s)
        # delete all faces in list
        if (len(shell_indices)):
            cmds.delete([ cleanup.obj + ".f[" + str(i) + "]" for i in shell_indices])
        # clear selection and remove item from scrollList
        cmds.select(clear = True)
        cmds.textScrollList(SLlist, e = 1, removeAll = 1)

        self.__updateProgress(Tprogress, 7)
        # fill possible holes and retriangulate polygon
        #cmds.polyCloseBorder(cleanup.obj)
        #cmds.polyTriangulate(cleanup.obj)
        self.__updateProgress(Tprogress, 4)

    def __applyScrollCallback(self, SLlist, Tprogress, *pArgs):
        """Callback to update user selection from scrollList.
        """
        self.__updateProgress(Tprogress, 0)
        # get selected item in scrollList as string
        item = str(cmds.textScrollList(SLlist, q = 1, selectItem = 1)[0])
        # split string into shells index and extract substring to reference shell nested list by its index
        item = int(str.split(item)[0][5:])
        print("select Shell" + str(item))
        cmds.select([ cleanup.obj + ".f[" + str(i) + "]" for i in cleanup.shells[item]])

        self.__updateProgress(Tprogress, 3)

    def __applyCallback(self, FFthreshold, SLlist, Tprogress, *pArgs):
        cmds.textScrollList(SLlist, e = 1, removeAll = 1)
        self.__updateProgress(Tprogress, 0)

        options = {'t':cmds.floatField(FFthreshold, q = 1, value = 1), 'p':Tprogress}

        try:
            cmds.cleanup(**options)
        except:
            cmds.warning("No 'transform' object selected!")
            self.__updateProgress(Tprogress, 1)
            return

        if (len(cleanup.shells) == 0):
            self.__updateProgress(Tprogress, 5)
        else:
            for i,s in enumerate(cleanup.shells):
                cmds.textScrollList(SLlist, e = 1, append = ["Shell" + str(i) + " (" + str(len(s)) + ")"])
            cmds.textScrollList(SLlist, e = 1, sc = partial(self.__applyScrollCallback, SLlist, Tprogress))
            self.__updateProgress(Tprogress, 2)

    def createUI (self, *pArgs):
        if cmds.window(self.windowID, exists = True):
            cmds.deleteUI(self.windowID)
        # initialize window properties
        cmds.window(self.windowID, title = 'cleanupUI', sizeable = True, resizeToFitChildren = True)
        cmds.columnLayout(columnAttach = ('both', 5), columnWidth = 265)

        cmds.rowLayout(numberOfColumns = 3, columnWidth3 = (80, 100, 80))
        cmds.text(label = 'Threshold:', align = 'right', width = 80)
        threshold = cmds.floatField(visible = True, minValue = 0.001, value = 0.1, width = 50, pre = 3, step = 0.01)
        findFacesButton = cmds.button(label = 'findFaces', width = 70 , align = 'center')
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns = 1, columnWidth1 = (250))
        cmds.text(label = 'List of shells with number of triangles:', align = "center", width = 230)
        cmds.setParent('..')
        cmds.rowLayout(numberOfColumns = 1, columnWidth1 = (250))
        scrollList = cmds.textScrollList(width = 250)
        cmds.setParent('..')

        cmds.rowLayout(numberOfColumns = 1, columnWidth1 = (250))
        progress = cmds.text(label = "Press 'Find' to calculate shells", align = 'left', width = 200, height = 10)
        cmds.setParent('..')

        cmds.button(findFacesButton, e = 1, command = partial(self.__findFacesCallback, threshold, progress))

        cmds.rowLayout(numberOfColumns = 3, columnWidth3 = (80, 80, 80))
        cmds.button(label = 'Find', command = partial(self.__applyCallback, threshold, scrollList, progress), width = 70 , align = 'center')
        cmds.button(label = 'Select', command = partial(self.__applySelectCallback, scrollList, progress), width = 70, align = 'center')
        cmds.button(label = 'Cancel', command = self.__cancelCallback, width = 70, align = 'center')
        cmds.setParent('..')

        cmds.showWindow()

    def doIt(self, argList):
        # get only the first object from argument list
        try:
            cleanup.obj = misc.getArgObj(self.syntax(), argList)[0]
        except:
            cmds.warning("No object selected!")
            return
        if (cmds.objectType(cleanup.obj) != 'transform'):
            cmds.error("Object is not of type transform!")
            return

        argData = om.MArgParser (self.syntax(), argList)
        progress = argData.flagArgumentString('progress', 0) if (argData.isFlagSet('progress')) else ""

        cleanup.shells = []
        # get list of strings with shell indices and cast them to regular integer lists
        lShells = cmds.getShells(cleanup.obj, p = progress)
        for s in lShells:
            cleanup.shells.append(np.fromstring(s[1:len(s)-1], dtype = int, sep = ",").tolist())

# creator function
def cleanupCreator():
    return OpenMayaMPx.asMPxPtr( cleanup() )

# syntax creator function
def cleanupSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("t", "threshold", om.MSyntax.kDouble)
    syntax.addFlag("p", "progress", om.MSyntax.kString)
    return syntax

# create button for shelf
def addButton(parentShelf):
    cmds.shelfButton(parent = parentShelf, i = 'pythonFamily.png', c=cleanup().createUI, imageOverlayLabel = 'clean', ann='clean current selection by removing holes')
