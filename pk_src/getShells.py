"""
Calculates all shells of current selection and returns them as list of strings. One shell are all triangles which are connected, i.e. there exists a path from each triangle to all other triangles using common edges.
Command only accepts 'transform' nodes and will only be applied to the first object of the current selection.

**command**: cmds.getShells([obj])

**Args:**
    :obj: string with object's name inside maya

:returns: list of strings. Each string stands for one shell with a list of indices separated by comma and enclosed by '[...]', e.g. ['[0, 1, 2, 3, 4, ...]', '[...]', '[...]', ...]
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.api.OpenMaya as om2     # needed for easier access to dag graph and vertices of object
import maya.OpenMaya as om
import maya.cmds as cmds
import misc

class getShells(OpenMayaMPx.MPxCommand):
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
        # get parser object and use (optional) progress string as reference to UI-text to show current progress of calculation
        argData = om.MArgParser (self.syntax(), argList)
        progress = argData.flagArgumentString('progress', 0) if (argData.isFlagSet('progress')) else ""

        simplices = misc.getTriangles(obj)

        # save each shell in nested loop with indices to objects faces
        shells = []
        shell_indices = []
        # save a list of booleans to indicate if tri already belongs to a found shell
        triInShell = [False]*len(simplices)
        # remember number of triangles already in one shell (to show progress)
        count = 0
        # loop over shells until each facet is mapped to one shell
        while len(shell_indices) < len(simplices):
            try:
                # get indices of all faces connected to given face (smallest index not used yet)
                # method ".index" will raise an exception if element is not found --> all triangles are in one shell --> break
                shell_indices = cmds.polySelect(obj, ets = triInShell.index(False), noSelection = 1)
            except:
                break

            # append shell indices to nested list structure (to remember which indices belong to which shell)
            #shells.append(shell_indices)
            shells.append(cmds.polySelect(obj, ets = triInShell.index(False), noSelection = 1, ass = 1))
            # set indices in boolean list to True
            for s in shell_indices:
                triInShell[s] = True
            count += len(shell_indices)
            # update progress in text field (percentage of triangles already in found shells)
            try:
                cmds.text(progress, e = 1, label = 'Processing... (' + "{0:.1f}%)".format(float(count)/len(simplices)  * 100) ) + '%%)'
                cmds.refresh(f = 1)
            except:
                pass
        # set result (string representation of each shell's indices list)
        for s in shells:
            self.appendToResult(str(s))

# creator function
def getShellsCreator():
    return OpenMayaMPx.asMPxPtr( getShells() )

# syntax creator function
def getShellsSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("p", "progress", om.MSyntax.kString)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
