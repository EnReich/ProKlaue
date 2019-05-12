"""
Calculates the eigenvectors of the current object, inserts one extra column and row and returns a 4x4 transformation matrix as 16 float-values. The eigenvectors are calculated in another function called :ref:`eigenvector`.
Command only accepts 'transform' nodes and will only be applied to the first object of the current selection.

**see also:** :ref:`eigenvector`, :ref:`normalize`

**command:** cmds.alignObj([obj], axisOrder = 'yzx', fast = False)

**Args:**
    :obj: string with object's name inside maya
    :axisOrder(ao): string to define axis order of eigenvectors (default 'yzx')
    :fast(f): boolean flag to indicate if calculation should use convex hull (faster but inaccurate)

**Example:**
    .. code-block:: python

        cmds.polyTorus()
        # Result: [u'pTorus1', u'polyTorus1'] #
        cmds.alignObj(ao = 'xyz')
        # Result: [0.609576559498125, 0.7927272028323672, -5.465342261024642e-10, 0.0, -1.3544498855821985e-09, 3.520841396209562e-10, -1.0, 0.0, -0.7927272028323671, 0.6095765594981248, 1.288331507658196e-09, 0.0, 0.0, 0.0, 0.0, 1.0] #
        cmds.xform(m = cmds.alignObj(ao = 'xyz'))
        # torus is positioned 'upwards' in x-y-plane
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om      # needed for syntax creator function
import maya.cmds as cmds
import numpy as np
from pk_src import misc

class alignObj(OpenMayaMPx.MPxCommand):
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
        # parse arguments and get flags
        argData = om.MArgParser (self.syntax(), argList)
        axisOrder = argData.flagArgumentString('axisOrder', 0) if (argData.isFlagSet('axisOrder')) else 'yzx'
        fast = argData.flagArgumentBool('fast', 0) if (argData.isFlagSet('fast')) else False
        # get eigenvectors as matrix (column-wise), reshape matrix and append extra row/column
        eig = np.matrix(cmds.eigenvector(obj, ao = axisOrder, f = fast))
        eig.shape = (3,3)
        eig = np.append(eig, [[0,0,0]], axis = 0)
        eig = np.append(eig.transpose(), [[0,0,0,1]], axis = 0).transpose()

        # return 4x4 matrix as 16 float values
        util = om.MScriptUtil()
        util.createFromList(eig.getA1().tolist(), eig.size)

        self.setResult(om.MDoubleArray(util.asDoublePtr(), eig.size))

# creator function
def alignObjCreator():
    return OpenMayaMPx.asMPxPtr( alignObj() )

# syntax creator function
def alignObjSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("ao", "axisOrder", om.MSyntax.kString)
    syntax.addFlag("f", "fast", om.MSyntax.kBoolean)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
