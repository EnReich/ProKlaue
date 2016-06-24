"""
Calculates the volume of a polygonal object by using signed volumes.
All triangle faces of the mesh will be expanded to tetrahedra which are topped off at the origin (0,0,0). Sign of the volume is determined by the direction of the origin from the current face
(`source <http://stackoverflow.com/questions/1406029/how-to-calculate-the-volume-of-a-3d-mesh-object-the-surface-of-which-is-made-up>`_).
Command only accepts 'transform' nodes and will only be applied to the first object of the current selection.

There is a mel-command in Maya named `computePolysetVolume <http://help.autodesk.com/cloudhelp/2016/ENU/Maya-Tech-Docs/Commands/computePolysetVolume.html>`_ which calculates the volume of the selected object. The Maya command returns the exact same result (except numerical error :math:`< 10^{-5}`) as this implementation but runs roughly 100 times slower.

**Args:**
    :obj: string with object's name inside maya

:returns: float value with volume (based on maya's unit of length)

**Example:**
    .. code-block:: python

        cmds.polyTorus()
        # Result: [u'pTorus1', u'polyTorus1'] #
        cmds.getVolume()
        # Result: 4.774580351278257 #
        # MEL command
        computePolysetVolume();
        makeIdentity -apply true -t 1 -r 1 -s 1 -n 0 -pn 1;
        // pTorus2 faces = 400 //
        // TOTAL VOLUME = 4.774580265 //
        // Result: 4.77458 //
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.api.OpenMaya as om2     # needed for easier access to dag graph and vertices of object
import maya.OpenMaya as om
import maya.cmds as cmds
import misc
import numpy as np
import operator

dot = lambda x,y: sum(map(operator.mul, x, y))
cross = lambda x,y: map(operator.sub, [x[1]*y[2],x[2]*y[0],x[0]*y[1]], [x[2]*y[1],x[0]*y[2],x[1]*y[0]])

class getVolume(OpenMayaMPx.MPxCommand):
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

        # get mesh, triangles and points
        mesh = misc.getMesh(obj)
        simplices = misc.getTriangles(obj, mesh)
        points = [[p.x, p.y, p.z] for p in misc.getPoints(obj, mesh)]
        volume = 0.0
        # if object is a tetrahedron then use explicit volume formula instead of signed volume approach
        if (len(points) == 4):
            volume = abs(dot(map(operator.sub, points[0], points[3]), np.cross(map(operator.sub, points[1], points[3]), map(operator.sub, points[2], points[3])))) / 6.0
        # only proceed if there are any points in mesh object (less than 4 points cannot form a 3d simplex)
        elif (len(points) > 4):
            # get sub volume of each simplice
            values = [misc.signedVolumeOfTriangle(points[tri[0]], points[tri[1]], points[tri[2]]) for tri in simplices]
            # if there are any values in list, add them up to combined volume
            if (len(values)):
                volume = reduce(lambda x,y: x+y, values)
        self.setResult(volume)

# creator function
def getVolumeCreator():
    return OpenMayaMPx.asMPxPtr( getVolume() )

# syntax creator function
def getVolumeSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
