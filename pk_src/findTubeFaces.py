"""
Calculates the triangles of current object with an orthogonal distance to the next surface below a given threshold, which indicates a tube/tunnel to connect two shells of the object with each other. All those triangles will be selected and can be deleted by the standard user action.
Command only accepts 'transform' nodes and will only be applied to the first object of the current selection.

**see also:** :ref:`cleanup`, :ref:`getShells`

**command:** cmds.findTubeFaces([obj], t = 0.1)

**Args:**
    :obj: string with object's name inside maya
    :threshold(t): threshold of orthogonal distance between triangles. All triangles closer to each other than this threshold will be selected (default 0.1)
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.api.OpenMaya as om2     # needed for easier access to dag graph and vertices of object
import maya.OpenMaya as om
import maya.cmds as cmds
import operator
import numpy as np
import misc
import time

class findTubeFaces(OpenMayaMPx.MPxCommand):
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
        progress = argData.flagArgumentString('progress', 0) if (argData.isFlagSet('progress')) else ""
        threshold = argData.flagArgumentDouble('threshold', 0) if (argData.isFlagSet('threshold')) else 1e-1

        # get mesh object, its points, simplices and normals
        mfnObject = misc.getMesh(obj)
        # use numpy array instead of normal array to use index lists --> points[[1,2,3]] can be used
        points = np.asarray([[p.x, p.y, p.z] for p in misc.getPoints(obj)])
        simplices = misc.getTriangles(obj)
        normals = misc.getFaceNormals(obj)
        # build acceleration structure
        accel = mfnObject.autoUniformGridParams()
        tubeCandidates = []

        # normal and origin are needed for intersection test
        normal = om2.MFloatVector()
        origin = om2.MFloatPoint()
        tempProgress = len(simplices)/1000

        # for each triangle check if nearest intersection distance is less than given threshold
        for i in range(len(simplices)):
            # get center of current triangle
            center = misc.centroidTriangle(points[simplices[i]])
            # set normal and origin
            try:
                for index in range(3):
                    normal[index] = normals[i][index]
                    # to avoid self intersection use a centroid point a little bit away from face (in normal direction)
                    origin[index] = center[index] + 1e-5*normal[index]
            except:
                print(i)

            # method returns list ([hitPoint], hitParam, hitFace, hitTriangle, hitBary2, hitBary2)
            # value hitFace equals -1 iff no intersection was found
            hitResult = mfnObject.closestIntersection(origin, normal, om2.MSpace.kObject, threshold, False, accelParams = accel)
            # check for hit and save current triangle's ID
            if (hitResult[3] != -1):
                #tubeCandidates.extend([obj + ".f[" + str(i) + "]"])
                tubeCandidates.extend(["%s.f[%d]" % (obj, i)])

            try:
                # set progress bar
                if (not i % tempProgress):
                    cmds.text(progress, e = 1, label = 'Processing... (' + "{0:.1f}%)".format(float(i)/len(simplices)  * 100) ) + '%%)'
                    cmds.refresh(f = 1)
            except:
                pass
        cmds.select(tubeCandidates)

# creator function
def findTubeFacesCreator():
    return OpenMayaMPx.asMPxPtr( findTubeFaces() )

# syntax creator function
def findTubeFacesSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("t", "threshold", om.MSyntax.kDouble)
    syntax.addFlag("p", "progress", om.MSyntax.kString)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
