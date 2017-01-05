"""
Find exactly the faces who share no edge with any other face of the mesh. Works now only for triangulated Meshs.

**command:** cmds.findIsolatedFaces()

**Args:**
       :select(s): boolean (default True) indicating whether the faces should be selected
       :cleanup(c): perform a cleanup (delete isolated faces, default False)

       :returns: a list of face indices, which are isolated
"""


import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
import operator
import misc
import math

dot = lambda x, y: sum(map(operator.mul, x, y))
"""dot product as lambda function to speed up calculation"""
normalize = lambda v: map(operator.div, v, [math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])] * 3)
"""normalization of 3D-vector as lambda function to speed up calculation"""
EPSILON = 1e-5


class findIsolatedFaces(OpenMayaMPx.MPxCommand):
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def doIt(self, argList):
        # get objects from argument list
        try:
            objs = misc.getArgObj(self.syntax(), argList)
            if (len(objs) < 1):
                cmds.warning("There must be at least 1 selected object!")
                return
        except:
            cmds.warning("No object selected or wrong syntax!")
            return


        # parse arguments
        argData = om.MArgParser(self.syntax(), argList)
        select = argData.flagArgumentBool('select', 0) if (argData.isFlagSet('select')) else True
        cleanup = argData.flagArgumentBool('cleanup', 0) if (argData.isFlagSet('cleanup')) else False

        all_isolated_tris = []
        cmd_obj = []

        for obj in objs:
            # get faces
            tris = misc.getTriangles(obj)
            triRefs = misc.getTriangleEdgesReferences(tris)
            #coreTris = [i for i, tri in enumerate(tris) if ((len(triRefs[frozenset([tri[0], tri[1]])])>1) and (len(triRefs[frozenset([tri[0], tri[2]])])>1) and (len(triRefs[frozenset([tri[1], tri[2]])])>1))]
            isolatedTris = [i for i, tri in enumerate(tris) if ((len(triRefs[frozenset([tri[0], tri[1]])])<2) + (len(triRefs[frozenset([tri[0], tri[2]])])<2) + (len(triRefs[frozenset([tri[1], tri[2]])])<2)) >2]
            all_isolated_tris.append(isolatedTris)
            cmd_obj.extend(["{0}.f[{1}]".format(obj, i) for i in isolatedTris])

        if len(cmd_obj)>0:
            if select and not cleanup:
                cmds.select(cmd_obj)

            if cleanup:
                cmds.delete(cmd_obj)

        self.setResult(str(all_isolated_tris))


# creator function
def findIsolatedFacesCreator():
    return OpenMayaMPx.asMPxPtr(findIsolatedFaces())


# syntax creator function
def findIsolatedFacesSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("s", "select", om.MSyntax.kBoolean)
    syntax.addFlag("c", "cleanup", om.MSyntax.kBoolean)
    return syntax


# create button for shelf
def addButton(parentShelf):
    pass
