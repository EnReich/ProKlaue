import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
import misc
import numpy as np
import os
import csv


class applyTransformationFile(OpenMayaMPx.MPxCommand):
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def doIt(self, argList):
        # get objects from argument list (size >= 2)
        try:
            objs = misc.getArgObj(self.syntax(), argList)
            for i in range(len(objs)):
                if cmds.objectType(objs[i]) != 'transform':
                    cmds.warning("{} is not of type transform!".format(objs[i]))
                    cmds.error("Object is not of type transform!")
                    return
        except:
            cmds.warning("Not suitable object(s) selected!")
            return

        argData = om.MArgParser(self.syntax(), argList)

        # read all arguments and set default values
        path = argData.flagArgumentString('file', 0) if (argData.isFlagSet('file')) else "./transformation.csv"
        file = open(os.path.abspath(path), 'rb')
        reader = csv.reader(file, delimiter=',', quotechar='"')
        header = reader.next()

        for row in reader:
            type = row[0]

            x = float(row[1])
            y = float(row[2])
            z = float(row[3])

            if row[4] == "0":
                p = False
            else:
                p = [float(i) for i in row[4].replace("[", "").replace("]", "").split()]

            order = row[5]
            r = bool(row[6])
            ws = bool(row[7])

            if type == 'translation':
                cmds.move(x, y, z, objs, r=r, ws=ws)

            if type == 'rotation':
                cmds.xform(objs, p=1, roo=order)
                cmds.rotate(x, y, z, objs, r=r, ws=ws, p=p)

        file.close()



# creator function
def applyTransformationFileCreator():
    return OpenMayaMPx.asMPxPtr(applyTransformationFile())


# syntax creator function
def applyTransformationFileSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("f", "file", om.MSyntax.kString)
    return syntax


# create button for shelf
def addButton(parentShelf):
    pass

