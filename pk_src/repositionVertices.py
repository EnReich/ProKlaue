"""
Produces/Uses data to reposition an objects vertices to the same position of an identical object vertices.
When used in export mode (export = 1), this produces a file with data about the center point and the position
of 3 vertices (6 degrees of freedom in 3d).
When used in reposition mode (export = 0), the given file is used to align the given object to the same position. This
means to rotate and translate in the object in such a way, that after the transformation both objects are congruent
and aligned (objects have to be identical in the relative position of their vertices / identical surface mesh).


**command:** cmds.repositionVertices([obj], ex=1, f="./reposition_data.csv")

**Args:**
    :obj: string with object's name inside maya
    :exportMode(ex): export to file mode (ex=1) or reposition mode with data from file (ex=0)
    :file(f): file to export/import data to/from
    :saveTransformations(st): has only an effect in reposition mode, saves the transformations for repositioning
        in the given directory.
    :vertexIndices(vid): has only an effect in export mode, sets the indices of the vertices where to align the objects.
    :singleFiles(sf): save one single file for each transform node selected (default false).
        Has only effect in export mode.
    :dontApplyTransformations(da): transformations are calculated and saved but not applied (default false).
        Has only effect in reposition mode.
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
import misc
import numpy as np
import os
import random
import math
import csv

RAD_TO_DEG = 180/math.pi
DEG_TO_RAD = math.pi/180
I = np.matrix([[1,0,0],
              [0,1,0],
              [0,0,1]])

class repositionVertices(OpenMayaMPx.MPxCommand):
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
            cmds.warning("Wrong Argument!")
            return

        argData = om.MArgParser(self.syntax(), argList)

        if not argData.isFlagSet('exportMode'):
            cmds.warning("exportMode flag not set, dont know in which mode to operate.")
            return

        # read all arguments and set default values
        s_file = argData.flagArgumentString('file', 0) if (argData.isFlagSet('file')) else "./reposition_data.csv"
        export = argData.flagArgumentBool('exportMode', 0)
        singleFiles = argData.flagArgumentBool('singleFiles', 0) if (argData.isFlagSet('singleFiles')) else False
        dontApplyTransformations = argData.flagArgumentBool('dontApplyTransformations', 0) if (argData.isFlagSet('dontApplyTransformations')) else False

        if argData.isFlagSet('vertexIndices'):
            sampleVertices = False
            vtx_ind = argData.flagArgumentInt('vertexIndices', 0)
        else:
            sampleVertices = True

        if argData.isFlagSet('saveTransformations'):
            saveTransformations = True
            trans_file = os.path.abspath(argData.flagArgumentString('saveTransformations', 0))
        else:
            saveTransformations = False

        if export:
            rep_file_paths = []
            s_file = os.path.abspath(s_file)
            if not singleFiles:
                o_file = open(s_file, 'w')
                o_file.write("index, name, center, rotation, p0, p1, p2, p0_idx, p1_idx, p2_idx\n")
                rep_file_paths = [s_file]
            else:
                if not os.path.isdir(s_file):
                    os.makedirs(s_file)
            i = 0

            if not sampleVertices:
                idx = vtx_ind

            for obj in objs:
                if sampleVertices:
                    idx = random.sample(range(cmds.polyEvaluate(obj, v=1)), 3)

                if singleFiles:
                    # print formatObjString(obj)
                    path = os.path.abspath("{}/{}.csv".format(s_file, formatObjString(obj)))
                    o_file = open(path, 'w')
                    o_file.write("index, name, center, rotation, p0, p1, p2, p0_idx, p1_idx, p2_idx\n")
                    rep_file_paths.append(path)

                center = cmds.centerPoint(obj)
                rotation = cmds.alignObj(obj)
                p0 = cmds.xform(obj + ".vtx[{}]".format(idx[0]), q=1, t=1, ws=1)
                p1 = cmds.xform(obj + ".vtx[{}]".format(idx[1]), q=1, t=1, ws=1)
                p2 = cmds.xform(obj + ".vtx[{}]".format(idx[2]), q=1, t=1, ws=1)
                o_file.write(
                    '{},"{}","{}","{}","{}","{}","{}","{}","{}","{}"\n'.format(i, obj, center, rotation, p0, p1, p2,
                                                                               idx[0], idx[1], idx[2]))
                o_file.flush()

                if singleFiles:
                    o_file.close();

                i += 1

            if not singleFiles:
                o_file.close()

            self.setResult(rep_file_paths)

        else:
            csvfile = open(os.path.abspath(s_file), 'rb')
            reader = csv.reader(csvfile, delimiter=',', quotechar='"')
            header = reader.next()

            i = 0
            if saveTransformations:
                if not os.path.isdir(trans_file):
                    os.makedirs(trans_file)
                trans_file_paths = []

            for row in reader:
                obj_neu = objs[i]

                if saveTransformations:
                    trans_file_path = os.path.abspath("{}/tf_{}.csv".format(trans_file, formatObjString(obj_neu)))
                    trans_file_paths.append(trans_file_path)
                    t_file = open(trans_file_path, 'w')
                    t_file.write("type,X,Y,Z,pivot,order,r,ws\n")


                p0 = np.matrix(row[4]).reshape(3, 1)
                p1 = np.matrix(row[5]).reshape(3, 1)
                p2 = np.matrix(row[6]).reshape(3, 1)

                r01 = p1 - p0

                p0_neu = np.matrix(cmds.xform("{}.vtx[{}]".format(obj_neu, row[7]), q=1, t=1, ws=1)).reshape(3, 1)
                p1_neu = np.matrix(cmds.xform("{}.vtx[{}]".format(obj_neu, row[8]), q=1, t=1, ws=1)).reshape(3, 1)
                p2_neu = np.matrix(cmds.xform("{}.vtx[{}]".format(obj_neu, row[9]), q=1, t=1, ws=1)).reshape(3, 1)

                r01_neu = p1_neu - p0_neu

                R1 = getRotationFromAToB(r01_neu, r01)

                r02 = p2 - p0

                r02_neu = R1 * (p2_neu - p0_neu)

                r02_projection = getProjectionOnto(r02, r01)
                r02_neu_projection = getProjectionOnto(r02_neu, r01)

                r02_lot = r02 - r02_projection
                r02_neu_lot = r02_neu - r02_neu_projection

                R2 = getRotationFromAToB(r02_neu_lot, r02_lot)

                angles = getEulerAnglesToMatrix(R2 * R1)
                if not dontApplyTransformations:
                    cmds.xform(obj_neu, roo="xyz", p=1)
                    cmds.rotate(angles[0], angles[1], angles[2], obj_neu, worldSpace=1, r=1, pivot=p0_neu.A1)

                if saveTransformations:
                    t_file.write('rotation,"{}","{}","{}","{}","xyz","{}","{}"\n'.format(angles[0], angles[1],
                                                                                        angles[2], p0_neu.A1, 1, 1))
                t = p0 - p0_neu

                if not dontApplyTransformations:
                    cmds.move(t.A1[0], t.A1[1], t.A1[2], obj_neu, worldSpace=1, r=1)

                if saveTransformations:
                    t_file.write('translation,"{}","{}","{}","{}","0","{}","{}"\n'.format(t.A1[0], t.A1[1],
                                                                                           t.A1[2], 0, 1, 1))

                i += 1

                if saveTransformations:
                    t_file.flush()
                    t_file.close()

            csvfile.close()

            if saveTransformations:
                self.setResult(trans_file_paths)


# creator function
def repositionVerticesCreator():
    return OpenMayaMPx.asMPxPtr(repositionVertices())


# syntax creator function
def repositionVerticesSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("ex", "exportMode", om.MSyntax.kBoolean)
    syntax.addFlag("vid", "vertexIndices", om.MSyntax.kLong)
    syntax.addFlag("f", "file", om.MSyntax.kString)
    syntax.addFlag("st", "saveTransformations", om.MSyntax.kString)
    syntax.addFlag("sf", "singleFiles", om.MSyntax.kBoolean)
    syntax.addFlag("da", "dontApplyTransformations", om.MSyntax.kBoolean)
    return syntax


# create button for shelf
def addButton(parentShelf):
    pass


def getSkew(v):
    return np.matrix([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])

def getProjectionOnto(a, b):
    return (float(np.dot(a.A1, b.A1)) / np.dot(b.A1, b.A1)) * b

def getRotation(angle, v, rad=True):
    if not rad:
        angle *= DEG_TO_RAD
    v_norm = v / np.linalg.norm(v)
    return np.matrix([[(1 - math.cos(angle)) * v_norm.A1[0] ** 2 + math.cos(angle),
                       v_norm.A1[0] * v_norm.A1[1] * (1 - math.cos(angle)) - v_norm.A1[2] * math.sin(angle),
                       v_norm.A1[0] * v_norm.A1[2] * (1 - math.cos(angle)) + v_norm.A1[1] * math.sin(angle)],
                      [v_norm.A1[1] * v_norm.A1[0] * (1 - math.cos(angle)) + v_norm.A1[2] * math.sin(angle),
                       (1 - math.cos(angle)) * v_norm.A1[1] ** 2 + math.cos(angle),
                       v_norm.A1[1] * v_norm.A1[2] * (1 - math.cos(angle)) - v_norm.A1[0] * math.sin(angle)],
                      [v_norm.A1[2] * v_norm.A1[0] * (1 - math.cos(angle)) - v_norm.A1[1] * math.sin(angle),
                       v_norm.A1[2] * v_norm.A1[1] * (1 - math.cos(angle)) + v_norm.A1[0] * math.sin(angle),
                       (1 - math.cos(angle)) * v_norm.A1[2] ** 2 + math.cos(angle)]])

def getRotationFromAToB(a, b):
    """Calculate a rotation matrix to rotate the first vector in the place of the second vector. (In case of normed
        vectors, the vectors will overlap after rotating a with the rotation matrix).
        :param a: first vector, to be rotated
        :type a: np.matrix, shape (3,1)
        :param b: second vector, determines the position to achieve
        :type b: np.matrix, shape (3,1)
        :returns: rotation matrix for a rotation from a to b, as a np.matrix shape (3,3)
    """
    a_norm = (a / np.linalg.norm(a))
    b_norm = (b / np.linalg.norm(b))
    v = np.matrix(np.cross(a_norm.transpose(), b_norm.transpose())).reshape(3, 1)
    s = np.linalg.norm(v)
    c = np.dot(a_norm.A1, b_norm.A1)
    if c != -1:
        R = I + getSkew(v.A1) + getSkew(v.A1) * getSkew(v.A1) * (1 / (1 + c))
    else:
        v_norm = v / s
        R = np.matrix(
            [[2 * v_norm.A1[0] ** 2 - 1, v_norm.A1[0] * v_norm.A1[1] * 2, v_norm.A1[0] * v_norm.A1[2] * 2],
             [v_norm.A1[1] * v_norm.A1[0] * 2, 2 * v_norm.A1[1] ** 2 - 1, v_norm.A1[1] * v_norm.A1[2] * 2],
             [v_norm.A1[2] * v_norm.A1[0] * 2, v_norm.A1[2] * v_norm.A1[1] * 2, 2 * v_norm.A1[2] ** 2 - 1]])

    return R

def getEulerAnglesToMatrix(m):
    """Calculate the Euler Angles for a given rotation matrix
        :param m: rotation matrix
        :type m: np.matrix, shape (3,3)
        :returns: [alpha, beta, gamma] for rotation around global X, Y, Z axis in that order
    """
    alpha = math.atan2(m[2, 1], m[2, 2]) * RAD_TO_DEG
    beta = math.atan2(-m[2, 0],
                      math.sqrt(math.pow(m[2, 1], 2) + math.pow(m[2, 2], 2))) * RAD_TO_DEG
    gamma = math.atan2(m[1, 0], m[0, 0]) * RAD_TO_DEG
    return [alpha, beta, gamma]


def formatObjString(name):
    return name.replace(":", "_")
