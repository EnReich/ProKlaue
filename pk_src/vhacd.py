"""
Uses the V-HACD library (https://github.com/kmammou/v-hacd) to calculate an approximate convex decomposition for a number of selected or given objects. Because Maya's object representation is usually a surface mesh, it is sometimes necessary to convert the object to a solid geometry (especially when using complex objects or boolean operations). Using convex decomposition avoids problems concerning holes or non-manifold geometry. The original object will be approximated by a finite number of convex polyhedra organized in one group per original object.

The available properties and settings of V-HACD are taken directly from the `description page <http://kmamou.blogspot.de/2014/12/v-hacd-20-parameters-description.html>`_.

Script makes use of the *wrl2ma*-command under *maya/bin* to parse to and from different formats. Apparently there are some old parser-inconsistencies which may produce warning or error messages, which (as far as observed) have no visible effect on the produces mesh. Also the path to the directories *maya/bin* for *wrl2ma* and *plug-ins/bin* for V-HACD is guessed through the environment path variable of the operating system.

**NOTE:** There are observed cases where V-HACD was not able to create a real solid geometry for a given mesh (without any topological holes) but rather creates a convex decomposition where the summed volume of the convex parts is smaller than the volume of the original mesh, i.e. the decomposition creates a topologically different model where the inner area is hollow. Therefore one should always verify the output mesh and in case of a faulty decomposition choose different parameter settings (a smaller *depth*-value usually avoids this problem).

**see also:** :ref:`getVolume`, :ref:`intersection`

**command:** cmds.vhacd([obj], tmp = '~/', exe = '../maya/bin/plug-ins/bin/testVHACD', res = 100000, d = 20, con = 0.001, pd = 4, chd = 4, a = 0.05, b = 0.05, g = 0.0005, pca = False, m = 0, vtx = 64, vol = 0.0001)

**Args:**
    :tmp(temporaryDir): directory to save temporary created files (needs read and write access). If no path is given the temporary files will be written into the user home directory
    :executable(exe): absolute path to the executable V-HACD file. If no path is given maya/bin/plug-ins/bin/testVHACD is used.
    :resolution(res): maximum number of voxels generated during the voxelization stage (10.000 - 64.000.000, default: 100.000)
    :depth(d): maximum number of clipping stages. During each split stage, all the model parts (with a concavity higher than the user defined threshold) are clipped according the "best" clipping plane (1 - 32, default: 20)
    :concavity(con): maximum concavity (0.0 - 1.0, default: 0.001)
    :planeDownsampling(pd): controls the granularity of the search for the "best" clipping plane (1 - 16, default: 4)
    :convexHullDownsampling(chd): controls the precision of the convex-hull generation process during the clipping plane selection stage (1 - 16, default: 4)
    :alpha(a): controls the bias toward clipping along symmetry planes (0.0 - 1.0, default: 0.05)
    :beta(b): controls the bias toward clipping along revolution axes (0.0 - 1.0, default: 0.05)
    :gamma(g): maximum allowed concavity during the merge stage (0.0 - 1.0, default: 0.0005)
    :normalizeMesh(pca): enable/disable normalizing the mesh before applying the convex decomposition (True/False, default: False)
    :mode(m): voxel-based approximate convex decomposition (0, default) or tetrahedron-based approximate convex decomposition (1)
    :maxNumVerticesPerCH(vtx): controls the maximum number of triangles per convex-hull (4 - 1024, default: 64)
    :minVolumePerCH(vol): controls the adaptive sampling of the generated convex-hulls (0.0 - 0.01, default: 0.0001)

:returns: list of Maya group names where each group corresponds with the approximate convex decomposition of one object (structure is: group name | sub group | convex mesh)
"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
import maya.mel as mel
import subprocess   # needed for system call to V-HACD
import os           # needed for environment path
from distutils.spawn import find_executable     # needed to check for executable v-hacd
import re
import sys
import misc
import platform

class vhacd(OpenMayaMPx.MPxCommand):
    mayaBin = ""
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    @staticmethod
    def readArgs(argData):
        """
        Read arguments passed to command, set default values and check for plausibility (range constrictions). Returns a dictionary structure with the short names of each argument or 'None' in case of error.
        """
        # get maya binary directory (needed for format parser bin/wrl2ma)
        if platform.system() == "Windows":
            vhacd.mayaBin = [p for p in os.environ['PATH'].split(';') if re.search('/[m,M]aya\d*/bin$', p) or re.search('\\\\[m,M]aya\d*\\\\bin$', p)]
        else:
            vhacd.mayaBin = [p for p in os.environ['PATH'].split(':') if re.search('maya[^/]*/bin', p) or re.search('maya[^(\\\\)]*[(\\\\)]bin', p)]
        if (len(vhacd.mayaBin) < 1):
            print vhacd.mayaBin
            cmds.error('maya/bin directory not found!')
            return
        else:
            vhacd.mayaBin = os.path.abspath(vhacd.mayaBin[0])

        # read all arguments and set default values
        tmp = os.path.abspath(argData.flagArgumentString('temporaryDir', 0)) if (argData.isFlagSet('temporaryDir')) \
            else os.path.abspath(os.path.expanduser('~') + '/')

        exe = ""
        if (argData.isFlagSet('executable')):
            exe = os.path.abspath(argData.flagArgumentString('executable', 0))
        else:
            exe = os.path.abspath(vhacd.mayaBin + "/plug-ins/bin/testVHACD.exe") if platform.system()=="Windows"\
                else os.path.abspath(vhacd.mayaBin + "/plug-ins/bin/testVHACD")

        res = argData.flagArgumentInt('resolution', 0) if (argData.isFlagSet('resolution')) else 100000
        d = argData.flagArgumentInt('depth', 0) if (argData.isFlagSet('depth')) else 20
        con = argData.flagArgumentDouble('concavity', 0) if (argData.isFlagSet('concavity')) else 0.001
        pd = argData.flagArgumentInt('planeDownsampling', 0) if (argData.isFlagSet('planeDownsampling')) else 4
        chd = argData.flagArgumentInt('convexHullDownsampling', 0) if (argData.isFlagSet('convexHullDownsampling')) else 4
        a = argData.flagArgumentDouble('alpha', 0) if (argData.isFlagSet('alpha')) else 0.05
        b = argData.flagArgumentDouble('beta', 0) if (argData.isFlagSet('beta')) else 0.05
        g = argData.flagArgumentDouble('gamma', 0) if (argData.isFlagSet('gamma')) else 0.0005
        norm = argData.flagArgumentBool('normalizeMesh', 0) if (argData.isFlagSet('normalizeMesh')) else False
        mode = argData.flagArgumentBool('mode', 0) if (argData.isFlagSet('mode')) else 0
        vtx = argData.flagArgumentInt('maxNumVerticesPerCH', 0) if (argData.isFlagSet('maxNumVerticesPerCH')) else 64
        vol = argData.flagArgumentDouble('minVolumePerCH', 0) if (argData.isFlagSet('minVolumePerCH')) else 0.0001

        # check each range constriction and print warnings messages in case of violation
        rangeViolation = False
        # check write access to given tmp-directory
        if (not os.access(tmp, os.W_OK)):
            rangeViolation = True
            cmds.warning("V-HACD: no write access to given tmp directory (" + tmp + ")")
        if (not find_executable(exe)):
            rangeViolation = True
            cmds.warning("V-HACD: could not find V-HACD executable (" + exe + ")")
        if (res < 10000 or res > 64000000):
            rangeViolation = True
            cmds.warning("V-HACD: resolution is not in range 10.000 - 64.000.000 (" + str(res) + ")")
        if (d < 1 or d > 32):
            rangeViolation = True
            cmds.warning("V-HACD: depth is not in range 1 - 32 (" + str(d) + ")")
        if (con < 0.0 or con > 1.0):
            rangeViolation = True
            cmds.warning("V-HACD: concavity is not in range 0.0 - 1.0 (" + str(con) + ")")
        if (pd < 1 or pd > 16):
            rangeViolation = True
            cmds.warning("V-HACD: planeDownsampling is not in range 1 - 16 (" + str(pd) + ")")
        if (chd < 1 or chd > 16):
            rangeViolation = True
            cmds.warning("V-HACD: convexHullDownsampling is not in range 1 - 16 (" + str(chd) + ")")
        if (a < 0.0 or a > 1.0):
            rangeViolation = True
            cmds.warning("V-HACD: alpha is not in range 0.0 - 1.0 (" + str(a) + ")")
        if (b < 0.0 or b > 1.0):
            rangeViolation = True
            cmds.warning("V-HACD: beta is not in range 0.0 - 1.0 (" + str(b) + ")")
        if (g < 0.0 or g > 1.0):
            rangeViolation = True
            cmds.warning("V-HACD: gamma is not in range 0.0 - 1.0 (" + str(g) + ")")
        if (norm != True and norm != False):
            rangeViolation = True
            cmds.warning("V-HACD: normalizeMesh is not TRUE/FALSE (" + str(norm) +")")
        if (mode < 0 or mode > 1):
            rangeViolation = True
            cmds.warning("V-HACD: mode is not 0/1 (" + str(mode) + ")")
        if (vtx < 4 or vtx > 1024):
            rangeViolation = True
            cmds.warning("V-HACD: maxNumVerticesPerCH is not in range 4 - 1024 (" + str(vtx) + ")")
        if (vol < 0.0 or vol > 0.01):
            rangeViolation = True
            cmds.warning("V-HACD: minVolumePerCH is not in range 0.0 - 0.01 (" + str(vol) + ")")

        if (rangeViolation):
            return None

        else:
            return {'tmp' : tmp, 'exe' : exe, 'res' : res, 'd' : d, 'con' : con, 'pd' : pd, 'chd' : chd, 'a' : a, 'b' : b, 'g' : g, 'pca' : norm, 'mode' : mode, 'vtx' : vtx, 'vol' : vol}

    def doIt(self, argList):
        # get objects from argument list
        try:
            selList = misc.getArgObj(self.syntax(), argList)
        except:
            cmds.warning("No object selected!")
            return

        # read and check arguments
        args = vhacd.readArgs(om.MArgParser(self.syntax(), argList))
        if (args == None):
            cmds.error("V-HACD: one or more arguments are invalid!")
            return

        # list to hold names of convex decomposition result groups
        objCD = []
        # save processes to parallelize calculations
        processes = []
        # loop over all objects: select, export and convex decomposition
        for i,obj in enumerate(selList):
            cmds.select(obj)
            tmpFile = os.path.abspath('{0}/tmp{1}'.format(args['tmp'], i))
            # export current object as *.obj file format
            cmds.file(tmpFile + '.obj', es = 1, type = "OBJexport", f = 1)
            # system call to vhacd library with all parameters (use non-blocking subprocess)
            exeDir, exeName = os.path.split(args['exe'])
            p = subprocess.Popen([exeName,
                '--input', tmpFile + '.obj',
                '--output', tmpFile + '_out.obj',
                '--log', tmpFile + '.log',
                '--resolution', str(args['res']),
                '--depth', str(args['d']),
                '--concavity', str(args['con']),
                '--planeDownsampling', str(args['pd']),
                '--convexhullDownsampling', str(args['chd']),
                '--alpha', str(args['a']),
                '--beta', str(args['b']),
                '--gamma', str(args['g']),
                '--pca', str(args['pca']),
                '--mode', str(int(args['mode'])),
                '--maxNumVerticesPerCH', str(args['vtx']),
                '--minVolumePerCH', str(args['vol']),
            ], executable=args['exe'])
            # save process, current object and index
            processes.append((p, obj, i))

        # loop over all created processes, wait till each one is completed and import result into maya
        for p, obj, i in processes:
            p.wait()
            tmpFile = os.path.abspath('{0}/tmp{1}'.format(args['tmp'], i))
            #import OBJ
            rnn = cmds.file(tmpFile+'_out.obj', type='OBJ', i=True, rnn=True, ns = obj+'_vhacd')
            cmds.polySeparate(rnn[0], ch=False)

            # use eval and mel-exclusive command 'catchQuiet' to avoid error messages popping up during file import
            # cause of error messages: (ma-parser of Maya since version 2012)
            # mel.eval(' catchQuiet (` file -ra 1 -type "mayaAscii" -rpr "{0}_vhacd" -pmt 0 -i "{1}_out.ma" `) '.format(obj, tmpFile))
            # rename created group for convenience
            # objCD.append(cmds.rename(obj + '_vhacd_root', obj + '_vhacd'))

            objCD.append(rnn[0])

            # delete all temporarily created files
            # cmds.sysFile(tmpFile + '.mtl', delete = 1)
            # cmds.sysFile(tmpFile + '.obj', delete = 1)
            # cmds.sysFile(tmpFile + '.wrl', delete = 1)
            # cmds.sysFile(tmpFile + '.log', delete = 1)
            # cmds.sysFile(tmpFile + '_out.ma', delete = 1)

        # set name of groups (which hold convex decomposition) as result of command
        for o in objCD:
            self.appendToResult(o)

# creator function
def vhacdCreator():
    return OpenMayaMPx.asMPxPtr( vhacd() )

# syntax creator function
def vhacdSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("tmp", "temporaryDir", om.MSyntax.kString)
    syntax.addFlag("exe", "executable", om.MSyntax.kString)
    syntax.addFlag("res", "resolution", om.MSyntax.kLong)
    syntax.addFlag("d", "depth", om.MSyntax.kLong)
    syntax.addFlag("con", "concavity", om.MSyntax.kDouble)
    syntax.addFlag("pd", "planeDownsampling", om.MSyntax.kLong)
    syntax.addFlag("chd", "convexHullDownsampling", om.MSyntax.kLong)
    syntax.addFlag("a", "alpha", om.MSyntax.kDouble)
    syntax.addFlag("b", "beta", om.MSyntax.kDouble)
    syntax.addFlag("g", "gamma", om.MSyntax.kDouble)
    syntax.addFlag("pca", "normalizeMesh", om.MSyntax.kBoolean)
    syntax.addFlag("m", "mode", om.MSyntax.kBoolean)
    syntax.addFlag("vtx", "maxNumVerticesPerCH", om.MSyntax.kLong)
    syntax.addFlag("vol", "minVolumePerCH", om.MSyntax.kDouble)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass


def splitObjSequence(input_file, ouput_file_prefix):
    input = open(input_file, 'r')
    i = 0
    v_count = 0
    v_ind_begin = 0
    dest = None
    for line in input:
        if line[0:2] == 'o ':
            if dest:
                dest.close()
            dest = open(ouput_file_prefix+'_'+str(i)+'.obj', 'w')
            v_ind_begin += v_count
            i += 1
        if line[0:2] == 'v ':
            #vertex line
            v_count += 1
        if line[0:2] == 'f ':
            #extract vertex indices and decrement it (cause prev. vertices dont show up in the current file)
            indices_str = [str(int(s)-v_ind_begin) for s in line[2:].split() if s.isdigit()]
            nline = 'f ' + ' '.join(indices_str)+'\n'
        else:
            nline = line
        dest.write(nline)
    dest.close()

