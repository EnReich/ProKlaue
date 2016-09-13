"""
.. module:: proKlaue
    :platform: Unix, Windows
    :synopsis: Master Plugin-File for Maya Autodesk

.. moduleauthor:: Enrico Reich

`ProKlaue <http://www.zv.uni-leipzig.de/service/kommunikation/medienredaktion/nachrichten.html?ifab_modus=detail&ifab_id=6004>`_ is a biomechanic research project at the Faculty of Veterinary Medicine of the University of Leipzig (Saxony, Germany). The primary goal is to be able to align bone models in a repeatable and deterministic way, as well as to export comparable measures of each bone model over a whole animation for further analysis.\n
To support the necessary work and to provide some useful routines inside the 3D-modelling and animation program *Maya Autodesk* the plugin *proKlaue* was written, which uses the `Python-API <http://download.autodesk.com/us/maya/2011help/CommandsPython/>`_ of Maya and Python-Libraries for numerical computation (`numpy <http://www.numpy.org/>`_, `scipy <https://www.scipy.org/>`_, `V-HACD <https://github.com/kmammou/v-hacd>`_). The Plugin introduces not only different scripts which are registered as commands inside Maya (for easier usage) but also a number of useful functions for general mesh-based tasks. There are a few additional functions like calculation of the convex hull, delaunay triangulation, intersection volume of multiple mesh objects and a cleanup process where one can extract a specific shell of a mesh (to eliminate possible entrapments inside a bone model caused by e.g. air, vessels or imaging errors).

Requirements
============
- Maya Autodesk 2013 (or newer)
- Numpy Package for Python2.7 (at least 0.12.0)
- Scipy Package for Python2.7 (at least 0.16.0)
- V-HACD Library

Installation of Numpy & Scipy for Maya
--------------------------------------
Maya Autodesk is shipped with its own Python2.7-Installation which does not make any use of already existing Python-Installations on the system. This includes all Python-Libraries as well. So in order to make Python-Libraries work directly inside Maya, the following steps are necessary:

- Installation of Python 2.7 for the current system
- Install *Numpy* and *Scipy* (either using the Linux install repository or the *pip*-command of python, e.g. 'pip install -i https://pypi.anaconda.org/carlkl/simple numpy')
- Search subdirectory 'site-packages' in Python2.7 install path
- Select and copy directories *numpy* & *scipy*
- Go to the Maya Directory (e.g. 'C://Program Files/Autodesk/Maya2014/Python/lib/site-packages') and insert copied files

After a restart of Maya the command 'import numpy' should be usable; if not the wrong version was installed.

Troubleshooting
^^^^^^^^^^^^^^^
- **command pip was not found**: navigate to the python-subdirectory *Scripts* ('cd "C://Program Files/Python27/Scripts/"') and try again. Or insert Path 'Python27/Scripts' to PATH-Variable
- **Maya Error: module 'numpy' not found**: the 'numpy' directory is in the wrong Maya subdirectory. It needs to be inside Maya's Python-Directory 'lib/site-packages/'
- **Maya throws dll-Error with reference to Win32**: wrong Python-Version and consequently wrong library versions are installed. Since Maya Autodesk 2014 there is only a 64bit version. This means that Python also needs to be installed as 64bit version. It is also advisable to update 'pip' ('python -m pip install -U pip') **before** installation of the libraries. Further the packages should be installed without any optimizations to avoid conflicts.

Installation and configuration of the Plugin
============================================
- Copy everything to 'Autodesk/maya<version>/bin/plug-ins' so that the file 'proKlaue.py' lies directly inside this directory
- Inside Maya: Windows --> Settings/Preferences --> Plugin-in Manager --> Refresh
- Search for the entry 'proKlaue.py' and check 'Loaded' and 'Auto load'

After the plugin has been successfully loaded a new shelf tab named 'ProKlaue' will appear.

At last, activate the following setting: Windows --> Settings/Preferences --> Preferences --> Setting --> Selection --> 'Track selection order'

To use the console commands directly inside Maya, one need to execute the Python command 'import maya.cmds as cmds' after each program restart (this can be done automatically by creating a file 'userSetup.py' inside the directory '/home/user/maya/version/scripts/' and write the line 'import maya.cmds as cmds' inside). All commands can then be executed by typing 'cmds.<cmd>' where *cmd* is a placeholder for the command's name.
"""

import maya.OpenMaya as om
import maya.OpenMayaMPx as OpenMayaMPx
import maya.cmds as cmds
import maya.mel as mel
import sys
import re

# list with all commands (command names must have the same name as their class AND module to be loaded without errors)
kPluginCmdName = ["centerPoint", "centroidPoint", "normalize", "eigenvector", "alignObj", "exportData", "rangeObj", "convexHull", "getShells", "findTubeFaces", "cleanup", "getVolume", "delaunay", "intersection", "adjustAxisDirection", "vhacd", "axisParallelPlane", "altitudeMap", "coordinateSystem"]

# current plugin version
pk_version = "0.3.1"

# Initialize the script plug-in
def initializePlugin(mobject):
    """Initialization method for Maya Autodesk plugin system; creates new shelf and registers each command."""

    # make sure the plug-in path is actually found by the python interpreter
    if (not len([p for p in sys.path if re.search('plug-ins', p)])):
        if (sys.platform == "linux" or sys.platform == "linux2"):
            sys.path.append([p for p in sys.path if re.search('bin$', p)][0] + '/plug-ins')
        else:
            sys.path.append([p for p in sys.path if re.search('bin$', p)][0] + '\\plug-ins')

    # import all sub-modules from directory 'pk_src' (print to console in case of error)
    for cmd in kPluginCmdName:
        try:
            exec('from pk_src import ' + cmd)
        except:
            sys.stderr.write("Could not find module 'pk_src.%s'\n" % cmd)
    # force reload of modules to avoid having maya to restart every time a class definition changes
    for cmd in kPluginCmdName:
        try:
            reload(sys.modules["pk_src." + cmd])
        except:
            sys.stderr.write("Error reloading " + cmd)
    # get maya plugin module
    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    mplugin.setVersion(pk_version)
    # get top level shelf to create new shelf layout where each command gets its own button to execute command
    topShelf = mel.eval('$nul = $gShelfTopLevel')
    # check if old reference still exists and delete it
    if (cmds.shelfLayout("ProKlaue", exists = 1)):
        cmds.deleteUI("ProKlaue", lay = 1)
    # create new shelf layout proKlaue
    proKlaue = cmds.shelfLayout("ProKlaue", parent = topShelf)
    # maya messes up the shelf optionVars shelfName1 --> recreate shelf names
    shelves = cmds.shelfTabLayout(topShelf, query=True, tabLabelIndex=True)
    for index, shelf in enumerate(shelves):
        cmds.optionVar(stringValue=("shelfName%d" % (index+1), str(shelf)))

    # register commands and add buttons to shelf
    for cmd in kPluginCmdName:
        try:
            mplugin.registerCommand( cmd, eval(cmd + "." + cmd + "Creator"), eval(cmd + "." + cmd + "SyntaxCreator") )
            exec(cmd + ".addButton('%s')" %(proKlaue))
        except:
            sys.stderr.write("Failed to register command or to add shelf button: %s\n" % cmd)

# Uninitialize the script plug-in
def uninitializePlugin(mobject):
    """Uninitialization method for Maya Autodesk plugin system; deregisters commands and removes shelf tab."""

    mplugin = OpenMayaMPx.MFnPlugin(mobject)
    # deregister command and delete shelfTab from UI
    for cmd in kPluginCmdName:
        try:
            mplugin.deregisterCommand( cmd )
        except:
            pass
            #sys.stderr.write( "Failed to unregister command: %s\n" % cmd )
    cmds.deleteUI("ProKlaue", lay = 1)
