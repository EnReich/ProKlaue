.. ProKlaue documentation master file, created by
   sphinx-quickstart2 on Thu Jun  9 12:40:36 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


.. _main:

Welcome to ProKlaue's documentation!
====================================

.. automodule:: proKlaue

.. _commands:

Commands
---------

.. toctree::
   :maxdepth: 4

   pk_src

.. _sec-misc:

Misc
-----

.. toctree::

   collision_tet_tet <pk_src.collision_tet_tet>
   intersection_tet_tet <pk_src.intersection_tet_tet>
   misc <pk_src.misc>

HowTo: Add new commands to plugin *ProKlaue*
--------------------------------------------
The file *proKlaue.py* is the master-file of the plugin where all submodules will be individually imported and registered as commands. To keep the general structure and clarity one should always use separate files/modules for each new command. Additionally the file name, module name and command name should be equal to simplify the registration steps.

To add a new command *cmds.foo* implemented in the source file *foo.py* to the plugin, one only needs to add the name *foo* to the list of commands *kPluginCmdName* in *proKlaue.py* (l. 57) and it will be automatically imported, registered with syntax and button definition:

.. code-block:: python

   kPluginCmdName = ["centerPoint", "centroidPoint", "normalize", "eigenvector",
      "alignObj", "exportData", "rangeObj", "convexHull", "getShells",
      "findTubeFaces", "cleanup", "getVolume", "delaunay", "intersection",
      "adjustAxisDirection", "vhacd", "foo"]

The necessary parts of a new command need to be defined in a source file *foo.py*:

.. code-block:: python
   :emphasize-lines: 6,9,30,35,37,41

   import maya.OpenMayaMPx as OpenMayaMPx
   import maya.OpenMaya as om
   import maya.cmds as cmds
   from functools import partial

   class foo(OpenMayaMPx.MPxCommand):
      windowID = "wFoo"

      def __init__(self):
         OpenMayaMPx.MPxCommand.__init__(self)

      def __cancelCallback(*pArgs):
         if cmds.window(exportData.windowID, exists = True):
         cmds.deleteUI(exportData.windowID)

      def __applyCallback(self, textF, *pArgs):
         options = {"text":cmds.textField(textF, q = 1, text = 1)}
         cmds.foo(**options)

      def createUI (self, *pArgs):
         if cmds.window(self.windowID, exists = True):
            cmds.deleteUI(self.windowID)
            cmds.window(self.windowID, title = "fooUI", sizeable = True, resizeToFitChildren = True)
            cmds.rowColumnLayout(numberOfColumns = 2)
            text = cmds.textField(visible = True, width = 140)
            cmds.button(label = "apply", command = partial(self.__applyCallback, text), width = 100 )
            cmds.button(label = "cancel", command = self.__cancelCallback, width = 100)
            cmds.showWindow()

      def doIt(self, argList):
         argData = om.MArgParser (self.syntax(), argList)
         text = argData.flagArgumentString("text", 0)
         print(text)

   def fooCreator():
      return OpenMayaMPx.asMPxPtr( foo() )
   def fooSyntaxCreator():
      syntax = om.MSyntax()
      syntax.addFlag("t", "text", om.MSyntax.kString)
      return syntax
   def addButton(parentShelf):
      cmds.shelfButton(parent = parentShelf, i = "pythonFamily.png",
         c=foo().createUI, imageOverlayLabel = "foo", ann="do something")

All emphasized lines are required definitions: class **foo** inherited from *OpenMayaMPx.MPxCommand*, its constructor **foo.__init__**, the function **foo.doIt** which will be triggered at command invocation, function **fooCreator** which just returns a class instance, function **fooSyntaxCreator** which returns the syntax definition and function **addButton** where a shelf button under the shelf tab *ProKlaue* can be defined.

In this echo-server example a command button is defined with an input text field and two buttons (*apply* and *cancel*) inside the user interface. By pressing button *apply* the text field content will be echoed to the console whereas *cancel* closes the user interface. The same effect (without the user interface) can be achieved by typing *cmds.foo(t = 'echo')*.


Indices and tables
------------------


* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
