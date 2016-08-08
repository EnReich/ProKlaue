"""
Module with a few helper functions for plugin *ProKlaue*.
"""

import maya.api.OpenMaya as om2
import maya.OpenMaya as om
import maya.cmds as cmds
import numpy as np
import operator

def getArgObj(syntax, argList):
    """
    Method to return list of objects from argument list (throws exception if no object is given or selected).

    :param syntax: Function to syntax definition of current command
    :type syntax: OpenMaya.MSyntax (api 1.0)
    :param argList: Argument list for command
    :raises AttributeError: Raised when no object is selected

    :returns: list of object names in argument list (or from current selection)
    """
    argData = om.MArgParser (syntax, argList)
    objStrings = []
    argData.getObjects(objStrings)
    # check if objects are selected or object is given as argument (only use first selection/argument)
    selList = cmds.ls(orderedSelection = True)
    try:
        obj = objStrings if (len(objStrings)) else selList
        #obj = selList[0] if (len(selList)) else objStrings[0]
    except:
        obj = []

    if (len(obj) == 0):
        raise AttributeError('No object selected')
    return obj

def getMesh(obj):
    """
    Method to return mesh object from dag path.

    :param obj: name of object
    :type obj: string
    :returns: MFnMesh object (api 2.0)
    """
    # put name of object into selection list
    selectionLs = om2.MSelectionList()
    selectionLs.add ( obj )
    # Get the dag path of the first item in the selection list
    dag = selectionLs.getDagPath(0)
    # create a Mesh functionset from our dag object
    return (om2.MFnMesh(dag))

def getPoints(obj, mfnObject = None, worldSpace = True):
    """
    Method to return MPointArray with points of object's mesh.

    :param obj: name of object
    :type obj: string
    :param mfnObject: mesh object
    :type mfnObject: MFnMesh
    :param worldSpace: should coordinates be in world space (with transform) or in local space (without transform); default True
    :type worldSpace: Boolean
    :returns: MPointArray (api 2.0)

    **Example:**
        .. code-block:: python

            from pk_src import misc
            cmds.polyCube()
            # Result: [u'pCube1', u'polyCube1'] #
            cmds.xform(t = [2,1,0])
            # vertices in local space (without transform)
            misc.getPoints("pCube1", worldSpace = 0)
            # Result: maya.api.OpenMaya.MPointArray([maya.api.OpenMaya.MPoint(-0.5, -0.5, 0.5, 1), maya.api.OpenMaya.MPoint(0.5, -0.5, 0.5, 1), maya.api.OpenMaya.MPoint(-0.5, 0.5, 0.5, 1), maya.api.OpenMaya.MPoint(0.5, 0.5, 0.5, 1), maya.api.OpenMaya.MPoint(-0.5, 0.5, -0.5, 1), maya.api.OpenMaya.MPoint(0.5, 0.5, -0.5, 1), maya.api.OpenMaya.MPoint(-0.5, -0.5, -0.5, 1), maya.api.OpenMaya.MPoint(0.5, -0.5, -0.5, 1)]) #
            # vertices in world space (with transform)
            misc.getPoints("pCube1", worldSpace = 1)
            # Result: maya.api.OpenMaya.MPointArray([maya.api.OpenMaya.MPoint(1.5, 0.5, 0.5, 1), maya.api.OpenMaya.MPoint(2.5, 0.5, 0.5, 1), maya.api.OpenMaya.MPoint(1.5, 1.5, 0.5, 1), maya.api.OpenMaya.MPoint(2.5, 1.5, 0.5, 1), maya.api.OpenMaya.MPoint(1.5, 1.5, -0.5, 1), maya.api.OpenMaya.MPoint(2.5, 1.5, -0.5, 1), maya.api.OpenMaya.MPoint(1.5, 0.5, -0.5, 1), maya.api.OpenMaya.MPoint(2.5, 0.5, -0.5, 1)]) #
    """
    if (mfnObject is None):
        # put name of object into selection list
        selectionLs = om2.MSelectionList()
        selectionLs.add ( obj )
        # Get the dag path of the first item in the selection list
        dag = selectionLs.getDagPath(0)
        # create a Mesh functionset from our dag object
        mfnObject = om2.MFnMesh(dag)
    # set space
    space = om2.MSpace.kWorld if (worldSpace == 1) else om2.MSpace.kObject
    # get Array of MPoint-Objects --> for better accessing reorganize MPoint-Array to array
    return (mfnObject.getPoints(space))

def getFaceNormals(obj, worldSpace = True):
    """
    Method to return all face normals. Uses the command `polyInfo <http://download.autodesk.com/us/maya/2011help/Commands/polyInfo.html>`_ to access the normal information as string and parse them to a numpy array

    :param obj: name of object
    :type obj: string

    :returns: list of numpy arrays with the 3 float values for each normal

    **Example:**
        .. code-block:: python

            from pk_src import misc
            cmds.polyCube()
            # Result: [u'pCube1', u'polyCube1'] #
            misc.getFaceNormals("pCube1")
            # Result: [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]] #
    """
    normalStrings = cmds.polyInfo(obj, fn = 1)
    normals = [ np.fromstring(str(s[20:(len(s)-1)]), dtype = float, sep = " ").tolist() for s in normalStrings]
    # if normals should be in world space coordinates, one needs to apply the transformation of the object
    if (worldSpace):
        # get transformation matrix and cut away 4th row/column
        transform = np.matrix(cmds.xform(obj, q = 1, m = 1)).reshape(4,4)[:-1]
        transform = transform.transpose()[:-1].transpose()
        # multiply normals with transform, discard 4th value and organize them as list of 3 floats each
        normals = [(n * transform).tolist()[0] for n in normals]
    return (normals)

def getTriangles(obj, mfnObject = None):
    """
    Method to return all triangles in object mesh as nested list. Length of list is equal to number of triangles for the object. Each list element is another list of 3 vertex indices which refer to the point set of the current object mesh.

    :param obj: name of object
    :type obj: string
    :param mfnObject: mesh object
    :type mfnObject: MFnMesh
    :returns: numpy.ndarray (e.g. array([[2,1,3],[2,3,4],[4,3,5],[4,5,6],...]) for a polygon cube)

    **Example:**
        .. code-block:: python

            from pk_src import misc
            cmds.polyCube()
            # Result: [u'pCube1', u'polyCube1'] #
            misc.getTriangles("pCube1")
            # Result: array([[0, 1, 2],
                   [2, 1, 3],
                   [2, 3, 4],
                   [4, 3, 5],
                   [4, 5, 6],
                   [6, 5, 7],
                   [6, 7, 0],
                   [0, 7, 1],
                   [1, 7, 3],
                   [3, 7, 5],
                   [6, 0, 4],
                   [4, 0, 2]]) #

    """
    if (mfnObject is None):
        # put name of object into selection list
        selectionLs = om2.MSelectionList()
        selectionLs.add ( obj )
        # Get the dag path of the first item in the selection list
        dag = selectionLs.getDagPath(0)
        # create a Mesh functionset from our dag object
        mfnObject = om2.MFnMesh(dag)
    # list of triangles indices
    tri = [x for x in mfnObject.getTriangles()[1]]
    return (np.squeeze(np.asarray([ [tri[i], tri[i+1], tri[i+2]] for i in range(len(tri))[0::3]] )) )

def areaTriangle (vertices):
    """
    Returns the area of a triangle given its three vertices.

    :param vertices: list of 3 vertices defining the triangle
    :returns: float value with area of triangle

    **Example:**
        .. code-block:: python

            from pk_src import misc
            cmds.polyCube()
            # Result: [u'pCube1', u'polyCube1'] #
            points = [[p.x, p.y, p.z] for p in misc.getPoints("pCube1")]
            misc.areaTriangle([points[0], points[1], points[2]])
            # Result: 0.5 #
    """
    AB = map(operator.sub, vertices[1], vertices[0])
    AC = map(operator.sub, vertices[2], vertices[0])
    # calculate area of triangle with half of the cross-product vector
    area = 0.5 * np.sqrt(   np.power(AB[1] * AC[2] - AB[2] * AC[1], 2) +
                            np.power(AB[2] * AC[0] - AB[0] * AC[2], 2) +
                            np.power(AB[0] * AC[1] - AB[1] * AC[0], 2) )
    return area

def centroidTriangle (vertices):
    """
    Returns the centroid of a triangle given its three vertices.

    :param vertices: list of 3 vertices defining the triangle
    :returns: list of 3 float values representing the 3D-coordinates of the triangle's centroid

    **Example:**
        .. code-block:: python

            from pk_src import misc
            cmds.polyCube()
            # Result: [u'pCube1', u'polyCube1'] #
            points = [[p.x, p.y, p.z] for p in misc.getPoints("pCube1")]
            misc.centroidTriangle([points[0], points[1], points[2]])
            # Result: [-0.16666666666666666, -0.16666666666666666, 0.5] #
    """
    return map(operator.mul, reduce(lambda x,y: map(operator.add, x, y), vertices), [1.0/3.0]*3)

def project(p, v):
    """
    Orthogonal projection of one vector onto another.

    :param p: vector to be projected
    :type p: [x,y,z]
    :param v: vector where p shall be projected to
    :type v: [x,y,z]
    :returns: orthogonal projection vector [x,y,z]
    """
    t1 = np.dot(p, v)
    t2 = np.dot(v, v)
    return (map(operator.mul, v, [t1*t2]*3))

def signedVolumeOfTriangle(p1, p2, p3, center = [0,0,0]):
    """Calculates signed volume of given triangle (volume of tetrahedron with triangle topped off at origin (0,0,0))

    :param p1: first point of triangle
    :type p1: [x,y,z]
    :param p2: second point of triangle
    :type p2: [x,y,z]
    :param p3: third point of triangle
    :type p3: [x,y,z]
    :param center: top of tetrahedron (default (0,0,0))
    :type center: [x,y,z]
    :returns: signed volume of tetrahedron

    **Example:**
        .. code-block:: python

            from pk_src import misc
            cmds.polyCube()
            # Result: [u'pCube1', u'polyCube1'] #
            points = [[p.x, p.y, p.z] for p in misc.getPoints("pCube1")]
            misc.signedVolumeOfTriangle(points[0], points[1], points[2])
            # Result: 0.08333333333333333 #

            # Volume of whole object
            triangles = misc.getTriangles("pCube1")
            reduce(lambda x,y: x+y, [misc.signedVolumeOfTriangle(points[tri[0]], points[tri[1]], points[tri[2]]) for tri in triangles])
            # Result: 1.0 #
    """
    if (center != [0,0,0]):
        p1 = map(operator.sub, p1, center)
        p2 = map(operator.sub, p2, center)
        p3 = map(operator.sub, p2, center)

    v123 = p1[0]*p2[1]*p3[2]
    v132 = p1[0]*p3[1]*p2[2]
    v213 = p2[0]*p1[1]*p3[2]
    v231 = p2[0]*p3[1]*p1[2]
    v312 = p3[0]*p1[1]*p2[2]
    v321 = p3[0]*p2[1]*p1[2]

    return (1.0/6.0 * reduce(lambda x,y: x+y, [v123, -v132, v312, -v213, v231, -v321]))
