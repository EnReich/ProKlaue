"""
Module with a few helper functions for plugin *ProKlaue*.
"""

import maya.api.OpenMaya as om2
import maya.OpenMaya as om
import maya.cmds as cmds
import numpy as np
import operator
import math

RAD_TO_DEG = 180/math.pi
DEG_TO_RAD = math.pi/180
GRAD_TO_RAD = math.pi/180

I = np.matrix([[1,0,0],
              [0,1,0],
              [0,0,1]])


def getArgObj(syntax, argList):
    """
    Method to return list of objects from argument list (throws exception if no object is given or selected).

    :param syntax: Function to syntax definition of current command
    :type syntax: OpenMaya.MSyntax (api 1.0)
    :param argList: Argument list for command
    :raises AttributeError: Raised when no object is selected

    :returns: list of object names in argument list (or from current selection)
    """
    argData = om.MArgParser(syntax, argList)
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


def getPointsAsList(obj, worldSpace = True):
    """
    Method to return the points of an object's mesh in a list of lists with coordinates.

    :param obj: name of object
    :type obj: str
    :param worldSpace: should coordinates be in world space (with transform) or in local space (without transform); default True
    :type worldSpace: bool
    :returns: list of lists of coordinates (length 3) of the points of the objects mesh

    **Example:**
        .. code-block:: python

            from pk_src import misc
            cmds.polyCube()
            cmds.xform("pCube1", t=[1,0,0])
            misc.getPointsAsList("pCube1", worldSpace=True)
            # Result: [(0.5, -0.5, 0.5), (1.5, -0.5, 0.5), (0.5, 0.5, 0.5), (1.5, 0.5, 0.5), (0.5, 0.5, -0.5), (1.5, 0.5, -0.5), (0.5, -0.5, -0.5), (1.5, -0.5, -0.5)] #
            misc.getPointsAsList("pCube1", worldSpace=False)
            # Result: [(-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5)] #
    """

    xOrig = cmds.xform(obj+'.vtx[*]', q=True, ws=worldSpace, t=True)
    origPts = zip(xOrig[0::3], xOrig[1::3], xOrig[2::3])
    return origPts

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
        transform = np.matrix(cmds.xform(obj, q = 1, m = 1, ws =1)).reshape(4,4)[:-1, :-1]
        # multiply normals with transform, discard 4th value and organize them as list of 3 floats each
        #normals = [(n * transform).tolist()[0] for n in normals]
        normals = (normals * transform).tolist()
    return (normals)

def getFaceNormals2(obj, worldSpace = True):
    """
    Method to return all face normals. Uses the command
    `polyInfo <http://download.autodesk.com/us/maya/2011help/Commands/polyInfo.html>`_
    to access the normal information as string and parse them to a numpy array.
    Instead of a transformation matrix this method uses a calculated rotation matrix, and by that does not scale
    the normals.

    :param obj: name of object
    :type obj: string

    :returns: list of numpy arrays with the 3 float values for each normal

    **Example:**
        .. code-block:: python

            from pk_src import misc
            cmds.polyCube()
            # Result: [u'pCube1', u'polyCube1'] #
            misc.getFaceNormals2("pCube1")
            # Result: [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]] #
    """
    normalStrings = cmds.polyInfo(obj, fn = 1)
    normals = [np.fromstring(str(s[20:(len(s)-1)]), dtype=float, sep=" ").tolist() for s in normalStrings]
    # if normals should be in world space coordinates, one needs to apply the rotation of the object
    if (worldSpace):
        # get rotation matrix
        rotation_angles = cmds.xform(obj, q=True, ro=True, ws=True)
        rotation_order = cmds.xform(obj, q=True, roo=True, ws=True)
        rot_mat = getRotationMatrix(alpha=rotation_angles[0], beta=rotation_angles[1], gamma=rotation_angles[2],
                                    order=rotation_order, rad=False)[:-1, :-1]
        # multiply normals with transform
        normals = (normals * rot_mat.transpose()).tolist()
    return normals

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

def getTriangleEdges(triangles):
    """
    Method to return all edges of the given triangle list as dictionary. Given triangle must use indices to point list, so the key-value pairs only contain indices to the vertices' list; key is the index of the current vertex and for each outgoing edge the value contains an index to the corresponding vertex. Values are always lists of indices.

    :param triangles: list of triangle definitions (in index represenation, see *misc.getTriangles*)
    :type triangles: [[1, 2, 3], [1, 3, 4], ...]
    :returns: dict (keys: index of current vertex, values: list of vertices with existing edge)

    **Example:**
        .. code-block:: python

            from pk_src import misc
            cmds.polyCube()
            # Result: [u'pCube1', u'polyCube1'] # 
            cmds.polyTriangulate()
            # Result: [u'polyTriangulate1'] # 
            misc.getTriangleEdges(misc.getTriangles("pCube1"))
            # Result: {0: [1, 2, 4, 6, 7], 
                    1: [0, 2, 3, 7], 
                    2: [0, 1, 3, 4], 
                    3: [1, 2, 4, 5, 7], 
                    4: [0, 2, 3, 5, 6], 
                    5: [3, 4, 6, 7], 
                    6: [0, 4, 5, 7], 
                    7: [0, 1, 3, 5, 6]} # 
    """

    # initialize dictionary with edge connections
    edges = {}
    # for each triangle definition add list of edges
    for tri in triangles:
        # each vertex has 2 outgoing edges, if index already exist as key in dictionary, add the two other indices to the value-list and eliminate duplicates
        edges[tri[0]] = list(set(edges[tri[0]] + [tri[1], tri[2]])) if (tri[0] in edges) else [tri[1], tri[2]]
        edges[tri[1]] = list(set(edges[tri[1]] + [tri[0], tri[2]])) if (tri[1] in edges) else [tri[0], tri[2]]
        edges[tri[2]] = list(set(edges[tri[2]] + [tri[0], tri[1]])) if (tri[2] in edges) else [tri[0], tri[1]]
    return edges

def getTriangleEdgesCount(triangles):
    """
    Method to return the reference count of all  edges of the given triangle list as dictionary. Given triangle must use indices to point list, so the key-value pairs only contain indices to the vertices' list; key is the set of indices of the current vertex-pair for an edge and the value contains the count of triangles with a reference to that edge. Values are always integer. Only edges with at least 1 reference appear in the dict.

    :param triangles: list of triangle definitions (in index represenation, see *misc.getTriangles*)
    :type triangles: [[1, 2, 3], [1, 3, 4], ...]
    :returns: dict (keys: a frozenset with exactly 2 indices of vertices, values: count of triangles which reference the given edge)
    """

    # initialize dictionary with edge counts
    edgeCount = {}
    # for each triangle definition count references of edges
    for tri in triangles:
        edgeCount[frozenset([tri[0], tri[1]])] = edgeCount.get(frozenset([tri[0], tri[1]]), 0) + 1
        edgeCount[frozenset([tri[0], tri[2]])] = edgeCount.get(frozenset([tri[0], tri[2]]), 0) + 1
        edgeCount[frozenset([tri[1], tri[2]])] = edgeCount.get(frozenset([tri[1], tri[2]]), 0) + 1
    return edgeCount

def getTriangleEdgesReferences(triangles):
    """
    Method to return the references to face indices of all  edges of the given triangle list as dictionary. Given triangle must use indices to point list, so the key-value pairs only contain indices to the vertices' list; key is the set of indices of the current vertex-pair for an edge and the value contains a set of triangle indices with a reference to that edge. Values are always integer. Only edges with at least 1 reference appear in the dict.

    :param triangles: list of triangle definitions (in index represenation, see *misc.getTriangles*)
    :type triangles: [[1, 2, 3], [1, 3, 4], ...]
    :returns: dict (keys: a frozenset with exactly 2 indices of vertices, values: set of triangle indices which reference the given edge)
    """

    # initialize dictionary with edge counts
    edgeRefs = {}
    # for each triangle definition count references of edges
    for ind, tri in enumerate(triangles):

        if frozenset([tri[0], tri[1]]) in edgeRefs:
            edgeRefs[frozenset([tri[0], tri[1]])].add(ind)
        else:
            edgeRefs[frozenset([tri[0], tri[1]])] = {ind}

        if frozenset([tri[1], tri[2]]) in edgeRefs:
            edgeRefs[frozenset([tri[1], tri[2]])].add(ind)
        else:
            edgeRefs[frozenset([tri[1], tri[2]])] = {ind}

        if frozenset([tri[0], tri[2]]) in edgeRefs:
            edgeRefs[frozenset([tri[0], tri[2]])].add(ind)
        else:
            edgeRefs[frozenset([tri[0], tri[2]])] = {ind}

    return edgeRefs


def areaTriangle(vertices):
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

def alignYAxis():
    """Calculates the necessary rotations to align the y axis of one object with the position of another object, i.e. the local y-vector of the first selected objects will point towards the position (in worldspace) of the second selected object.

    If not exactly two objects are selected, a warning message will be displayed.
    """

    # normalization of 3d vector
    normalize = lambda v: map(operator.div, v, [math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])]*3)
    # sign function (returns +1 true iff given value is greater 0, -1 iff given value is less 0, and 0 if value equals 0)
    sign = lambda x: (x > 0) - (x < 0)

    objList = cmds.ls(orderedSelection = 1)
    if (len(objList) == 2):
        # vector FROM 1st TO 2nd selected object
        t = map(operator.sub, cmds.xform(objList[1], q = 1, t = 1, ws = 1), cmds.xform(objList[0], q = 1, t = 1, ws = 1))
        t = normalize(t)
        # calculate angles a and z (rotation around x and z axis)
        # calculation are derived from the transformation matrix: R*y = t where t is the new direction and y = (0,1,0). The three equations are: (1) -sin(c) = t_x, (2) cos(a)*cos(c) = t_y and (3) sin(a) = t_z
        a = math.asin(t[2])
        # the term sign(math.asin(...)) is necessary because the direction of rotation depends on the quadrant in which the target direction is (function z = math.atan2(math.asin(-t[0]), t[1]) does more or less the same but with visible inaccuracy)
        c = sign(math.asin(-t[0])) * math.acos(t[1] / math.cos(a))
        # set rotation angles (degree) for 1st object
        cmds.xform(objList[0], ro = map(operator.mul, [a, 0, c], [180/math.pi]*3), ws = 1)
    else:
        cmds.warning("Select exactly two coordinate systems!")


def getRotationMatrix(alpha, beta, gamma, order="xyz", rad=False):
    """Calculate a Rotation Matrix for given euler angles
    :param alpha: angle for rotation around x-axis
    :type alpha: float
    :param beta: angle for rotation around y-axis
    :type beta: float
    :param gamma: angle for rotation around z-axis
    :type gamma: float
    :param order: rotation order
    :type order: str
    :param rad: set true for angles given in radians
    :type rad: bool
    :returns: rotation matrix for the rotations in the given order

    **Example:**
        .. code-block:: python

            from pk_src import misc
            misc.getRotationMatrix(alpha=45, beta=60, gamma=10, order="xzy")
            # Result: matrix([[ 0.49240388,  0.55097853,  0.67376634,  0.        ],
            [ 0.17364818,  0.69636424, -0.69636424,  0.        ],
            [-0.85286853,  0.45989075,  0.24721603,  0.        ],
            [ 0.        ,  0.        ,  0.        ,  1.        ]]) #
            misc.getRotationMatrix(alpha=0, beta=90, gamma=0, order="xyz")
            # Result: matrix([[  6.12323400e-17,   0.00000000e+00,   1.00000000e+00,
               0.00000000e+00],
            [  0.00000000e+00,   1.00000000e+00,   0.00000000e+00,
               0.00000000e+00],
            [ -1.00000000e+00,   0.00000000e+00,   6.12323400e-17,
               0.00000000e+00],
            [  0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
               1.00000000e+00]]) #
    """
    if not rad:
        alpha *= GRAD_TO_RAD
        beta *= GRAD_TO_RAD
        gamma *= GRAD_TO_RAD

    rotx = np.matrix([[1, 0, 0, 0],
                      [0, math.cos(alpha), -math.sin(alpha), 0],
                      [0, math.sin(alpha), math.cos(alpha), 0],
                      [0, 0, 0, 1]])
    roty = np.matrix([[math.cos(beta), 0, math.sin(beta), 0],
                      [0, 1, 0, 0],
                      [-math.sin(beta), 0, math.cos(beta), 0],
                      [0, 0, 0, 1]])
    rotz = np.matrix([[math.cos(gamma), -math.sin(gamma), 0, 0],
                      [math.sin(gamma), math.cos(gamma), 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]])

    rot = np.matrix([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    for axis in order:
        if axis == "x":
            rot = rotx * rot
        elif axis == "y":
            rot = roty * rot
        elif axis == "z":
            rot = rotz * rot

    return rot


def getSkew(v):
    return np.matrix([[0, -v[2], v[1]],
                      [v[2], 0, -v[0]],
                      [-v[1], v[0], 0]])

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
