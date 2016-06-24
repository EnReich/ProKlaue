"""
Calculates the eigenvectors and eigenvalues of the covariance matrix of all points in current object's mesh. The eigenvector with the largest eigenvalue corresponds to the first axis defined in axis order, second largest to second axis and third largest to third axis. Command is used by :ref:`alignObj` and :ref:`exportData`.
Command only accepts 'transform' nodes and will only be applied to the first object of the current selection.

The calculation of the covariance matrix is defined as:

.. math::
    C = [c_{i,j}] = \\biggl[ \\biggl (  \\frac{1}{a^H}\\sum\\limits_{k=0}^{n-1}\\frac{a^k}{12}(9m_i^km_j^k + p_i^kp_j^k + q_i^kq_j^k + r_i^kr_j^k) \\biggr ) - m_i^Hm_j^H \\biggr]

where :math:`m^H = \\frac {1}{a^H}\\sum\\limits_{k=0}^{n-1}a^km^k` is the centroid of the convex hull with :math:`m^i = \\frac{p^i+q^i+r^i}3` as centroid of triangle :math:`i` and the surface of the convex hull :math:`a^H = \\sum\\limits_{k=0}^{n-1}a^k`. The area of triangle :math:`k` with its is vertices :math:`\\Delta p^kq^kr^k` is defined as :math:`a^k`.

The eigenvectors and eigenvalue of :math:`C` are calculated using *numpy.linalg.eigh*.

**see also:** :ref:`alignObj`, :ref:`exportData`

**command:** cmds.eigenvector([obj], ao = 'yzx', f = False)

**Args:**
    :obj: string with object's name inside maya
    :axisOrder(ao): string to define axis order of eigenvectors (default 'yzx')
    :fast (f): boolean flag to indicate if calculation should use convex hull; faster but inaccurate (default False)

:returns: list of 9 float values corresponding to first eigenvector ([0:3]), second eigenvector ([3:6]) and third eigenvector ([6:9])

**Example:**
    .. code-block:: python

        cmds.polyTorus()
        # Result: [u'pTorus1', u'polyTorus1'] #
        cmds.eigenvector()
        # Result: [5.465342261024642e-10, -0.609576559498125, 0.7927272028323672, 1.0, 1.3544498855821985e-09, 3.520841396209562e-10, -1.288331507658196e-09, 0.7927272028323671, 0.6095765594981248] #

"""

import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.api.OpenMaya as om2
import maya.cmds as cmds
import numpy as np
from scipy.spatial import ConvexHull
import operator
import misc

class eigenvector(OpenMayaMPx.MPxCommand):
    def __init__(self):
        OpenMayaMPx.MPxCommand.__init__(self)

    def __covariance(self, obj, fast):
        """Returns the 3D-covariance matrix (3x3) of the given object over all points or only of the points in convex hull (if flag 'fast' is true)

        :param obj: name of object as string
        :param fast: if true all calculations will be done using the convex hull which is much faster but less accurate
        :returns: numpy.matrix object with shape (3,3)
        """
        # get and cast array of MPoint-Objects to ndarray structure (needed for ConvexHull)
        pSet = np.squeeze(np.asarray([[p.x, p.y, p.z] for p in misc.getPoints(obj)]))
        simplices = None

        if (not fast):
            simplices = misc.getTriangles(obj)
        else:
            # execute convex hull algorithm on point set of current object
            hull = ConvexHull(pSet)
            simplices = hull.simplices

        # initialize array with area and centroid of each triangle (to avoid reallocation of array)
        area_t = [0] * len(simplices)
        centroid_t = [[0,0,0]] * len(simplices)
        # centroid hull will calculate the centroid of the convex hull
        centroid_hull = [0,0,0]
        hull_area = 0.0
        # iterate over each triangle
        for i, tri in enumerate(pSet[simplices]):
            # get area and centroid of current triangle
            area_t[i] = misc.areaTriangle(tri)
            hull_area += area_t[i]
            centroid_t[i] = misc.centroidTriangle(tri)
            # centroid of hull is weighted mean of triangle centroids
            t = map(operator.mul, centroid_t[i], [area_t[i]]*3)
            centroid_hull = map(operator.add, centroid_hull, t)

        centroid_hull = map(operator.div, centroid_hull, [hull_area]*3)
        # initialize 3x3 covariance matrix as numpy matrix object
        mCov = np.matrix( ( (0.0,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,0.0)) )

        # calculate covariance matrix
        for i in range(0,3):
            for j in range(0,3):
                sum = 0.0
                for index, tri in enumerate(pSet[simplices]):
                    sum += area_t[index]/12 * (9 * centroid_t[index][i]*centroid_t[index][j] +
                            tri[0][i]*tri[0][j] + tri[1][i]*tri[1][j] + tri[2][i]*tri[2][j])
                mCov[i,j] = (1.0 / hull_area * sum - centroid_hull[i]*centroid_hull[j])

        return mCov

    def __orderBasisVectors(self, m, eVals, order):
        """Reorganizes list of basis vectors to match the given order.

        :param m: matrix of basis vectors (column-wise)
        :param eVals: the eigenvalues of the eigenvectors given in m
        :param order: axis order as string with length 3 and characters x,y,z (e.g. 'yzx')
        """
        options = {'x':0, 'y':1, 'z':2}
        # find largest eigenvalue
        axis = eVals.index(max(eVals))

        # swap column 0 with axis (if neccessary)
        if (axis != options[order[0]]):
            m[0,options[order[0]]],m[0,axis] = m[0,axis],m[0,options[order[0]]]
            m[1,options[order[0]]],m[1,axis] = m[1,axis],m[1,options[order[0]]]
            m[2,options[order[0]]],m[2,axis] = m[2,axis],m[2,options[order[0]]]
            # swap eigenvalues
            eVals[options[order[0]]],eVals[axis] = eVals[axis],eVals[options[order[0]]]
        # find second largest eigenvalue (in list without value at index 1)
        xAxis = eVals.index(max(eVals[:options[order[0]]] + eVals[(options[order[0]]+1):]))

        # swap column 1 with axis (if not already in correct place)
        if (xAxis != options[order[1]]):
            m[0,options[order[1]]],m[0,xAxis] = m[0,xAxis],m[0,options[order[1]]]
            m[1,options[order[1]]],m[1,xAxis] = m[1,xAxis],m[1,options[order[1]]]
            m[2,options[order[1]]],m[2,xAxis] = m[2,xAxis],m[2,options[order[1]]]
            eVals[options[order[1]]],eVals[xAxis] = eVals[xAxis],eVals[options[order[1]]]

        # make sure, that the eigenvectors with the two largest eigenvalues are always pointing in positive world direction
        for i in range(0,2):
            if (m[options[order[i]], options[order[i]]] < 0):
                m[:, options[order[i]]] = - m[:, options[order[i]]]
        # third eigenvector will be cross product of first and second vector (right-handed coordinate system)
        # to ensure right-handed coordinate system, the first vector needs to be lexicografic in front of second vector (xy, xz, yz)
        m[:, options[order[2]]] = np.cross(m[:, min(options[order[0]], options[order[1]])].transpose(), m[:, max(options[order[0]], options[order[1]])].transpose()).transpose()

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
        # parse arguments and get flags
        argData = om.MArgParser (self.syntax(), argList)
        axisOrder = argData.flagArgumentString('axisOrder', 0) if (argData.isFlagSet('axisOrder')) else "yzx"
        fast = argData.flagArgumentBool('fast', 0) if (argData.isFlagSet('fast')) else False
        # get eigenvectors of covariance matrix
        w, v = np.linalg.eigh(self.__covariance(obj, fast))
        # order basis vectors (biggest eigenvalue corresponds to first axis, 2nd biggest to second axis, etc)
        self.__orderBasisVectors(v, w.tolist(), axisOrder)
        # set results via MScriptUtil (to avoid having the results converted to strings)
        util = om.MScriptUtil()
        util.createFromList(v.getA1().tolist(), v.size)
        self.setResult(om.MDoubleArray(util.asDoublePtr(), v.size))

# creator function
def eigenvectorCreator():
    return OpenMayaMPx.asMPxPtr( eigenvector() )

# syntax creator function
def eigenvectorSyntaxCreator():
    syntax = om.MSyntax()
    syntax.setObjectType(om.MSyntax.kStringObjects)
    syntax.addFlag("ao", "axisOrder", om.MSyntax.kString)
    syntax.addFlag("f", "fast", om.MSyntax.kBoolean)
    return syntax

# create button for shelf
def addButton(parentShelf):
    pass
