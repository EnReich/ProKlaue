import maya.OpenMayaMPx as OpenMayaMPx
import maya.OpenMaya as om
import maya.cmds as cmds
from functools import partial

class projectionArea(OpenMayaMPx.MPxCommand):
   windowID = "projA"

   def __init__(self):
      OpenMayaMPx.MPxCommand.__init__(self)

   def __cancelCallback(*pArgs):
      if cmds.window(projectionArea.windowID, exists = True):
        cmds.deleteUI(projectionArea.windowID)

   def __applyCallback(self, textF, *pArgs):
      options = {"text":cmds.textField(textF, q = 1, text = 1)}
      cmds.projectionArea(**options)

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
      # get objects from argument list
      try:
         obj = misc.getArgObj(self.syntax(), argList)
         if (len(obj) != 2):
            cmds.warning("There must be exactly 2 selected objects (object and plane)!")
            return
         # get vertices and face normals of object and plane (in local space to use transformation matrices of each frame and avoid overhead using the getPoints-method)
         # use 4D-Vectors for transformation of points
         obj_vtx, plane_vtx = [[p.x, p.y, p.z, p.w] for p in misc.getPoints(obj[0], worldSpace=0)], [
            [p.x, p.y, p.z, p.w] for p in misc.getPoints(obj[1], worldSpace=0)]
         obj_n, plane_n = misc.getFaceNormals(obj[0], worldSpace=0), misc.getFaceNormals(obj[1], worldSpace=0)
         # check if one of the selected object has plane properties (4 vertices, 1 normal)
         if not ((len(obj_vtx) == 4 or len(plane_vtx) == 4) and (len(obj_n) == 1 or len(plane_n) == 1)):
            cmds.warning("None of the objects is a plane (4 vtx, 1 normal)!")
            return
         # in case the selection order is wrong, switch the variables (obj[0] is the object, obj[1] is the plane)
         if (len(obj_vtx) == 4 and len(obj_n) == 1):
            obj[0], obj[1] = obj[1], obj[0]
            obj_vtx, plane_vtx = plane_vtx, obj_vtx
            obj_n, plane_n = plane_n, obj_n

         plane_n = plane_n[0]
         plane_centroid = map(operator.div, reduce(lambda x, y: map(operator.add, x, y), plane_vtx), [4.0] * 4)
      except:
         cmds.warning("No object selected!")
         return

      # parse arguments
      argData = om.MArgParser(self.syntax(), argList)
      s_file = argData.flagArgumentString('file', 0) if (argData.isFlagSet('file')) else ""
      threshold = argData.flagArgumentDouble('threshold', 0) if (argData.isFlagSet('threshold')) else 10.0
      animation = argData.flagArgumentBool('anim', 0) if (argData.isFlagSet('anim')) else False
      # get keyframes of object and export data for whole animation if flag is set
      keyframes = sorted(list(set(cmds.keyframe(q=1))) if (animation) else [cmds.currentTime(q=1)])
      if (len(keyframes) == 1 and animation):
         cmds.warning("There seems to be no keyframe information, but animation flag is true!")
         return
      # if export shall be done over the animation, use only file export (because of data size)
      if (s_file == "" and animation):
         cmds.warning("When animation flag is true, there must be a given file name!")
         return

      # get triangles of object model
      obj_tri = misc.getTriangles(obj[0])
      # make sure each face normal belongs to one triangle
      if (len(obj_n) != len(obj_tri)):
         cmds.warning(
            "Number of face normals and number of triangles are not equal! Please triangulate mesh and retry!")
         return

      # open file stream
      if (s_file != ""):
         o_file = open(s_file, 'w')
         o_file.write("Frame\tN\tTx\tTy\tTz\td\n")

      # list of replace operations to parse list of lists to tab separated string using reduce operation
      repl = ('[', ''), (']', ''), (',', ''), (' ', '\t')

      # loop over all keyframes and gather the distances
      for key in keyframes:
         cmds.currentTime(key)
         ### in the following lines the object vertices/normals and the plane vertices/normals are calculated for the current frame (to have world space vectors). Instead of requesting the information from the maya-API, standard matrix multiplication is used
         # get 4x4 transformation matrix of object
         transform_obj = np.matrix(cmds.xform(obj[0], q=1, m=1)).reshape(4, 4)
         # transform object vertices to world space and discard 4th dimension
         # TODO: only use vertices from last iteration
         obj_vtx_ws = (obj_vtx * transform_obj).transpose()[:-1].transpose().tolist()

         # multiply transformation matrix of plane and parent object (when plane is grouped under object, the world space normal and centroid needs to be adjusted)
         parent = cmds.listRelatives(obj[1], p=1)[0]
         if (parent == obj[0]):
            tmp = np.matrix(cmds.xform(obj[1], q=1, m=1)).reshape(4, 4) * transform_obj
         else:
            tmp = np.matrix(cmds.xform(obj[1], q=1, m=1)).reshape(4, 4) * np.matrix(
               cmds.xform(parent, q=1, m=1)).reshape(4, 4)
         # plane centroid transformation (after which discard 4th dimension)
         plane_centroid_ws = (plane_centroid * tmp).tolist()[0][:-1]
         # cut 4th row/column to perform 3d vector multiplication with 3x3 rotation matrix (speedup)
         tmp = tmp[:-1].transpose()[:-1].transpose()
         # transform the plane's normal
         plane_n_ws = (plane_n * tmp).tolist()[0]

         # cut 4th row/column and transform object normals to world space (no translation necessary)
         transform_obj = transform_obj[:-1].transpose()[:-1].transpose()
         obj_n_ws = (obj_n * transform_obj).tolist()

         # remove all triangles whose face normal points in same direction as plane normal (backface culling), but keep index of normal/triangle
         obj_tri_bc = [np.append(i, obj_tri[i]) for i, n in enumerate(obj_n_ws) if dot(plane_n_ws, n) < 0]

         # get centroid points of each triangle (to construct a ray in direction of the plane)
         obj_centroid = [[tri[0]] + misc.centroidTriangle([obj_vtx_ws[tri[1]], obj_vtx_ws[tri[2]], obj_vtx_ws[tri[3]]])
                         for tri in obj_tri_bc]

         # build acceleration structure
         mfnObject = misc.getMesh(obj[0])
         accel = mfnObject.autoUniformGridParams()

         # list to store points and their distance to the plane
         altitudeMap = []

         # now construct ray from face centroids (in inverse direction of plane normal) and intersect with object mesh
         # if intersection is found with object, ray cannot intersect plane; in case of no intersection, calculate distance of centroid to plane
         ray_dir = om2.MFloatVector(map(operator.mul, plane_n_ws, [-1] * 3))
         for c in obj_centroid:
            # move origin a little bit away from centroid to avoid self intersection
            origin = om2.MFloatPoint(c[1:]) + EPSILON * ray_dir

            # method returns list ([hitPoint], hitParam, hitFace, hitTriangle, hitBary2, hitBary2)
            # value hitFace equals -1 iff no intersection was found
            hitResult = mfnObject.closestIntersection(origin, ray_dir, om2.MSpace.kWorld, threshold, False,
                                                      accelParams=accel)
            # if no intersection is found, get distance of centroid to plane
            if (hitResult[3] == -1):
               # distance from centroid to plane is the length of the projection v (centroid plane to centroid triangle) onto unit normal vector of plane
               d = dot(map(operator.sub, c[1:], plane_centroid_ws), plane_n_ws)
               # add centroid as 3D point and distance to plane to altitude map
               if (d <= threshold):
                  altitudeMap.append([key] + c + [d])

         # if file name is given, write altitude map
         if (s_file != ""):
            # for each item in altitude map, remove brackets and comma
            for item in altitudeMap:
               # o_file.write("%s\n" % string.replace(string.replace(string.strip(str(item), '[]'), ',', ''), ' ', '\t'))
               o_file.write("%s\n" % reduce(lambda a, kv: a.replace(*kv), repl, str(item)))

      # if animation flag is False, return altitudeMap
      if (s_file != ""):
         print('Altitude Map written to file \'%s\'!' % s_file)
      if (not animation):
         self.setResult(str(altitudeMap))

def projectionAreaCreator():
   return OpenMayaMPx.asMPxPtr( projectionArea() )
def projectionAreaSyntaxCreator():
   syntax = om.MSyntax()
   syntax.addFlag("t", "text", om.MSyntax.kString)
   return syntax
def addButton(parentShelf):
   cmds.shelfButton(parent = parentShelf, i = "pythonFamily.png",
      c=projectionArea().createUI, imageOverlayLabel = "proAr", ann="calculates the projected area onto a plane")