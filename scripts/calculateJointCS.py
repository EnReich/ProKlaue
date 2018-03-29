# script to calculate joint cs on principal curvature directions
# one has to select first the bone which will get the axis for flexion (probably the distal bone)
# second the counterpart bone (the proximal bone) and third an object which indicates the direction towards
# the center of the body (left/right counterpart, could be also a cylinder or a box though)

from pk_src import misc
import maya.cmds as cmds
import numpy as np
import math
import scipy.optimize
import scipy.interpolate
import scipy.spatial
import sklearn
from sklearn import decomposition
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import Pipeline
from timeit import default_timer as timer

# important settings
threshold = 0.3                                # threshold for defining of the joint surface, cow 0.3, horse 6.5, 4.5
radius =  0.9                                  # radius to average the principal curvature (from the saddle), cow 0.9,
                                               # horse 6.5, 5.5
save_dir = "C:/Users/Kai/Documents/tmp"        # save dir for information file (used for plots later)

# more settings
order = 5                       # order of polynom used to interpolate joint surface
regression_method = "ts"        # regression estimator
RANSAC_SPECIFIERS = ["rs", "ransac"]
TS_SPECIFIERS = ["ts", "theil-sen", "theilsen"]
interpolation_order = 3         # order of bivariate spline used to interpolate joint surface in the radius
left = "auto"                   # whether the first selected bone is the left bone (might be able to detect it
                                # automatically by name)
axis_used = ["auto"]            # one can specify if to use the minimal or maximal curvature or some automatic setting


# material settings
set_materials = True            # whether to set materials for regions used to calculate joint surface and curvature
shading_grp_surface = "shadingGrpJointSurface"  # name of the shading grp for the region to calculate the joint surface
shading_grp_curvature = "shadingGrpCurvatures"  # for the average of surface curvature
shading_grp_bones = "shadingGrpBones"       # for the bones
shading_grp_axis1 = "shadingGrpAxis1"       # for bodyfixed axis1
shading_grp_axis2 = "shadingGrpAxis2"       # for bodyfixed axis2
shading_grp_axisRef = "shadingGrpAxisRef"   # for reference axis
shading_grp_saddle = "shadingGrpSaddle"     # for saddle point

# reset shading for the bones with the bones shading grp
reset_shading_for_0 = False
reset_shading_for_1 = False


start = timer()

# calculate the principal curvature for a given bivariate polynomial spline,
# surface is x(u,v)=x, y(u,v)=v, z(u,v)=P(u,v),
# tangent space is spanned by (1,0,df/du) = (1,0,df/dx) and (0,1,df/dv) = (0,1,df/dy), with
# N = (1,0,df/dx) x (0,1,df/dy)
# shape operator is calculated by I^-1*II
# tangent space basis is returned in the second component
def shape_operator_spline(spline, points):
    x = points[:,0]
    y = points[:,1]

    f_u = spline.__call__(x=x, y=y, dx=1, dy=0, grid=False)
    f_uu = spline.__call__(x=x, y=y, dx=2, dy=0, grid=False)
    f_v = spline.__call__(x=x, y=y, dx=0, dy=1, grid=False)
    f_vv = spline.__call__(x=x, y=y, dx=0, dy=2, grid=False)
    f_uv = spline.__call__(x=x, y=y, dx=1, dy=1, grid=False)

    # first fundamental form
    E = 1+f_u**2
    F= f_u*f_v
    G = 1+f_v**2

    # second fundamental form
    div_second_form = ((1.+f_u**2+f_v**2)**(1./2))
    L = f_uu/div_second_form
    M = f_uv/div_second_form
    N = f_vv/div_second_form

    # np.stack([a11, a12, a21, a22], axis=1).reshape(-1, 2, 2)
    shape_operator = [np.linalg.inv(np.array([E[i],F[i],F[i],G[i]]).reshape(2,2)).dot(np.array([L[i],M[i],M[i],N[i]]).reshape(2,2)) for i in range(len(E))]
    tangent_basis = [[np.array([1,0,f_u[i]]), np.array([0,1,f_v[i]])] for i in range(len(E))]

    return shape_operator, tangent_basis


# calculate the principal curvature for a given bivariate polynomial as coefficient matrix,
# surface is x(u,v)=x, y(u,v)=v, z(u,v)=P(u,v),
# tangent space is spanned by (1,0,df/du) = (1,0,df/dx) and (0,1,df/dv) = (0,1,df/dy), with
# N = (1,0,df/dx) x (0,1,df/dy)
# shape operator is calculated by I^-1*II
# tangent space basis is returned in the second component
def shape_operator_polynomial(C_as_matrix, points):
    x = points[:,0]
    y = points[:,1]

    # gradient
    C_dx_matrix = np.polynomial.polynomial.polyder(C_as_matrix, axis=0)
    C_dy_matrix = np.polynomial.polynomial.polyder(C_as_matrix, axis=1)
    C_dxdx_matrix = np.polynomial.polynomial.polyder(C_dx_matrix, axis=0)
    C_dydy_matrix = np.polynomial.polynomial.polyder(C_dy_matrix, axis=1)
    C_dxdy_matrix = np.polynomial.polynomial.polyder(C_dx_matrix, axis=1)

    f_u = np.polynomial.polynomial.polyval2d(x, y, C_dx_matrix)
    f_uu = np.polynomial.polynomial.polyval2d(x, y, C_dxdx_matrix)
    f_v = np.polynomial.polynomial.polyval2d(x, y, C_dy_matrix)
    f_vv = np.polynomial.polynomial.polyval2d(x, y, C_dydy_matrix)
    f_uv = np.polynomial.polynomial.polyval2d(x, y, C_dxdy_matrix)

    # first fundamental form
    E = 1+f_u**2
    F= f_u*f_v
    G = 1+f_v**2

    # second fundamental form
    div_second_form = ((1.+f_u**2+f_v**2)**(1./2))
    L = f_uu/div_second_form
    M = f_uv/div_second_form
    N = f_vv/div_second_form

    # np.stack([a11, a12, a21, a22], axis=1).reshape(-1, 2, 2)
    shape_operator = [np.linalg.inv(np.array([E[i],F[i],F[i],G[i]]).reshape(2,2)).dot(np.array([L[i],M[i],M[i],N[i]]).reshape(2,2)) for i in range(len(E))]
    tangent_basis = [[np.array([1,0,f_u[i]]), np.array([0,1,f_v[i]])] for i in range(len(E))]

    return shape_operator, tangent_basis


# calculate the shape operator at given points for a given bivariate polynomial spline,
# surface is x(u,v)=x, y(u,v)=v, z(u,v)=P(u,v)
def shape_operator_spline2(spline, points):
    x = points[:,0]
    y = points[:,1]

    f_u = spline.__call__(x=x, y=y, dx=1, dy=0, grid=False)
    f_uu = spline.__call__(x=x, y=y, dx=2, dy=0, grid=False)
    f_v = spline.__call__(x=x, y=y, dx=0, dy=1, grid=False)
    f_vv = spline.__call__(x=x, y=y, dx=0, dy=2, grid=False)
    f_uv = spline.__call__(x=x, y=y, dx=1, dy=1, grid=False)

    # first fundamental form
    E = 1+f_u**2
    F= f_u*f_v
    G = 1+f_v**2

    # second fundamental form
    div_second_form = ((1.+f_u**2+f_v**2)**(1./2))
    L = f_uu/div_second_form
    M = f_uv/div_second_form
    N = f_vv/div_second_form

    # shape operator
    div_shape = E*G-F**2
    a11 = (M*F-L*G)/div_shape
    a12 = (L*F-M*E)/div_shape
    a21 = (N*F-M*G)/div_shape
    a22 = (M*F-N*E)/div_shape

    # np.stack([a11, a12, a21, a22], axis=1).reshape(-1, 2, 2)
    shape_operator = np.concatenate(np.array([a11, a12, a21, a22]).reshape(-1, 4, 1), axis=1).reshape(-1, 2, 2)
    return shape_operator

# see: http://stackoverflow.com/questions/11317579/surface-curvature-matlab-equivalent-in-python
def surfature(X,Y,Z):
# where X, Y, Z matrices have a shape (lr+1,lb+1)

    #First Derivatives
    Xv,Xu=np.gradient(X)
    Yv,Yu=np.gradient(Y)
    Zv,Zu=np.gradient(Z)

    #Second Derivatives
    Xuv,Xuu=np.gradient(Xu)
    Yuv,Yuu=np.gradient(Yu)
    Zuv,Zuu=np.gradient(Zu)

    Xvv,Xuv=np.gradient(Xv)
    Yvv,Yuv=np.gradient(Yv)
    Zvv,Zuv=np.gradient(Zv)

    #Reshape to 1D vectors
    nrow=(lr+1)*(lb+1) #total number of rows after reshaping
    Xu=Xu.reshape(nrow,1)
    Yu=Yu.reshape(nrow,1)
    Zu=Zu.reshape(nrow,1)
    Xv=Xv.reshape(nrow,1)
    Yv=Yv.reshape(nrow,1)
    Zv=Zv.reshape(nrow,1)
    Xuu=Xuu.reshape(nrow,1)
    Yuu=Yuu.reshape(nrow,1)
    Zuu=Zuu.reshape(nrow,1)
    Xuv=Xuv.reshape(nrow,1)
    Yuv=Yuv.reshape(nrow,1)
    Zuv=Zuv.reshape(nrow,1)
    Xvv=Xvv.reshape(nrow,1)
    Yvv=Yvv.reshape(nrow,1)
    Zvv=Zvv.reshape(nrow,1)

    Xu=np.c_[Xu, Yu, Zu]
    Xv=np.c_[Xv, Yv, Zv]
    Xuu=np.c_[Xuu, Yuu, Zuu]
    Xuv=np.c_[Xuv, Yuv, Zuv]
    Xvv=np.c_[Xvv, Yvv, Zvv]

    #% First fundamental Coeffecients of the surface (E,F,G)
    E=np.einsum('ij,ij->i', Xu, Xu)
    F=np.einsum('ij,ij->i', Xu, Xv)
    G=np.einsum('ij,ij->i', Xv, Xv)

    m=np.cross(Xu,Xv,axisa=1, axisb=1)
    p=sqrt(np.einsum('ij,ij->i', m, m))
    n=m/np.c_[p,p,p]

    #% Second fundamental Coeffecients of the surface (L,M,N)
    L= np.einsum('ij,ij->i', Xuu, n)
    M= np.einsum('ij,ij->i', Xuv, n)
    N= np.einsum('ij,ij->i', Xvv, n)

    #% Gaussian Curvature
    K=(L*N-M**2)/(E*G-L**2)
    K=K.reshape(lr+1,lb+1)

    #% Mean Curvature
    H = (E*N + G*L - 2*F*M)/(2*(E*G - F**2))
    H = H.reshape(lr+1,lb+1)

    #% Principle Curvatures
    Pmax = H + sqrt(H**2 - K)
    Pmin = H - sqrt(H**2 - K)

    return Pmax,Pmin


def makeAxis(origin, direction,
             cylinder_name="Cylinder",
             cone_name="Cone",
             cylinder_length=3,
             cylinder_size=0.025,
             cone_length=1,
             cone_size=0.1,
             shading_grp = ""):
    direct = direction/np.linalg.norm(direction)

    cylinder = cmds.polyCylinder(name=cylinder_name)
    cmds.scale(cylinder_size, cylinder_length, cylinder_size, cylinder)
    r = misc.getRotationFromAToB(a=np.matrix([0, 1, 0]).reshape(3, 1),
                                 b=np.matrix(direct).reshape(3, 1))
    m = np.matrix(cmds.xform(cylinder, q=1, ws=1, m=1)).reshape(4, 4).transpose()
    m_new = np.matrix(np.r_[np.c_[r, [0, 0, 0]], [[0, 0, 0, 1]]]) * m
    cmds.xform(cylinder, m=m_new.transpose().A1, ws=1)
    cmds.move(origin[0], origin[1], origin[2], cylinder, absolute=True)

    cone = cmds.polyCone(name=cone_name)

    #shader grps
    if (shading_grp != ""):
        if not cmds.objExists(shading_grp):
            shading_grp = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shading_grp)
        else:
            if not cmds.objectType(shading_grp) == "shadingEngine":
                while (cmds.objExists(shading_grp)):
                    shading_grp = "{0}1".format(shading_grp)

                shading_grp = cmds.sets(renderable=True, noSurfaceShader=True, empty=True,
                                                name=shading_grp)

        cmds.sets(cone[0], fe=shading_grp )
        cmds.sets(cylinder[0], fe=shading_grp)


    #position and scale
    cmds.scale(cone_size, cone_length, cone_size, cone)
    m = np.matrix(cmds.xform(cone, q=1, ws=1, m=1)).reshape(4, 4).transpose()
    m_new = np.matrix(np.r_[np.c_[r, [0, 0, 0]], [[0, 0, 0, 1]]]) * m
    cmds.xform(cone, m=m_new.transpose().A1, ws=1)
    pos_cone = origin + (cylinder_length + cone_length) * direct
    cmds.move(pos_cone[0],
              pos_cone[1],
              pos_cone[2], cone, absolute=True)
    cmds.parent(cone[0], cylinder[0])


    return cylinder, cone

print "Time: {}".format(timer()-start)
print "Finding close point pairs"

# get point to the 2 bones who are selected, build kd trees
objs = cmds.ls(sl=1) # first bone is the lower bone or the bone which has the axis for flexion
p0 = np.array(misc.getPointsAsList(objs[0], worldSpace=True))
p1 = np.array(misc.getPointsAsList(objs[1], worldSpace=True))
p = [p0, p1]
t0 = scipy.spatial.KDTree(p[0])
t1 = scipy.spatial.KDTree(p[1])
t = [t0, t1]
direction_up = np.average(p1, axis=0)-np.average(p0, axis=0)
direction_up /= np.linalg.norm(direction_up)


if(left == "auto"):
    if objs[0].find("_links")>0 or objs[0].find("_left")>0:
        left = True
    else:
        left = False

if len(objs)>2:
    direction_in = np.average(misc.getPointsAsList(objs[2], worldSpace=True), axis=0)-np.average(p0, axis=0)
    direction_in /= np.linalg.norm(direction_in)

# open save file
if save_dir != "":
    path = "{}/cs-{}-{}.csv".format(save_dir, objs[0].split(":")[0], objs[1].split(":")[0])
    save_file = open(path, 'w')
    save_file.write('variable,value\n')
    save_file.write('"objs[0]","{}"\n'.format(objs[0]))
    save_file.write('"objs[1]","{}"\n'.format(objs[1]))
    save_file.write('"TM[0]","{}"\n'.format(cmds.xform(objs[0], q=1, ws=1, m=1)))
    save_file.write('"TM[1]","{}"\n'.format(cmds.xform(objs[1], q=1, ws=1, m=1)))
    save_flag = True
else:
    save_flag = False


# visual indication for bones (overrides shadergrps)
if(set_materials):
    if not cmds.objExists(shading_grp_bones):
        shading_grp_bones=cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shading_grp_bones)
    else:
        if not cmds.objectType(shading_grp_bones)=="shadingEngine":
            while(cmds.objExists(shading_grp_bones)):
                shading_grp_bones = "{0}1".format(shading_grp_bones)

            shading_grp_bones = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shading_grp_bones)

    if(reset_shading_for_0):
        cmds.sets(objs[0], fe=shading_grp_bones)
    if(reset_shading_for_1):
        cmds.sets(objs[1], fe=shading_grp_bones)


# find pairs of points who are not further away than a given threshold
rIntersection = t[0].query_ball_tree(other=t[1], r=threshold)
idx0 = np.array(list(set([i for i in range(len(p0)) if rIntersection[i]])))
idx1 = np.array(list(set(np.uint32(np.concatenate(rIntersection)))))
idx = [idx0, idx1]

# selection of used vertices
# cmds.select(clear=True)
# for i in idx0: cmds.select("{}.vtx[{}]".format(objs[0], i), add=True)
# for i in idx1: cmds.select("{}.vtx[{}]".format(objs[1], i), add=True)

# visual indication
if(set_materials):
    if not cmds.objExists(shading_grp_surface):
        shading_grp_surface=cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shading_grp_surface)
    else:
        if not cmds.objectType(shading_grp_surface)=="shadingEngine":
            while(cmds.objExists(shading_grp_surface)):
                shading_grp_surface = "{0}1".format(shading_grp_surface)

            shading_grp_surface = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shading_grp_surface)

    faces_idx0 = cmds.polyListComponentConversion(["{}.vtx[{}]".format(objs[0], i) for i in idx0], fromVertex=1, toFace=1, internal=1)
    faces_idx1 = cmds.polyListComponentConversion(["{}.vtx[{}]".format(objs[1], i) for i in idx1], fromVertex=1, toFace=1, internal=1)
    cmds.sets(faces_idx0, fe=shading_grp_surface)
    cmds.sets(faces_idx1, fe=shading_grp_surface)


    if not cmds.objExists(shading_grp_saddle):
        shading_grp_saddle=cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shading_grp_saddle)
    else:
        if not cmds.objectType(shading_grp_saddle)=="shadingEngine":
            while(cmds.objExists(shading_grp_saddle)):
                shading_grp_saddle = "{0}1".format(shading_grp_saddle)

            shading_grp_saddle = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shading_grp_saddle)


print "Time: {}".format(timer()-start)

print "Transform coordinates"

# pca on the data points to transform the coordinates

p0_scope = p[0][idx[0]]
p1_scope = p[1][idx[1]]
p_scope = [p0_scope, p1_scope]

pca0 = decomposition.PCA(n_components=3)
pca0.fit(p_scope[0])
pca1 = decomposition.PCA(n_components=3)
pca1.fit(p_scope[1])

pca = [pca0, pca1]

# all_points = np.concatenate((p0[idx0], p1[idx1]), axis=0)
# pca = decomposition.PCA(n_components=3)
# pca.fit(all_points)
# pca = [pca, pca]

p0_pca = pca[0].transform(p_scope[0])
t0_pca = scipy.spatial.KDTree(p0_pca)
p1_pca = pca[1].transform(p_scope[1])
t1_pca = scipy.spatial.KDTree(p1_pca)

p_pca = [p0_pca, p1_pca]
t_pca = [t0_pca, t1_pca]

min_curvature = np.empty([2,3])
max_curvature = np.empty([2,3])
saddle = np.empty([2,3])

for objIndex in [0, 1]:
    print "Time: {}".format(timer()-start)
    print "Fit polynomial"

    if regression_method.lower() in TS_SPECIFIERS:
        model = Pipeline([('poly', PolynomialFeatures(degree=order)),
                          ('regr', TheilSenRegressor())])
    elif regression_method.lower() in RANSAC_SPECIFIERS:
        model = Pipeline([('poly', PolynomialFeatures(degree=order)),
                          ('regr', RANSACRegressor())])
    else:
        model = Pipeline([('poly', PolynomialFeatures(degree=order)),
                          ('regr', LinearRegression(fit_intercept=False))])

    # z as a response to x, y (coords in pca, z-3rd component)
    model = model.fit(p_pca[objIndex][:,:2], p_pca[objIndex][:, 2])

    # coefficients of the polynomial
    if regression_method.lower() in RANSAC_SPECIFIERS:
        C = model.named_steps['regr'].estimator_.coef_
    else:
        C = model.named_steps['regr'].coef_

    print "Coefficients: "
    print C
    #print "Sum of Residuals:"
    #print model.named_steps['regr'].residues_
    #print model.named_steps['regr'].residues_/len(p_pca[objIndex])


    print "Time: {}".format(timer()-start)
    print "Find a critical point"

    # axis = 0 is x and axis = 1 is y (powers of y rise within a row, powers of x rise within a coloumn
    C_as_matrix = np.array([[C[i+j*(j+1)/2] if j<=order else 0 for j in range(i, order+i+1)] for i in range(order+1)]).transpose()

    # gradient
    C_dx_matrix = np.polynomial.polynomial.polyder(C_as_matrix, axis = 0)
    C_dy_matrix = np.polynomial.polynomial.polyder(C_as_matrix, axis = 1)

    def grad(x):
        return [np.polynomial.polynomial.polyval2d(x[0], x[1], C_dx_matrix),
                np.polynomial.polynomial.polyval2d(x[0], x[1], C_dy_matrix)]

    # hessian = jacobian of the gradient
    C_dx_dx_matrix = np.polynomial.polynomial.polyder(C_dx_matrix, axis = 0)
    C_dx_dy_matrix = np.polynomial.polynomial.polyder(C_dx_matrix, axis = 1)
    C_dy_dx_matrix = np.polynomial.polynomial.polyder(C_dy_matrix, axis = 0)
    C_dy_dy_matrix = np.polynomial.polynomial.polyder(C_dy_matrix, axis = 1)

    def hess(x):
        return [[np.polynomial.polynomial.polyval2d(x[0], x[1], C_dx_dx_matrix), np.polynomial.polynomial.polyval2d(x[0], x[1], C_dx_dy_matrix)],
                [np.polynomial.polynomial.polyval2d(x[0], x[1], C_dy_dx_matrix), np.polynomial.polynomial.polyval2d(x[0], x[1], C_dy_dy_matrix)]]

    print ""

    stop = False
    iter = 0
    maxIter = 10
    guesses = np.concatenate(([[0.0,0.0]], np.random.normal(loc = [0,0], scale = 2.0, size = (maxIter-1, 2))))
    while (not stop) and (iter < maxIter):
        # find roots of the gradient
        sol = scipy.optimize.root(fun=grad, x0=guesses[iter], jac=hess)
        if sol.success:
            print "SUCCESS"
            print sol.message
            print "Check if saddle"
            if (np.linalg.det(hess(sol.x)) < 0):
                print "FOUND SADDLE"
                stop = True
            else:
                print "NOT A SADDLE"
        else:
            print "FAIL"
            print sol.message
        iter += 1

    saddle_pca = [sol.x[0], sol.x[1], np.polynomial.polynomial.polyval2d(sol.x[0], sol.x[1], C_as_matrix)]
    saddle[objIndex] = pca[objIndex].inverse_transform(saddle_pca)

    print "Time: {}".format(timer()-start)

    # calculate curvature in a scope around saddle

    # first find all points in a lightly higher radius than the given, right now its just all of the points
    # used for finding the saddle point
    # then calculate principal curvature for these points and then average for all points within a given radius
    scope_outer_pca = p_pca[objIndex]


    # interpolate through spline
    spline = scipy.interpolate.SmoothBivariateSpline(x=scope_outer_pca[:, 0],
                                            y=scope_outer_pca[:, 1],
                                            z=scope_outer_pca[:, 2],
                                            kx=interpolation_order,
                                            ky=interpolation_order)

    print "Time: {}".format(timer()-start)


    # find points in inner scope
    scope_inner_idx = t_pca[objIndex].query_ball_point(saddle_pca, radius)
    scope_inner_pca = p_pca[objIndex][scope_inner_idx]

    # set materials for inner scope
    if (set_materials):
        if not cmds.objExists(shading_grp_curvature):
            shading_grp_curvature = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shading_grp_curvature)
        else:
            if not cmds.objectType(shading_grp_curvature) == "shadingEngine":
                while (cmds.objExists(shading_grp_curvature)):
                    shading_grp_curvature = "{0}1".format(shading_grp_curvature)

                shading_grp_curvature = cmds.sets(renderable=True, noSurfaceShader=True, empty=True,
                                                name=shading_grp_curvature)

        faces_inner_idx  = cmds.polyListComponentConversion(
            ["{}.vtx[{}]".format(objs[objIndex], idx[objIndex][i]) for i in scope_inner_idx],
            fromVertex=1, toFace=1, internal=1)
        cmds.sets(faces_inner_idx, fe=shading_grp_curvature)

    # calculate the shape operator at these points
    shape_operator_scope, tangent_basis = shape_operator_spline(spline=spline, points=scope_inner_pca)

    # calculate the eigenvectors of the shape operator and average them
    shape_eigen_values, shape_eigen_vectors = np.linalg.eig(shape_operator_scope)
    shape_sorted_idx = shape_eigen_values.argsort()[:,::-1]
    shape_eigen_values = np.array([shape_eigen_values[i][shape_sorted_idx[i]] for i in range(len(shape_sorted_idx))])
    shape_eigen_vectors = np.array([shape_eigen_vectors[i][:,shape_sorted_idx[i]] for i in range(len(shape_sorted_idx))])

    # shape_eigen_vectors[:,:,0] are the max eigenvectors
    max_curvature_pca = np.average([shape_eigen_vectors[:,:,0][i][0] * tangent_basis[i][0] +
                                    shape_eigen_vectors[:,:,0][i][1] * tangent_basis[i][1]
                                    for i in range(len(shape_eigen_vectors))],
                                   axis=0)#,
                                   #weights=abs(shape_eigen_values[:,0]))
    min_curvature_pca = np.average([shape_eigen_vectors[:,:,1][i][0] * tangent_basis[i][0] +
                                    shape_eigen_vectors[:,:,1][i][1] * tangent_basis[i][1]
                                    for i in range(len(shape_eigen_vectors))],
                                   axis=0)#,
                                   #weights=abs(shape_eigen_values[:, 1]))

    if ((objIndex==0 and pca[objIndex].components_[2].dot(direction_up)<0) or
            (objIndex==1 and pca[objIndex].components_[2].dot(direction_up)>0)):
        max_curvature_pca, min_curvature_pca = min_curvature_pca, max_curvature_pca

    max_curvature[objIndex] = pca[objIndex].inverse_transform(saddle_pca+max_curvature_pca)-saddle[objIndex]
    max_curvature[objIndex] *= 1/np.linalg.norm(max_curvature[objIndex])
    min_curvature[objIndex] = pca[objIndex].inverse_transform(saddle_pca+min_curvature_pca)-saddle[objIndex]
    min_curvature[objIndex] *= 1/np.linalg.norm(min_curvature[objIndex])

    print "Time: {}".format(timer()-start)

floating_min = np.cross(min_curvature[0], min_curvature[1])
floating_min *= 1./np.linalg.norm(floating_min)
floating_max = np.cross(max_curvature[0], max_curvature[1])
floating_max *= 1./np.linalg.norm(floating_max)

auto = np.empty([2,3])
if "auto" in axis_used:
    if len(objs)>2:
        # Y points down for left, up for right
        # Z points away from the body for left, towards the body for right
        if abs(direction_in.dot(min_curvature[0]))>abs(direction_in.dot(max_curvature[0])):
            auto[0] = min_curvature[0]
        else:
            auto[0] = max_curvature[0]

        if ((left) and direction_in.dot(auto[0])>0) or ((not left) and  direction_in.dot(auto[0])<0):
            auto[0] *= -1

        # X points in the direction of sight for left, opposite direction for right
        sight_estimate = np.cross(auto[0], direction_up)
        if abs(sight_estimate.dot(min_curvature[1]))>abs(sight_estimate.dot(max_curvature[1])):
            auto[1] = min_curvature[1]
        else:
            auto[1] = max_curvature[1]

        if ((left) and sight_estimate.dot(auto[1]) < 0) or ((not left) and sight_estimate.dot(auto[1]) > 0):
            auto[1] *= -1

    else:
        auto[0] = min_curvature[0] # Z
        auto[1] = min_curvature[1] # X

    floating_auto = np.cross(auto[0], auto[1]) # Y
    floating_auto *= 1./np.linalg.norm(floating_auto)


# center, body fixed axis and reference axis
for objIndex in [0, 1]:
    objName = objs[objIndex].split(":")[0]
    objName_other = (objs[0] if objIndex==1 else objs[1]).split(":")[0]

    sphere = cmds.polySphere(name="saddle_{}_{}".format(objName, objName_other), radius = 0.1)
    cmds.move(saddle[objIndex][0], saddle[objIndex][1], saddle[objIndex][2], sphere, absolute = True)

    if(set_materials):
        cmds.sets(sphere[0], fe = shading_grp_saddle)

    shading_grp_used = shading_grp_axis2 if objIndex == 1 else shading_grp_axis1

    cylinder_len = 2*radius
    cylinder_si = 0.025*radius
    cone_si = 0.1*radius

    if "auto" in axis_used:
        ax_label = "X" if objIndex==1 else "Z"

        cylinder, _ = makeAxis(origin=saddle[objIndex], direction=auto[objIndex],
                               cylinder_name="cyl_{}_{}_{}".format(ax_label, objName, objName_other),
                               cone_name="cone_{}_{}_{}".format(ax_label, objName, objName_other),
                               shading_grp = shading_grp_used,
                               cylinder_length = cylinder_len,
                               cylinder_size = cylinder_si,
                               cone_size=cone_si)
        cmds.parent(cylinder[0], sphere[0])

        cylinder, _ = makeAxis(origin=saddle[objIndex], direction=floating_auto,
                               cylinder_name="cyl_{}_ref_{}_{}".format(ax_label, objName, objName_other),
                               cone_name="cone_{}_ref_{}_{}".format(ax_label, objName, objName_other),
                               shading_grp = shading_grp_axisRef,
                               cylinder_length = cylinder_len,
                               cylinder_size = cylinder_si,
                               cone_size=cone_si)
        cmds.parent(cylinder[0], sphere[0])

    if "min" in axis_used:
        cylinder, _ = makeAxis(origin=saddle[objIndex], direction=min_curvature[objIndex],
                                    cylinder_name="cyl_min_{}_{}".format(objName, objName_other),
                                    cone_name="cone_min_{}_{}".format(objName, objName_other),
                                    cylinder_length=cylinder_len,
                                    cylinder_size=cylinder_si,
                                    cone_size=cone_si,
                                    shading_grp = shading_grp_used)
        cmds.parent(cylinder[0], sphere[0])

    if "max" in axis_used:
        cylinder, _ = makeAxis(origin=saddle[objIndex], direction=max_curvature[objIndex],
                                    cylinder_name="cyl_max_{}_{}".format(objName, objName_other),
                                    cone_name="cone_max_{}_{}".format(objName, objName_other),
                                    cylinder_length=cylinder_len,
                                    cylinder_size=cylinder_si,
                                    cone_size=cone_si,
                                    shading_grp=shading_grp_used)
        cmds.parent(cylinder[0], sphere[0])

    if "floating_min" in axis_used:
        cylinder, _ = makeAxis(origin=saddle[objIndex], direction=floating_min,
                                    cylinder_name="cyl_floating_min_{}_{}".format(objName, objName_other),
                                    cone_name="cone_floating_min_{}_{}".format(objName, objName_other),
                                    cylinder_length = cylinder_len,
                                    cylinder_size = cylinder_si,
                                    cone_size = cone_si,
                                    shading_grp=shading_grp_used)

        cmds.parent(cylinder[0], sphere[0])

    if "floating_max" in axis_used:
        cylinder, _ = makeAxis(origin=saddle[objIndex], direction=floating_max,
                                    cylinder_name="cyl_floating_max_{}_{}".format(objName, objName_other),
                                    cone_name="cone_floating_max_{}_{}".format(objName, objName_other),
                                    cylinder_length = cylinder_len,
                                    cylinder_size = cylinder_si,
                                    cone_size = cone_si,
                                    shading_grp=shading_grp_used)
        cmds.parent(cylinder[0], sphere[0])



if save_flag:
    for objIndex in [0, 1]:
        save_file.write('"saddle[{}]","{}"\n'.format(objIndex, saddle[objIndex]))
        save_file.write('"max[{}]","{}"\n'.format(objIndex, max_curvature[objIndex]))
        save_file.write('"min[{}]","{}"\n'.format(objIndex, min_curvature[objIndex]))

    save_file.write('"axis_used","{}"\n'.format(axis_used))

    if "auto" in axis_used:
        save_file.write('"X","{}"\n'.format(auto[1]))
        save_file.write('"Y","{}"\n'.format(floating_auto))
        save_file.write('"Z","{}"\n'.format(auto[0]))
        x_axis = auto[1]
        y_axis = floating_auto
        z_axis = auto[0]


    elif axis_used[0]=="min":
        save_file.write('"X","{}"\n'.format(min_curvature[1]))
        save_file.write('"Y","{}"\n'.format(floating_min))
        save_file.write('"Z","{}"\n'.format(min_curvature[0]))
        x_axis = min_curvature[1]
        y_axis = floating_min
        z_axis = min_curvature[0]

    else:
        save_file.write('"X","{}"\n'.format(max_curvature[1]))
        save_file.write('"Y","{}"\n'.format(floating_max))
        save_file.write('"Z","{}"\n'.format(max_curvature[0]))
        x_axis = max_curvature[1]
        y_axis = floating_max
        z_axis = max_curvature[0]

    r_floating = math.acos(np.clip(np.dot(z_axis, x_axis), -1, 1)) * misc.RAD_TO_DEG
    x_ahead = np.array((misc.getRotationAroundAxis(angle=math.pi / 2, v=y_axis, rad=True) * np.matrix(
        x_axis).reshape(3, 1))).reshape(3)
    if np.dot(z_axis, x_ahead) > 0:
        r_floating *= -1
    save_file.write('"RF","{}"\n'.format(r_floating))

    save_file.close()