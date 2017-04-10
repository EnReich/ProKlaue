import pk_src
from pk_src import misc
import maya.cmds as cmds
import numpy as np
import scipy
import scipy.optimize
import scipy.spatial
import scipy.linalg
import scipy.interpolate
import sklearn
from sklearn import decomposition
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
import sympy
from timeit import default_timer as timer

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


threshold = 0.4
order = 5
radius = 1
radius_outer = 1.2*radius
interpolation_order = 3
interpolation_stepsize = 0.05

print "Time: {}".format(timer()-start)
print "Finding close point pairs"

# get point to the 2 bones who are selected, build kd trees
objs = cmds.ls(sl=1)
p0 = np.array(misc.getPointsAsList(objs[0], worldSpace=True))
p1 = np.array(misc.getPointsAsList(objs[1], worldSpace=True))
p = [p0, p1]
t0 = scipy.spatial.KDTree(p[0])
t1 = scipy.spatial.KDTree(p[1])
t = [t0, t1]

# find pairs of points who are not further away than a given threshold
rIntersection = t[0].query_ball_tree(other=t[1], r=threshold)
idx0 = [i for i in range(len(p0)) if rIntersection[i]]
idx1 = np.array(list(set(np.uint32(np.concatenate(rIntersection)))))
idx = [idx0, idx1]

# # visual indication by selection
# cmds.select(clear=True)
# for i in idx0: cmds.select("{}.vtx[{}]".format(objs[0], i), add=True)
# for i in idx1: cmds.select("{}.vtx[{}]".format(objs[1], i), add=True)

print "Time: {}".format(timer()-start)

print "Transform coordinates"

# pca on the data points to transform the coordinates


p0_scope = p[0][idx[0]]
p1_scope = p[1][idx[1]]
p_scope = [p0_scope, p1_scope]

pca0 = decomposition.PCA(n_components=3)
pca0.fit(p0_scope)
pca1 = decomposition.PCA(n_components=3)
pca1.fit(p1_scope)

pca = [pca0, pca1]

# all_points = np.concatenate((p0[idx0], p1[idx1]), axis=0)
# pca = decomposition.PCA(n_components=3)
# pca.fit(all_points)
# pca = [pca, pca]

p0_pca = pca[0].transform(p0_scope)
t0_pca = scipy.spatial.KDTree(p0_pca)
p1_pca = pca[1].transform(p1_scope)
t1_pca = scipy.spatial.KDTree(p1_pca)

p_pca = [p0_pca, p1_pca]
t_pca = [t0_pca, t1_pca]

for objIndex in [0, 1]:
    print "Time: {}".format(timer()-start)
    print "Fit polynomial"

    model = Pipeline([('poly', PolynomialFeatures(degree=order)),
                      ('linear', LinearRegression(fit_intercept=False))])

    # z as a response to x, y (coords in pca, z-3rd component)
    model = model.fit(p_pca[objIndex][:,:2], p_pca[objIndex][:, 2])

    # coefficients of the polynomial
    C = model.named_steps['linear'].coef_

    print "Coefficients: "
    print C
    print "Sum of Residuals:"
    print model.named_steps['linear'].residues_
    print model.named_steps['linear'].residues_/len(p_pca[objIndex])


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
    saddle = pca[objIndex].inverse_transform(saddle_pca)

    print "Time: {}".format(timer()-start)

    # calculate curvature in a scope around saddle

    # first find all points in a lightly higher radius than the given
    # then calculate principal curvature for these points and then average for all points within a given radius
    # scope_outer_idx = idx0
    # scope_outer = p0_scope
    scope_outer_pca = p_pca[objIndex]

    # scope_outer_idx = t0.query_ball_point(saddle, radius_outer)
    # scope_outer = p0[scope_outer_idx]
    # scope_outer_pca = pca.transform(scope_outer)

    # interpolate through spline
    spline = scipy.interpolate.SmoothBivariateSpline(x=scope_outer_pca[:, 0],
                                            y=scope_outer_pca[:, 1],
                                            z=scope_outer_pca[:, 2],
                                            kx=interpolation_order,
                                            ky=interpolation_order)

    print "Time: {}".format(timer()-start)

    # coords for evaluation
    # coords_x_range = np.arange(start=saddle_pca[0]-radius, stop=saddle_pca[0]+radius+interpolation_stepsize, step=interpolation_stepsize)
    # coords_y_range = np.arange(start=saddle_pca[0]-radius, stop=saddle_pca[0]+radius+interpolation_stepsize, step=interpolation_stepsize)
    # coords = [[x,y] for x in coords_x_range for y in coords_y_range if scipy.spatial.distance.euclidean([x,y], [saddle_pca[0], saddle_pca[1]])<radius]

    # find points in inner scope
    scope_inner_idx = t_pca[objIndex].query_ball_point(saddle_pca, radius)
    scope_inner_pca = p_pca[objIndex][scope_inner_idx]

    # scope_inner_idx = t0.query_ball_point(saddle, radius)
    # scope_inner = p0[scope_inner_idx]
    # scope_inner_pca = pca.transform(scope_inner)

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

    max_curvature = pca[objIndex].inverse_transform(saddle_pca+max_curvature_pca)-saddle
    max_curvature *= 1/np.linalg.norm(max_curvature)
    min_curvature = pca[objIndex].inverse_transform(saddle_pca+min_curvature_pca)-saddle
    min_curvature *= 1/np.linalg.norm(min_curvature)


    print "Time: {}".format(timer()-start)

    objName = objs[objIndex]
    objName_other = objs[0] if objIndex==1 else objs[1]

    sphere = cmds.polySphere(name="saddle_{}_{}".format(objName, objName_other), radius = 0.1)
    cmds.move(saddle[0], saddle[1], saddle[2], sphere, absolute = True)

    # cylinderMax = cmds.polyCylinder()
    # cmds.scale(0.01,10,0.01, cylinderMax)
    # r = misc.getRotationFromAToB(a=np.matrix([0,1,0]).reshape(3,1), b=np.matrix(max_curvature).reshape(3,1))
    # m = np.matrix(cmds.xform(cylinderMax, q=1, ws=1, m=1)).reshape(4,4).transpose()
    # m_new = np.matrix(np.r_[np.c_[r, [0,0,0]],[[0,0,0,1]]])*m
    # cmds.xform(cylinderMax, m=m_new.transpose().A1, ws=1)
    # cmds.move(saddle[0], saddle[1], saddle[2], cylinderMax, absolute = True)

    cylinderMin = cmds.polyCylinder(name = "cyl_min_{}_{}".format(objName, objName_other))
    cmds.scale(0.01,10,0.01, cylinderMin)
    r = misc.getRotationFromAToB(a=np.matrix([0,1,0]).reshape(3,1), b=np.matrix(min_curvature).reshape(3,1))
    m = np.matrix(cmds.xform(cylinderMin, q=1, ws=1, m=1)).reshape(4,4).transpose()
    m_new = np.matrix(np.r_[np.c_[r, [0,0,0]],[[0,0,0,1]]])*m
    cmds.xform(cylinderMin, m=m_new.transpose().A1, ws=1)
    cmds.move(saddle[0], saddle[1], saddle[2], cylinderMin, absolute = True)

    minXMax = np.cross(min_curvature, max_curvature)
    cylinderMinxMax = cmds.polyCylinder(name = "cyl_min-x-max_{}_{}".format(objName, objName_other))
    cmds.scale(0.01,10,0.01, cylinderMinxMax)
    r = misc.getRotationFromAToB(a=np.matrix([0,1,0]).reshape(3,1), b=np.matrix(minXMax).reshape(3,1))
    m = np.matrix(cmds.xform(cylinderMinxMax, q=1, ws=1, m=1)).reshape(4,4).transpose()
    m_new = np.matrix(np.r_[np.c_[r, [0,0,0]],[[0,0,0,1]]])*m
    cmds.xform(cylinderMinxMax, m=m_new.transpose().A1, ws=1)
    cmds.move(saddle[0], saddle[1], saddle[2], cylinderMinxMax, absolute = True)

    last = np.cross(minXMax, min_curvature)
    cylinderLast = cmds.polyCylinder(name = "cyl_last_{}_{}".format(objName, objName_other))
    cmds.scale(0.01,10,0.01, cylinderLast)
    r = misc.getRotationFromAToB(a=np.matrix([0,1,0]).reshape(3,1), b=np.matrix(last).reshape(3,1))
    m = np.matrix(cmds.xform(cylinderLast, q=1, ws=1, m=1)).reshape(4,4).transpose()
    m_new = np.matrix(np.r_[np.c_[r, [0,0,0]],[[0,0,0,1]]])*m
    cmds.xform(cylinderLast, m=m_new.transpose().A1, ws=1)
    cmds.move(saddle[0], saddle[1], saddle[2], cylinderLast, absolute = True)

# sphereMax = cmds.polySphere(radius = 0.1)
# cmds.move((saddle+max_curvature)[0], (saddle+max_curvature)[1], (saddle+max_curvature)[2], sphereMax, absolute = True)
#
# sphereMin = cmds.polySphere(radius = 0.1)
# cmds.move((saddle+min_curvature)[0], (saddle+min_curvature)[1], (saddle+min_curvature)[2], sphereMin, absolute = True)




# OLD CODE
#
# terms = ["x**{}*y**{}".format(i-j,j) for i in range(order+1) for j in range(i+1)]
# terms_and_coeffs = ['{}*{}'.format(str(c), t) for c,t in zip(C, terms)]
# expr = sympy.sympify('+'.join(terms_and_coeffs))
# diff_x = sympy.diff(expr, x)
# diff_y = sympy.diff(expr, y)
# solution = sympy.solve([diff_x, diff_y], [x,y])
#
# # evaluate z
# z = expr.subs(x, solution[0][0]).subs(y, solution[0][1])
# pca.inverse_transform([solution[0][0], solution[0][1], z])
#
# #check if real
# if(sympy.sympify(solution[0][0]).is_real): print 'real'
#
# data = p0[idx0]
# order = 3
# X_data, Y_data, Z_data = data[:, 0], data[:, 1], data[:, 2]
# # since 1.12. A = np.stack([(X_data**i)*(Y_data**j) for i in range(order+1) for j in range(order+1-i)], axis=1)
# A = np.concatenate([((X_data**i)*(Y_data**j)).reshape(-1,1) for i in range(order+1) for j in range(order+1-i)], axis = 1)
# C, _, _, _ = scipy.linalg.lstsq(A, Z_data)
# terms = ["x**{}*y**{}".format(i,j) for i in range(order+1) for j in range(order+1-i)]
#
#
#
# terms_and_coeffs = ['{}*{}'.format(str(c), t) for c,t in zip(C, terms)]
#
# x, y = sp.Symbol("x"), sp.Symbol("y")
# expr = sp.sympify('+'.join(terms_and_coeffs))
# diff_x = sp.diff(expr, x)
# diff_y = sp.diff(expr, y)
# solution = sp.solve([diff_x, diff_y], [x,y])
#
#
# # some 3-dim points
# mean = np.array([0.0, 0.0, 0.0])
# cov = np.array([[1.0, -0.5, 0.8], [-0.5, 1.1, 0.0], [0.8, 0.0, 1.0]])
# data = np.random.multivariate_normal(mean, cov, 50)
#
#
# # regular grid covering the domain of the data
# X, Y = np.meshgrid(np.arange(-3.0, 3.0, 0.5), np.arange(-3.0, 3.0, 0.5))
# XX = X.flatten()
# YY = Y.flatten()
#
# order = 2  # 1: linear, 2: quadratic
# if order == 1:
#     # best-fit linear plane
#     A = np.c_[data[:, 0], data[:, 1], np.ones(data.shape[0])]
#     C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])  # coefficients
#
#     # evaluate it on grid
#     Z = C[0] * X + C[1] * Y + C[2]
#
#     # or expressed using matrix/vector product
#     # Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)
#
# elif order == 2:
#     # best-fit quadratic curve
#     A = np.c_[np.ones(data.shape[0]), data[:, :2], np.prod(data[:, :2], axis=1), data[:, :2] ** 2]
#     C, _, _, _ = scipy.linalg.lstsq(A, data[:, 2])
#
#     # evaluate it on a grid
#     Z2 = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX * YY, XX ** 2, YY ** 2], C).reshape(X.shape)
#
#
# X_data, Y_data, Z_data = data[:, 0], data[:, 1], data[:, 2]
# A = np.stack([(X_data**i)*(Y_data**j) for i in range(order+1) for j in range(order+1-i)], axis=1)
# grid_points = np.stack([(XX**i)*(YY**j) for i in range(order+1) for j in range(order+1-i)], axis=1)
# C, _, _, _ = scipy.linalg.lstsq(A, Z_data)
# # evaluate it on a grid
# Z = np.dot(grid_points, C).reshape(X.shape)
#
# terms = ["x**{}*y**{}".format(i,j) for i in range(order+1) for j in range(order+1-i)]
# terms_and_coeffs = ['{}*{}'.format(str(c), t) for c,t in zip(C, terms)]
# expr = sp.sympify(terms_and_coeffs)
#
# # plot points and fitted surface
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2)
# ax.scatter(data[:, 0], data[:, 1], data[:, 2], c='r', s=50)
# plt.xlabel('X')
# plt.ylabel('Y')
# ax.set_zlabel('Z')
# ax.axis('equal')
# ax.axis('tight')
# plt.show()