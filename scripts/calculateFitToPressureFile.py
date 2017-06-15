import numpy as np
import pyclipper
import csv
import sklearn.decomposition
import sklearn.cluster
import math
import copy
from pressureStatistics import segs_from_file, trans_matrix_from_pca



# script to calculate a fit of the imprint to the pressure_file

def calculateFitToPressure(imprint_file_left_path,  imprint_file_right_path, pressure_file_path,
                           path_to_write_transform_imprint,
                           path_to_write_transform_left_zones,
                           path_to_write_transform_right_zones,
                           path_to_write_pressure_data,
                           path_to_write_pressure_metadata):

    MEAN_METHOD = 2  # 0 - mean by mean of single pca's, 1 - mean by center of bounding boxes after pcas, 2 - average between both (wrt weights (mean, bounding boxes))
    MEAN_METHOD_WEIGHTS = [0.75, 0.25]
    CLUSTERING_METHOD = 1  # 0 - Kmeans, 1 - Spectral
    CLUSTERING_SCALE_X = 1  # 0 - nothing 1 - scales the x - axis after centering
    CLUSTERING_SCALE_X_FACTOR = 0.1

    # --------------------------------------------------------------------------------------------------------
    # -------------------------------- PRESSURE FILE ---------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------

    keywords = ["ASCII_DATA @@", "ROW_SPACING", "COL_SPACING"]
    with open(pressure_file_path, 'r') as file:
        line = file.readline()
        while line != '':
            while (sum([1 if keyword in line else 0 for keyword in keywords]) < 1) and line!='':
                line =file.readline()
            if line!='':
                # keyword hit here
                keyword = keywords[[1 if keyword in line else 0 for keyword in keywords].index(1)]
                if keyword == "ROW_SPACING":
                    lsplit = line.split(" ")
                    row_spacing = float(lsplit[1])
                    if len(lsplit)>2:
                        if lsplit[2] == "millimeters":
                            row_spacing = row_spacing*0.1

                elif keyword == "COL_SPACING":
                    lsplit = line.split(" ")
                    col_spacing = float(lsplit[1])
                    if len(lsplit)>2:
                        if lsplit[2] == "millimeters":
                            col_spacing = col_spacing * 0.1

                elif keyword == "ASCII_DATA @@":
                    # next block is ascii data
                    pressure_data = []
                    line = file.readline()
                    rowNo = 0
                    while (line!='') and ("@@" not in line):
                        row = [float(v) for v in line.split(",")[:-1]]
                        for colNo, v in enumerate(row):
                            pressure_data.append([rowNo, colNo, v])
                        line=file.readline()
                        rowNo +=1
                    #end of data reached

                # go on to the next bit of the file
                line=file.readline()

    # file parsed
    # now calculate cords from row and col number
    pressure_data = np.array(pressure_data)
    pressure_data = np.c_[pressure_data, (max(pressure_data[:,0])-pressure_data[:,0])*row_spacing+row_spacing/2, pressure_data[:,1]*col_spacing+col_spacing/2]
    pressure_data[:, [-2,-1]] = pressure_data[:, [-1,-2]]

    with open(path_to_write_pressure_data, 'w') as file:
        file.write("x,y,pressure,rowNo,colNo\n")
        for row in pressure_data:
            file.write('"{}","{}","{}","{}","{}"\n'.format(row[-2], row[-1], row[2], row[0], row[1]))

    with open(path_to_write_pressure_metadata, 'w') as file:
        file.write("variable,value\n")
        file.write('"ROW_SPACING",{}\n'.format(row_spacing))
        file.write('"COL_SPACING",{}'.format(col_spacing))


    pts_pressure = np.array([pt for pt in pressure_data if pt[2] != 0])

    # ----------- CLUSTERING ----------------
    pts_for_clustering = copy.deepcopy(pts_pressure[:,-2:])
    if CLUSTERING_SCALE_X == 1:
        pts_for_clustering[:,0] -=  np.mean(pts_for_clustering[:,0], axis=0)
        pts_for_clustering[:, 0] *= CLUSTERING_SCALE_X_FACTOR

    if CLUSTERING_METHOD == 0:
        clustering_pressure = sklearn.cluster.SpectralClustering(n_clusters=2).fit(pts_for_clustering)
    else:
        clustering_pressure = sklearn.cluster.KMeans(n_clusters=2).fit(pts_for_clustering)

    pts_pressure_left = pts_pressure[[i for i, label in enumerate(clustering_pressure.labels_) if label<1]]
    pts_pressure_right = pts_pressure[[i for i, label in enumerate(clustering_pressure.labels_) if label>0]]

    mean_left_pressure = np.mean(pts_pressure_left[:,-2:], axis=0)
    mean_right_pressure = np.mean(pts_pressure_right[:,-2:], axis=0)

    if(mean_left_pressure[1]>mean_right_pressure[1]):
        pts_pressure_left, pts_pressure_right = pts_pressure_right, pts_pressure_left
        mean_left_pressure, mean_right_pressure = mean_right_pressure, mean_left_pressure

    # ----------- PCA ------------------------
    pca_pressure_single = [sklearn.decomposition.PCA(n_components=2).fit(pts) for pts in [pts_pressure_left[:,-2:], pts_pressure_right[:,-2:]]]
    for pca_pressure_s in pca_pressure_single:
        # pca axes flipped so that first axis is in same direction as x-axis and second axis is in same direction as y-axis
        if abs(pca_pressure_s.components_[0][1])>abs(pca_pressure_s.components_[1][1]):
            pca_pressure_s.components_ = pca_pressure_s.components_[::-1]
        if pca_pressure_s.components_[0][0]<0:
            pca_pressure_s.components_[0] *= -1
        if pca_pressure_s.components_[1][1]<0:
            pca_pressure_s.components_[1] *= -1



    pca_pressure = sklearn.decomposition.PCA(n_components=2).fit([[1,0],[0,1]]) # supply values with no meaning -> no meaningful explained variance stats but cuts runtime
    pca_pressure.components_ = np.array([np.mean([pca_pressure_single[1].components_[0], pca_pressure_single[0].components_[0]], axis=0),
                                np.mean([pca_pressure_single[1].components_[1], pca_pressure_single[0].components_[1]], axis=0)
                                ])


    if MEAN_METHOD == 0:
        # estimate mean by mean
        pca_pressure.mean_ = np.mean([mean_left_pressure, mean_right_pressure], axis=0)
    elif MEAN_METHOD==1:
        # estimate mean by middle point of bounding boxes
        pca_pressure.mean_ = estimate_mean_by_bounding_pca([pts_pressure_left[:,-2:], pts_pressure_right[:,-2:]], pca_pressure)
    elif MEAN_METHOD==2:
        pca_pressure.mean_ = np.average([np.mean([mean_left_pressure, mean_right_pressure], axis=0),
                                      estimate_mean_by_bounding_pca([pts_pressure_left[:, -2:], pts_pressure_right[:, -2:]],
                                                                    pca_pressure)],
                                     axis=0, weights=MEAN_METHOD_WEIGHTS)



    # -------------------------------------------------------------------------------------------------------------
    # ----------------------------------------------- SEGMENTS ----------------------------------------------------
    # -------------------------------------------------------------------------------------------------------------

    segs_left = segs_from_file(imprint_file_left_path)
    segs_right = segs_from_file(imprint_file_right_path)

    pts_segs_left = np.vstack(segs_left)
    left_mean = np.mean(pts_segs_left, axis=0)
    pts_segs_right = np.vstack(segs_right)
    right_mean = np.mean(pts_segs_right, axis=0)
    pts_all_segs = np.r_[pts_segs_left, pts_segs_right]

    pca = sklearn.decomposition.PCA(n_components=2)


    pca.fit(pts_all_segs)

    direction_right_left = left_mean-right_mean
    direction_right_left /= np.linalg.norm(direction_right_left)
    direction_left_right = -direction_right_left                                            # corresponds to x axis in pressure
    direction_back_front = np.array([-direction_right_left[1], direction_right_left[0]])    # 90 degrees from right->left
    direction_front_back = -direction_back_front                                            # corresponds to y axis in pressure


    pca_single = [sklearn.decomposition.PCA(n_components=2).fit(pts) for pts in [pts_segs_left, pts_segs_right]]
    for pca_s in pca_single:
        # check if axis of components have the right orientation else make them fit
        if abs(np.dot(direction_front_back, pca_s.components_[0])) < math.cos(math.pi / 4):
            pca_s.components_ = pca_s.components_[::-1]
        if (np.dot(direction_front_back, pca_s.components_[0]) < 0):
            pca_s.components_[0] *= -1
        if (np.dot(direction_left_right, pca_s.components_[1]) < 0):
            pca_s.components_[1] *= -1

    pca.components_ = np.array([np.mean([pca_single[1].components_[0], pca_single[0].components_[0]], axis=0),
                                np.mean([pca_single[1].components_[1], pca_single[0].components_[1]], axis=0)
                                ])

    if MEAN_METHOD == 0:
        # estimate mean by average
        pca.mean_ =  np.mean([left_mean, right_mean], axis=0)
    elif MEAN_METHOD == 1:
        # estimate mean by middle point of bounding boxes
        pca.mean_ = estimate_mean_by_bounding_pca(np.array([[pt for seg in segs for pt in seg] for segs in [segs_left, segs_right]]), pca)
    elif MEAN_METHOD==2:
        pca.mean_ = np.average([np.mean([left_mean, right_mean], axis=0),
                             estimate_mean_by_bounding_pca(
                                 np.array([[pt for seg in segs for pt in seg] for segs in [segs_left, segs_right]]), pca)
                             ],axis=0, weights=MEAN_METHOD_WEIGHTS)

    transformation = np.linalg.inv(trans_matrix_from_pca(pca_pressure))*trans_matrix_from_pca(pca)
    np.savetxt(path_to_write_transform_imprint, transformation, delimiter=',', fmt='%.25e')

    for idx_side in range(len(pca_single)):
        pts_transformed_s = pca_single[idx_side].transform(np.vstack([segs_left, segs_right][idx_side]))
        min_x = min(pts_transformed_s[:, 0])
        max_x = max(pts_transformed_s[:, 0])
        min_y = min(pts_transformed_s[:, 1])
        max_y = max(pts_transformed_s[:, 1])
        scale_matrix = np.matrix([[max_x-min_x,0,0],[0,max_y-min_y,0],[0,0,1]])
        reposition_matrix = np.matrix([1,0,0], [0,1,0],[min_x,min_y])
        # pca_pressure-1*pca**single_pca-1*reposition*scale
        transformation_s =  transformation*np.linalg.inv(trans_matrix_from_pca(pca_single[idx_side]))*reposition_matrix*scale_matrix
        output_path = path_to_write_transform_left_zones if idx_side==0 else path_to_write_transform_right_zones
        np.savetxt(output_path, transformation_s, delimiter=',', fmt='%.25e')

