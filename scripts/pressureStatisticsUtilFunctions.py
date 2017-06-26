import numpy as np
import pyclipper
import csv
import math
import sklearn.decomposition
import sklearn.cluster
import copy



def rotation_matrix(alpha, radian=True):
    if not radian:
        alpha = (alpha%360)*math.pi/180

    return np.array(math.cos(alpha), -math.sin(alpha), math.sin(alpha), math.cos(alpha)).reshape(2,2)

# signed angle between -180 and 180
def angle(x,y):
    return (180/math.pi)*math.atan2(np.cross(x,y), np.dot(x,y))

def get_th_point(p1, p2, th):
    # p1 under th, p2 over th
    under = np.array(p1)
    over = np.array(p2)
    if (under[2]>th):
        under, over = over, under

    diff = over - under
    to_th = (th - under[2])/diff[2]
    return under+to_th*diff

def segs_from_file(file_path):
    with open(file_path, 'rb') as def_f:
        def_reader = csv.DictReader(def_f)
        sid = 0
        seg = []
        segs = []
        for row in def_reader:
            if sid != int(row["SID"]):
                segs.append(seg)
                sid = int(row["SID"])
                seg = []
            seg.append([float(row["x"]), float(row["y"])])
        segs.append(seg)
    segs = np.array([np.array(seg) for seg in segs])
    return segs

def write_segs_to_file(segs, file_path):
    with open(file_path, 'w') as o_file_seg:
        o_file_seg.write("SID,PID,x,y\n")
        for SID, seg in enumerate(segs):
            for PID, p in enumerate(seg):
                o_file_seg.write('"{}","{}","{}","{}"\n'.format(SID, PID, p[0], p[1]))

def seg_cutoff(seg, th):
    pts_idx_under_th = np.array([i for i, pt in enumerate(seg) if pt[2]<=th])
    case = len(pts_idx_under_th)
    if case == len(seg):
        return seg
    elif case < 1:
        return
    elif len(seg)==3:
        if case == 2:
            idx_over_th = [i for i in range(len(seg)) if i not in pts_idx_under_th][0]
            replacement = [get_th_point(seg[idx_under], seg[idx_over_th], th) for idx_under in (np.array([idx_over_th-1, idx_over_th+1])%3)]
            replaced_seg = np.array([seg[(idx_over_th-1)%3], replacement[0], replacement[1], seg[(idx_over_th+1)%3]])
            return replaced_seg
        else:
            #case == 1
            idx_under_th = pts_idx_under_th[0]
            replacement_pts = [get_th_point(seg[idx_under_th], seg[idx_over], th) for idx_over in (np.array([idx_under_th-1, idx_under_th+1])%3)]
            replaced_seg = np.array([replacement_pts[0], seg[idx_under_th], replacement_pts[1]])
            return replaced_seg

    else:
        return

# cutoffs of segments for a given threshold in z (2nd col)
def segs_cutoff(segs, th):
    segs_under_th = [seg for seg in segs if sum([1 if i[2]<th else 0 for i in seg])>0]
    return np.array([seg_cutoff(seg, th) for seg in segs_under_th ])

# estimate mean by bounding box after pca
def estimate_mean_by_bounding_pca(pts_array, pca):
    pts_transformed_to_estimate_mean = np.array([pca.transform(pts) for pts in pts_array])
    min_x = [min(pts[:,0]) for pts in pts_transformed_to_estimate_mean]
    max_x = [max(pts[:,0]) for pts in pts_transformed_to_estimate_mean]
    min_y = [min(pts[:,1]) for pts in pts_transformed_to_estimate_mean]
    max_y = [max(pts[:,1]) for pts in pts_transformed_to_estimate_mean]
    estimated_mean = pca.inverse_transform([np.mean([np.mean(min_x), np.mean(max_x)]), np.mean([np.mean(min_y), np.mean(max_y)])])
    return estimated_mean

def trans_matrix_from_pca(pca):
    return np.matrix(np.vstack([np.hstack([pca.components_.transpose(), np.array([0,0]).reshape(-1,1)]), np.array([0,0,1])])) * \
           np.matrix(((np.vstack([np.hstack([np.array([1,0,0,1]).reshape(2,2), -pca.mean_.transpose().reshape(-1,1)]), np.array([0,0,1])]))))

def transformSegments(segs, matrix):
    return([[np.array((matrix*np.vstack([np.matrix(pt).reshape(2,1),[[1]]]))[:2,:]).reshape(2) for pt in seg] for seg in segs])



# function to calculate a fit of the imprint to the pressure_file
def calculateFitToPressure(imprint_file_left_path,  imprint_file_right_path, pressure_file_path,
                           path_to_write_transform_imprint,
                           path_to_write_transform_left_zones,
                           path_to_write_transform_right_zones,
                           path_to_write_pressure_data,
                           path_to_write_pressure_metadata,
                           mean_method = 2,                             #0 - mean by mean of single pca's, 1 - mean by center of bounding boxes after pcas, 2 - average between both (wrt weights (mean, bounding boxes))
                           mean_method_weights = (0.75, 0.25),
                           clustering_method = 1,                       # 0 - Kmeans, 1 - Spectral
                           clustering_scale_x = 1,                        # 0 - nothing 1 - scales the x - axis after centering
                           clustering_scale_x_factor = 0.1):


    MEAN_METHOD = mean_method
    MEAN_METHOD_WEIGHTS = mean_method_weights
    CLUSTERING_METHOD = clustering_method
    CLUSTERING_SCALE_X = clustering_scale_x
    CLUSTERING_SCALE_X_FACTOR = clustering_scale_x_factor

    # --------------------------------------------------------------------------------------------------------
    # -------------------------------- PRESSURE FILE ---------------------------------------------------------
    # --------------------------------------------------------------------------------------------------------

    keywords = ["ASCII_DATA @@", "ROW_SPACING", "COL_SPACING", "UNITS"]
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

                elif keyword == "UNITS":
                    lsplit = line.split(" ")
                    units = lsplit[1]

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
        file.write('"COL_SPACING",{}\n'.format(col_spacing))
        file.write('"UNITS",{}\n'.format(units))


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
        transform_single_pca = trans_matrix_from_pca(pca_single[idx_side])
        pts_transformed_s = pca_single[idx_side].transform(np.vstack([segs_left, segs_right][idx_side]))
        min_x = min(pts_transformed_s[:, 0])
        max_x = max(pts_transformed_s[:, 0])
        min_y = min(pts_transformed_s[:, 1])
        max_y = max(pts_transformed_s[:, 1])
        scale_matrix = np.matrix([[max_x-min_x,0,0],[0,max_y-min_y,0],[0,0,1]])
        reposition_matrix = np.matrix([[1,0,min_x], [0,1,min_y],[0,0,1]])
        # pca_pressure-1*pca**single_pca-1*reposition*scale
        transformation_s =  transformation*np.linalg.inv(transform_single_pca)*reposition_matrix*scale_matrix
        output_path = path_to_write_transform_left_zones if idx_side==0 else path_to_write_transform_right_zones
        np.savetxt(output_path, transformation_s, delimiter=',', fmt='%.25e')

    return([path_to_write_transform_imprint, path_to_write_transform_left_zones, path_to_write_transform_right_zones
            ])

# function to calculate the imprint of a list of front facing triangles of an object
def calculateImprint(tsf_file_left, tsf_file_right, path_to_write_left_segs, path_to_write_right_segs, th=0.5):
    paths_write = [path_to_write_left_segs,path_to_write_right_segs]

    for idx_path, path in enumerate([tsf_file_left, tsf_file_right]):
        file = open(path, 'r')
        reader = csv.DictReader(file, delimiter=',', quotechar='"')
        segs =[]
        currentSeg = []
        old_SID = 0
        for row in reader:
            SID = int(row['SID'])
            PID = int(row['PID'])
            cords = [float(row['x']), float(row['z']), float(row['y'])]
            if SID != old_SID:
                segs.append(currentSeg)
                currentSeg= []

            currentSeg.append(cords)
            old_SID = SID
        segs.append(currentSeg)
        file.close()

        segs = np.array(segs)
        segs = segs_cutoff(segs, th)

        segs = np.array([np.array([pts[:2] for pts in seg]) for seg in segs])

        signed_area = [sum((seg[np.r_[1:len(seg), 0],0]-seg[:,0])*(seg[np.r_[1:len(seg), 0],1]+seg[:,1]))/2. for seg in segs]
        to_reverse = [i for i,a in enumerate(signed_area) if a>0]
        for i in to_reverse:
            segs[i] = segs[i][::-1]


        union = pyclipper.scale_to_clipper(segs)
        clipper = pyclipper.Pyclipper()
        clipper.AddPaths(union, pyclipper.PT_CLIP, True)
        union = clipper.Execute(pyclipper.CT_UNION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
        union = pyclipper.scale_from_clipper(union)


        union2 = np.array(union)
        for i, u in enumerate(union2): union2[i]= np.array(union2[i])
        signed_area2 = [sum((seg[np.r_[1:len(seg), 0],0]-seg[:,0])*(seg[np.r_[1:len(seg), 0],1]+seg[:,1]))/2. for seg in union2]
        union_without_holes = union2[[i for i, a in enumerate(signed_area2) if a<0]]

        if idx_path == 0:
            segs_left = union_without_holes
        else:
            segs_right = union_without_holes

    for idx_side, segs in enumerate([segs_left, segs_right]):
        output_path = path_to_write_left_segs if idx_side==0 else path_to_write_right_segs
        write_segs_to_file(segs, output_path)


    return([path_to_write_left_segs, path_to_write_right_segs])



def calculateStatistics(path_to_imprint_file_left,
                        path_to_imprint_file_right,
                        path_to_zones_left,
                        path_to_zones_right,
                        path_to_transform_imprint,
                        path_to_transform_left_zones,
                        path_to_transform_right_zones,
                        path_to_pressure_data,
                        path_to_pressure_metadata,
                        path_to_write_statistics):

    transform_imprint = np.matrix(np.loadtxt(path_to_transform_imprint, delimiter=","))
    transform_left_zones = np.loadtxt(path_to_transform_left_zones, delimiter=",")
    transform_right_zones = np.loadtxt(path_to_transform_right_zones, delimiter=",")

    segs_left_orig = segs_from_file(path_to_imprint_file_left)
    segs_right_orig = segs_from_file(path_to_imprint_file_right)
    segs_zones_left_orig = segs_from_file(path_to_zones_left)
    segs_zones_right_orig = segs_from_file(path_to_zones_right)

    segs_left = transformSegments(segs_left_orig, transform_imprint)
    segs_right = transformSegments(segs_right_orig, transform_imprint)
    segs_zones_left = transformSegments(segs_zones_left_orig, transform_left_zones)
    segs_zones_right = transformSegments(segs_zones_right_orig, transform_right_zones)

    col_spacing = 0.5
    row_spacing = 0.5

    with open(path_to_pressure_metadata, "rb") as metadata_file:
        reader = csv.DictReader(metadata_file)
        for row in reader:
            variable = row["variable"]
            value = row["value"]
            if str(variable).lower()=="COL_SPACING".lower():
                col_spacing = float(value)
            elif str(variable).lower()=="ROW_SPACING".lower():
                row_spacing = float(value)
            elif str(variable).lower() == "UNITS".lower():
                units = value

    with open(path_to_pressure_data, "rb") as pressure_file:
        reader = csv.reader(pressure_file)
        header = reader.next()
        pressure_data = []
        for row in reader:
            pressure_data.append([float(val) for val in row])

    pressure_data = np.array(pressure_data)
    pressure_data_x_col = next(i for i, var in enumerate(header) if var=="x")
    pressure_data_y_col = next(i for i, var in enumerate(header) if var == "y")
    pressure_data_pressure_col = next(i for i, var in enumerate(header) if var == "pressure")

    pressure_data_not_null = pressure_data[pressure_data[:,pressure_data_pressure_col]!=0]

    pressure_meas = []
    area = []
    area_intersect = []
    sensor_segs = []
    sensor_clipped_against_imprint_scaled_for_clipper = []

    pc = pyclipper.Pyclipper()

    for row in pressure_data_not_null:
        x = row[pressure_data_x_col]
        y = row[pressure_data_y_col]
        subj = ((x-row_spacing/2., y-col_spacing/2.),
                (x+row_spacing/2., y-col_spacing/2.),
                (x+row_spacing/2., y+col_spacing/2.),
                (x-row_spacing/2., y+col_spacing/2.))

        sensor_segs.append(subj)

        ar = col_spacing*row_spacing
        area.append(ar)
        pressure_meas.append(row[pressure_data_pressure_col]*ar)

        # clip against segments
        # left
        pc.Clear()
        pc.AddPaths(pyclipper.scale_to_clipper(segs_left), pyclipper.PT_CLIP, True)
        pc.AddPaths(pyclipper.scale_to_clipper(segs_right), pyclipper.PT_CLIP, True)
        pc.AddPath(pyclipper.scale_to_clipper(subj), pyclipper.PT_SUBJECT, True)
        sol = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)

        sensor_clipped_against_imprint_scaled_for_clipper.append(sol)
        sol_for_ar = np.array([np.array(seg) for seg in pyclipper.scale_from_clipper(sol)])
        ar_intersect = sum(abs(np.array([sum((seg[np.r_[1:len(seg), 0],0]-seg[:,0])*(seg[np.r_[1:len(seg), 0],1]+seg[:,1]))/2. for seg in sol_for_ar])))

        area_intersect.append(ar_intersect)

    pc.Clear()

    area_zones = [[abs(sum((seg[np.r_[1:len(seg), 0],0]-seg[:,0])*(seg[np.r_[1:len(seg), 0],1]+seg[:,1]))/2.) for seg in np.array(segs)] for segs in (segs_zones_left, segs_zones_right)]

    area_intersect_zones=[[],[]]
    for side in (0, 1):
        segs_zone = segs_zones_left if side == 0 else segs_zones_right
        for seg_zone in segs_zone:
            ar_segment_with_pressure_points = []
            for sensor_clipped_segs in sensor_clipped_against_imprint_scaled_for_clipper:
                if sensor_clipped_segs!=[]:
                    pc.Clear()
                    pc.AddPath(pyclipper.scale_to_clipper(seg_zone), pyclipper.PT_CLIP, True)
                    pc.AddPaths(sensor_clipped_segs, pyclipper.PT_SUBJECT, True)
                    sol = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
                    sol_for_ar = np.array([np.array(seg) for seg in pyclipper.scale_from_clipper(sol)])
                    ar = sum(abs(np.array([sum((seg[np.r_[1:len(seg), 0],0]-seg[:,0])*(seg[np.r_[1:len(seg), 0],1]+seg[:,1]))/2. for seg in sol_for_ar])))
                else:
                    ar = 0
                ar_segment_with_pressure_points.append(ar)
            area_intersect_zones[side].append(ar_segment_with_pressure_points)

    area_zones_clipped_against_imprint = [[],[]]
    for side in (0, 1):
        segs_zone = segs_zones_left if side == 0 else segs_zones_right
        segs_imprint = segs_left if side == 0 else segs_right
        for seg_zone in segs_zone:
            pc.Clear()
            pc.AddPath(pyclipper.scale_to_clipper(seg_zone), pyclipper.PT_CLIP, True)
            pc.AddPaths(pyclipper.scale_to_clipper(segs_imprint), pyclipper.PT_SUBJECT, True)
            sol = pc.Execute(pyclipper.CT_INTERSECTION, pyclipper.PFT_NONZERO, pyclipper.PFT_NONZERO)
            sol_for_ar = np.array([np.array(seg) for seg in pyclipper.scale_from_clipper(sol)])
            ar = sum(abs(np.array([sum((seg[np.r_[1:len(seg), 0],0]-seg[:,0])*(seg[np.r_[1:len(seg), 0],1]+seg[:,1]))/2. for seg in sol_for_ar])))
            area_zones_clipped_against_imprint[side].append(ar)



    pressure_calculated_for_zones = [[sum([(ar_zone[k]/area_intersect[k])*pressure_meas[k] if area_intersect[k]!=0 else 0 for k in range(0, len(area_intersect))]) for j, ar_zone in enumerate(ar_zones_s)] for i, ar_zones_s in enumerate(area_intersect_zones)]

    pressure_all = sum(pressure_meas)


    if path_to_write_statistics!="":
        with open(path_to_write_statistics, 'w') as statistics_file:
            statistics_file.write('side,'
                                  'SID,'
                                  'force,'
                                  'force_rel_to_all,'
                                  'force_rel_to_side,'
                                  'force_rel_to_area,'
                                  'area,'
                                  'area_unclipped,'
                                  'area_with_pressure,'
                                  'area_with_pressure_unclipped'
                                  '\n')
            for side_idx in (0,1):
                side = "left" if side_idx==0 else "right"
                segs_zone = segs_zones_left if side_idx==0 else segs_zones_right
                pressure_side =sum(pressure_calculated_for_zones[side_idx])

                for sid in range(0, len(segs_zone)):
                    area_clipped = area_zones_clipped_against_imprint[side_idx][sid]
                    area_unclipped = area_zones[side_idx][sid]
                    area_with_pressure_clipped = sum(area_intersect_zones[side_idx][sid])
                    area_with_pressure_unclipped =sum([1 if a >0 else 0 for a in area_intersect_zones[side_idx][sid]])*row_spacing*col_spacing
                    pressure = pressure_calculated_for_zones[side_idx][sid]
                    statistics_file.write('"{}",{},{},{},{},{},{},{},{},{}\n'.format(
                        side,
                        sid,
                        pressure,
                        (pressure/pressure_all),
                        (pressure/pressure_side),
                        (pressure/area_clipped),
                        area_clipped,
                        area_unclipped,
                        area_with_pressure_clipped,
                        area_with_pressure_unclipped
                    ))




    return(pressure_calculated_for_zones)


















