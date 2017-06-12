import numpy as np
import pyclipper
import csv
import sklearn.decomposition
import sklearn.cluster
import math
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


MEAN_METHOD = 2 # 0 - mean by mean of single pca's, 1 - mean by center of bounding boxes after pcas, 2 - average between both (wrt weights (mean, bounding boxes))
MEAN_METHOD_WEIGHTS = [0.75,0.25]
CLUSTERING_METHOD = 1 # 0 - Kmeans, 1 - Spectral
CLUSTERING_SCALE_X = 1 # 0 - nothing 1 - scales the x - axis after centering
CLUSTERING_SCALE_X_FACTOR = 0.1
bone = "Klaue 1 (K1T)"
ground = "beton"



# -------------------------------- PRESSURE FILE ---------------------------------------------------------
path_to_tek_csv = "C:/Users/Kai/Documents/ProKlaue/testdaten/druck/{0}/{0}_{1}.csv".format(bone, ground.capitalize())
path_write_tek_csv = 'C:/Users/Kai/Documents/ProKlaue/testdaten/druck/{}/pressure_data_{}.csv'.format(bone, ground.capitalize())

keywords = ["ASCII_DATA @@", "ROW_SPACING", "COL_SPACING"]
with open(path_to_tek_csv, 'r') as file:
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
                    if lsplit[2] == "milimeters":
                        row_spacing = row_spacing*0.1

            elif keyword == "COL_SPACING":
                lsplit = line.split(" ")
                col_spacing = float(lsplit[1])
                if len(lsplit)>2:
                    if lsplit[2] == "milimeters":
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

with open(path_write_tek_csv, 'w') as file:
    file.write("x,y,pressure,rowNo,colNo\n")
    for row in pressure_data:
        file.write('"{}","{}","{}","{}","{}"\n'.format(row[-2], row[-1], row[2], row[0], row[1]))


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

pca_pressure = sklearn.decomposition.PCA(n_components=2).fit([[0,0],[0,0]])
pca_pressure.components_ = np.array([np.mean([pca_pressure_single[1].components_[0], pca_pressure_single[0].components_[0]], axis=0),
                            np.mean([pca_pressure_single[1].components_[1], pca_pressure_single[0].components_[1]], axis=0)
                            ])
#
# pca_pressure = sklearn.decomposition.PCA(n_components=2)
# pca_pressure.fit(pts_pressure[:, -2:], pts_pressure[:, 2])
#
# # pca axes flipped so that first axis is in same direction as x-axis and second axis is in same direction as y-axis
# if abs(pca_pressure.components_[0][1])>abs(pca_pressure.components_[1][1]):
#     pca_pressure.components_ = pca_pressure.components_[::-1]
# if pca_pressure.components_[0][0]<0:
#     pca_pressure.components_[0] *= -1
# if pca_pressure.components_[1][1]<0:
#     pca_pressure.components_[1] *= -1



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



# ----------------------------------------------- SEGMENTS ----------------------------------------------------


path_left = 'C:/Users/Kai/Documents/ProKlaue/testdaten/druck/{}/tsf_left.csv'.format(bone)
path_right = 'C:/Users/Kai/Documents/ProKlaue/testdaten/druck/{}/tsf_right.csv'.format(bone)
th = 0.5

for idx_path, path in enumerate([path_left, path_right]):
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
    file.close()

    segs = np.array(segs)
    # min_3rd_component = np.min(np.array([pt[2] for seg in segs for pt in seg ]))
    # segs = np.array([[[pt[0], pt[1], pt[2]-min_3rd_component] for pt in seg] for seg in segs])
    # segs = [seg for seg in segs if sum([1 if i[2]<th else 0 for i in seg])>2]
    segs = segs_cutoff(segs, th)

    segs = np.array([np.array([pts[:2] for pts in seg]) for seg in segs])

    signed_area = [sum((seg[np.r_[1:len(seg), 0],0]-seg[:,0])*(seg[np.r_[1:len(seg), 0],1]+seg[:,1])) for seg in segs]
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
    signed_area2 = [sum((seg[np.r_[1:len(seg), 0],0]-seg[:,0])*(seg[np.r_[1:len(seg), 0],1]+seg[:,1])) for seg in union2]
    union_without_holes = union2[[i for i, a in enumerate(signed_area2) if a<0]]

    if idx_path == 0:
        segs_left = union_without_holes
    else:
        segs_right = union_without_holes


pts_segs_left = np.vstack(segs_left)
left_mean = np.mean(pts_segs_left, axis=0)
pts_segs_right = np.vstack(segs_right)
right_mean = np.mean(pts_segs_right, axis=0)
pts_all_segs = np.r_[pts_segs_left, pts_segs_right]

pca = sklearn.decomposition.PCA(n_components=2)


#pca.components_ = np.array([[-v[1], -v[0]] for v in pca.components_])

pca.fit(pts_all_segs)

direction_right_left = left_mean-right_mean
direction_right_left /= np.linalg.norm(direction_right_left)
direction_left_right = -direction_right_left # corresponds to x axis in pressure
direction_back_front = np.array([-direction_right_left[1], direction_right_left[0]]) #90 degrees from right->left
direction_front_back = -direction_back_front # corresponds to y axis in pressure

# check if axis of components have the right orientation else make them fit
# if abs(np.dot(direction_front_back, pca.components_[0]))<math.cos(math.pi/4):
#     pca.components_ = pca.components_[::-1]
# if(np.dot(direction_front_back, pca.components_[0])<0):
#     pca.components_[0]*=-1
# if(np.dot(direction_left_right, pca.components_[1])<0):
#     pca.components_[1]*=-1


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
    # pts_transformed_to_estimate_mean = np.array([pca.transform([pt for seg in segs for pt in seg]) for segs in [segs_left, segs_right]])
    # min_x = [min(pts[:,0]) for pts in pts_transformed_to_estimate_mean]
    # max_x = [max(pts[:,0]) for pts in pts_transformed_to_estimate_mean]
    # min_y = [min(pts[:,1]) for pts in pts_transformed_to_estimate_mean]
    # max_y = [max(pts[:,1]) for pts in pts_transformed_to_estimate_mean]
    # estimated_mean = pca.inverse_transform([np.mean([np.mean(min_x), np.mean(max_x)]), np.mean([np.mean(min_y), np.mean(max_y)])])
    # pca.mean_ = estimated_mean
elif MEAN_METHOD==2:
    pca.mean_ = np.average([np.mean([left_mean, right_mean], axis=0),
                         estimate_mean_by_bounding_pca(
                             np.array([[pt for seg in segs for pt in seg] for segs in [segs_left, segs_right]]), pca)
                         ],axis=0, weights=MEAN_METHOD_WEIGHTS)

# pca.components_ = np.array([direction_front_back,
#                             direction_left_right])

#
# pca_aligned_at_pressure = copy.deepcopy(pca);
#
# pca.components_ = pca.components_[::-1]
# pca.components_[1] = -pca.components_[1]
# if pca.transform([left_mean])[0][1] > pca.transform([right_mean])[0][1]:
#     pca.components_ *= -1
# if angle(*pca.components_)>0:
#     pca.components_[0] *= -1
#


# if angle(*pca_pressure.components_)*angle(*pca_aligned_at_pressure.components_)<0:
#     pca_aligned_at_pressure.components_[0] *= -1
# angle_pca_pressure = angle(pca_aligned_at_pressure.components_[0], pca_pressure.components_[0])
# if abs(angle_pca_pressure)>135:
#     pca_aligned_at_pressure.components_= -pca_pressure.components_
# elif abs(angle_pca_pressure)>45:
#     if angle_pca_pressure>0:
#         pca_aligned_at_pressure.components_[0]= -pca_pressure.components_[1]
#         pca_aligned_at_pressure.components_[1] = pca_pressure.components_[0]
#     else:
#         pca_aligned_at_pressure.components_[0] = pca_pressure.components_[1]
#         pca_aligned_at_pressure.components_[1] = -pca_pressure.components_[0]
# else:
#     pca_aligned_at_pressure.components_ = pca_pressure.components_

# pca_aligned_at_pressure.mean_ -= pca_pressure.mean_

# pca_single = [sklearn.decomposition.PCA(n_components=2).fit(pts) for pts in [pts_segs_left, pts_segs_right]]
#
# angle_comps = [[angle(comp_s, pca.components_[i]) for i, comp_s in enumerate(pca_s.components_)] for pca_s in pca_single]
#
# for i, pca_s in enumerate(pca_single):
#     if abs(angle_comps[i][0]) > 90:
#         pca_s.components_ *= -1



# segs_transformed = [np.array([ pca_single[i].transform(seg) for seg in segs]) for i, segs in enumerate([segs_left, segs_right])]
# segs_transformed = [np.array([ pca.transform(seg) for seg in segs]) for segs in [segs_left, segs_right]]
segs_transformed = [np.array([ pca_pressure.inverse_transform(pca.transform(seg)) for seg in segs]) for segs in [segs_left, segs_right]]
# pts_transformed = [np.vstack(segs) for segs in segs_transformed]



#
# min_x = [min(pts[:,0]) for pts in pts_transformed]
# max_x = [max(pts[:,0]) for pts in pts_transformed]
# min_y = [min(pts[:,1]) for pts in pts_transformed]
# max_y = [max(pts[:,1]) for pts in pts_transformed]

# midpoint = np.array([(np.array(max_x)-np.array(min_x))/2 + min_x, (np.array(max_y)-np.array(min_y))/2 + min_y]).transpose()


# for i, segs in enumerate(segs_transformed):
#     for seg in segs:
#         seg[:,0] -= min_x[0]
#         seg[:,1] -= min_y[0]


# segs_single_transformed = [np.array([ pca_single[i].transform(seg) for seg in segs]) for i, segs in enumerate([segs_left, segs_right])]

for idx_side, segs in enumerate(segs_transformed):
    side = "left" if idx_side==0 else "right"
    with open('C:/Users/Kai/Documents/ProKlaue/testdaten/druck/{}/segments_without_holes_{}_{}.csv'.format(bone, side, ground), 'w') as o_file_seg:
        o_file_seg.write("SID,PID,x,y\n")
        for SID, seg in enumerate(segs):
            for PID, p in enumerate(seg):
                o_file_seg.write('"{}","{}","{}","{}"\n'.format(SID, PID, p[0], p[1]))




