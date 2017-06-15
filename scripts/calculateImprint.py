import numpy as np
import pyclipper
import csv
from pressureStatistics import segs_cutoff, write_segs_to_file

# script to calculate the imprint of a list of front facing triangles of an object

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

    for idx_side, segs in enumerate([segs_left, segs_right]):
        output_path = path_to_write_left_segs if idx_side==0 else path_to_write_right_segs
        write_segs_to_file(segs, output_path)


    return([path_to_write_left_segs, path_to_write_right_segs])


