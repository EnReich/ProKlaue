import numpy as np
import pyclipper
import csv
import sklearn.decomposition
import sklearn.cluster
import math
import copy
from scripts import pressureStatisticsUtilFunctions


bone = "Klaue 1 (K1T)"
ground = "beton"
base_dir = "C:/Users/Kai/Documents/ProKlaue/testdaten/druck"

path_to_tek_csv = "{2}/{0}/{0}_{1}.csv".format(bone, ground.capitalize(), base_dir)
path_write_tek_csv = '{}/{}/pressure_data_{}.csv'.format(base_dir, bone, ground.capitalize())

path_left = '{}/{}/tsf_left.csv'.format(base_dir, bone)
path_right = '{}/{}/tsf_right.csv'.format(base_dir, bone)
th = 0.5

path_zones_left = "{}/segments_for_measurements_left.csv".format(base_dir)
path_zones_right = "{}/segments_for_measurements_right.csv".format(base_dir)




imprintFiles = pressureStatisticsUtilFunctions.calculateImprint(path_left,
                                                                path_right,
                                                                '{}/{}/segments_imprint_{}.csv'.format(base_dir, bone, "left"),
                                                                '{}/{}/segments_imprint_{}.csv'.format(base_dir, bone, "right"))

transformFiles = pressureStatisticsUtilFunctions.calculateFitToPressure(imprint_file_left_path=imprintFiles[0],
                                                                        imprint_file_right_path=imprintFiles[1],
                                                                        pressure_file_path=path_to_tek_csv,
                                                                        path_to_write_transform_imprint='{}/{}/transformation_imprint_{}.csv'.format(base_dir, bone, ground),
                                                                        path_to_write_transform_left_zones='{}/{}/transformation_zones_left_{}.csv'.format(base_dir, bone, ground),
                                                                        path_to_write_transform_right_zones='{}/{}/transformation_zones_right_{}.csv'.format(base_dir, bone, ground),
                                                                        path_to_write_pressure_data='{}/{}/pressure_data_{}.csv'.format(base_dir, bone, ground),
                                                                        path_to_write_pressure_metadata='{}/{}/pressure_metadata_{}.csv'.format(base_dir, bone, ground))




