import sys, getopt
from scripts import pressureStatisticsUtilFunctions

def main(argv):
    step_set, th_set, bone_set, ground_set, base_dir_set = False, False, False, False, False
    try:
      opts, args = getopt.getopt(argv,"hs:t:b:g:d:",["step=","threshold=","bone=","ground=","baseDirectory="])
    except getopt.GetoptError:
      print 'pressureStatistics.py -s <stepNo> -t <threshold> -b <bone> -g <ground> -d <base_dir>'
      sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'pressureStatistics.py -s <stepNo> -t <threshold> -b <bone> -g <ground> -d <base_dir>'
            sys.exit()
        elif opt in ("-s", "--step"):
            step = int(arg)
            step_set=True
        elif opt in ("-t", "--threshold"):
            th = float(arg)
            th_set=True
        elif opt in ("-b", "--bone"):
            bone = arg
            bone_set=True
        elif opt in ("-g", "--ground"):
            ground = arg
            ground_set=True
        elif opt in ("-d", "--baseDirectory"):
            base_dir = arg
            base_dir_set=True

    if not step_set:
        print 'Step not set'
        step = 0         # 0 - do only imprint, 1 - do only transformation, 2 - do only statistics, 3 - do all


    if not th_set:
        print 'Threshold not set'
        th = 0.5

    if not bone_set:
        print 'Bone not set'
        bone = "Klaue 1 (K1T)"

    if not ground_set:
        print 'Ground type not set'
        ground = "gummi"

    if not base_dir_set:
        print 'Base Dir not set'
        base_dir = "C:/Users/Kai/Documents/ProKlaue/testdaten/druck"

    path_to_tek_csv = "{2}/{0}/{0}_{1}.csv".format(bone, ground.capitalize(), base_dir)
    path_left = '{}/{}/tsf_left.csv'.format(base_dir, bone)
    path_right = '{}/{}/tsf_right.csv'.format(base_dir, bone)
    path_zones_left = "{}/segments_for_measurements_left.csv".format(base_dir)
    path_zones_right = "{}/segments_for_measurements_right.csv".format(base_dir)


    if(step==0 or step==3):

        imprintFiles = pressureStatisticsUtilFunctions.calculateImprint(path_left,
                                                                        path_right,
                                                                        '{}/{}/segments_imprint_{}.csv'.format(base_dir, bone, "left"),
                                                                        '{}/{}/segments_imprint_{}.csv'.format(base_dir, bone, "right"),
                                                                        th=th)
    if (step == 1 or step == 3):

        transformFiles = pressureStatisticsUtilFunctions.calculateFitToPressure(imprint_file_left_path='{}/{}/segments_imprint_{}.csv'.format(base_dir, bone, "left"),
                                                                                imprint_file_right_path='{}/{}/segments_imprint_{}.csv'.format(base_dir, bone, "right"),
                                                                                pressure_file_path=path_to_tek_csv,
                                                                                path_to_write_transform_imprint='{}/{}/transformation_imprint_{}.csv'.format(base_dir, bone, ground),
                                                                                path_to_write_transform_left_zones='{}/{}/transformation_zones_left_{}.csv'.format(base_dir, bone, ground),
                                                                                path_to_write_transform_right_zones='{}/{}/transformation_zones_right_{}.csv'.format(base_dir, bone, ground),
                                                                                path_to_write_pressure_data='{}/{}/pressure_data_{}.csv'.format(base_dir, bone, ground),
                                                                                path_to_write_pressure_metadata='{}/{}/pressure_metadata_{}.csv'.format(base_dir, bone, ground))

    if (step == 2 or step == 3):

        pressureStatisticsUtilFunctions.calculateStatistics(path_to_imprint_file_left='{}/{}/segments_imprint_{}.csv'.format(base_dir, bone, "left"),
                                                            path_to_imprint_file_right='{}/{}/segments_imprint_{}.csv'.format(base_dir, bone, "right"),
                                                            path_to_pressure_data='{}/{}/pressure_data_{}.csv'.format(base_dir, bone, ground),
                                                            path_to_pressure_metadata='{}/{}/pressure_metadata_{}.csv'.format(base_dir, bone, ground),
                                                            path_to_transform_imprint='{}/{}/transformation_r_imprint_{}.csv'.format(base_dir, bone, ground),
                                                            path_to_transform_left_zones='{}/{}/transformation_r_zones_left_{}.csv'.format(base_dir, bone, ground),
                                                            path_to_transform_right_zones='{}/{}/transformation_r_zones_right_{}.csv'.format(base_dir, bone, ground),
                                                            path_to_zones_left=path_zones_left,
                                                            path_to_zones_right=path_zones_right,
                                                            path_to_write_statistics='{}/{}/statistics_{}.csv'.format(base_dir, bone, ground))

if __name__ == "__main__":
   main(sys.argv[1:])