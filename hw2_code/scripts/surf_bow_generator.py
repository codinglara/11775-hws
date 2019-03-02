import numpy as np
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from collections import Counter
from pathlib import Path

file_path = "../surf/" 

def get_surf_features(list_path, pickle_file, num_clusters, save_path):

    fileObject = open(pickle_file,'rb')

    # Get the centers obtained from KMeans clustering
    kmodel = pickle.load(fileObject)
    k_centers = np.array(kmodel.cluster_centers_)
    nodes = k_centers

    def closest_node(node):
       dist_2 = np.sum((nodes - node)**2, axis=1)
       min_val = np.argmin(dist_2)
       return min_val

    f = open(list_path, "r")

    final_df = []

    ctr = 0

    file_names = []
    target_vals = []

    for line in f:
        t = line.split()

        open_path = file_path + str(t[0]) + ".surf"

        if(Path(open_path).is_file()):
            print(list_path, "Num clusters: ", num_clusters, t[0], t[1])

            ctr += 1

            df=pd.read_csv(open_path, sep=',')

            surf_rows = np.array(df)

            # Array with all ocurences of the features
            occur_list = np.apply_along_axis(closest_node, 1, surf_rows)

            # Frequency row for features
            freq_row = Counter(occur_list)

            a = np.zeros(num_clusters, dtype=int)

            for key in freq_row:
                a[int(key)] = freq_row[key]

            file_names.append(t[0])
            target_vals.append(t[1])

            # Append a to final_df array
            final_df.append(a)

            print(ctr, t[0], t[1])
            
            column_names = []

    for i in range(num_clusters):
        column_names.append("S" + str(i))
        
    use_df = pd.DataFrame(final_df,index=None, columns=column_names)
    
    file_names_df = pd.DataFrame(file_names,index=None, columns=['name'])

    target_vals_df = pd.DataFrame(target_vals, index=None, columns=['target'])

    frames = [file_names_df, use_df, target_vals_df]

    result = pd.concat(frames, sort=False, axis=1)

    result.to_csv(path_or_buf=save_path)
    
    print("[INFO] Completed job ", save_path)

get_surf_features("../../all_trn.lst", "../kmeans.400.model", 400, "../surf_bow/surf_400_trn.csv")
get_surf_features("../../all_val.lst", "../kmeans.400.model", 400, "../surf_bow/surf_400_val.csv")
get_surf_features("../../all_test_fake.lst", "../kmeans.400.model", 400, "../surf_bow/surf_400_test.csv")

get_surf_features("../../all_trn.lst", "../kmeans.1000.model", 1000, "../surf_bow/surf_1000_trn.csv")
get_surf_features("../../all_val.lst", "../kmeans.1000.model", 1000, "../surf_bow/surf_1000_val.csv")
get_surf_features("../../all_test_fake.lst", "../kmeans.1000.model", 1000, "../surf_bow/surf_1000_test.csv")

get_surf_features("../../all_trn.lst", "../kmeans.2000.model", 2000, "../surf_bow/surf_2000_trn.csv")
get_surf_features("../../all_val.lst", "../kmeans.2000.model", 2000, "../surf_bow/surf_2000_val.csv")
get_surf_features("../../all_test_fake.lst", "../kmeans.2000.model", 2000, "../surf_bow/surf_2000_test.csv")
