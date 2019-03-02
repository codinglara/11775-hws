import numpy
import os
import sys

file_list = "../list/train.video"
ratio = 0.1
output_file = "select.surf.csv"

fread = open(file_list,"r")
fwrite = open(output_file,"w")

# random selection is done by randomizing the rows of the whole matrix, and then selecting the first 
# num_of_frame * ratio rows

numpy.random.seed(18877)

for line in fread.readlines():
    surf_path = "../surf/" + line.replace('\n','') + ".surf"
    if os.path.exists(surf_path) == False:
        continue
    print(surf_path)
    array = numpy.genfromtxt(surf_path, delimiter=",")
    numpy.random.shuffle(array)
    select_size = int(array.shape[0] * ratio)
    feat_dim = array.shape[1]

    for n in range(select_size):
        line = str(array[n][0])
        for m in range(1, feat_dim):
            line += ',' + str(array[n][m])
        fwrite.write(line + '\n')
fwrite.close()
