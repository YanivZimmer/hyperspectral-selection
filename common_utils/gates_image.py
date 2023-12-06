import os
import json
import uuid

import numpy as np
from matplotlib import pyplot as plt


class GatesImageCreator:
    def __init__(self, dir_path, dataset_name, n_bands):
        self.dir_path = dir_path
        self.gates_number_to_idx = {}
        self.n_bands = n_bands
        self.dataset_name = dataset_name

    def gates_progression_image(self,matrix):
        plt.clf()
        plt.imshow(matrix, cmap='gray_r', vmin=0, vmax=1)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.show()
        #plt.savefig(f'{dataset}/grayscale_image_final_gates_guid_{str(uuid.uuid4())}.png')


    def create_gates_matrix(self):
        gates_matrix = np.zeros((self.n_bands,))
        for k,v in sorted(self.gates_number_to_idx.items()):
            for gates in v:
                if int(k) != sum(np.array(gates) > 0):
                    continue
                gates_matrix = np.vstack((gates_matrix, np.array(gates)))
        return gates_matrix

    def append_data_to_gates_number_to_idx(self,data):
        for k_t,v in data.items():
            k=int(k_t)
            if k in self.gates_number_to_idx.keys():
                self.gates_number_to_idx[k].append(v)
            else:
                self.gates_number_to_idx[k]=[v]

    def iterate_files(self):
        file_names = os.listdir(self.dir_path)
        for name in file_names:
            path = os.path.join(self.dir_path,name)
            with open(path,mode='r') as f:
                data = json.load(f)
                self.append_data_to_gates_number_to_idx(data)

if __name__=='__main__':
    gCreator = GatesImageCreator('../PaviaU/gates',dataset_name='PaviaUmy',n_bands=103)
    gCreator.iterate_files()
    print(gCreator.gates_number_to_idx)
    matrix = gCreator.create_gates_matrix()
    print(matrix)
    gCreator.gates_progression_image(matrix)
