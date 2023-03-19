import torch
import os
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, img_dir, label_dir, S=7, B=2, C=20, transform= None): # transform for data augmentation
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        # ------------ processing labels ----------------
        label_path = os.path.join(self.label_dir, self.annotations.iloc[index, 1]) # the label filename is in the second column of the csv_file
        boxes = []
        with open(label_path) as f:
            for label in f.readlines():
                class_label, x, y, width, height = [float(x) if float(x)!=int(float(x)) else int(x)
                                                    for x in label.replace("\n", "").split()]
                
                boxes.append([class_label, x, y, width, height])
        
        # ------------- processing images ----------------
        img_path = os.path.join(self.img_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_path)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5*self.B))
        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            i, j = int(self.S*y), int(self.S*x) # to know the cell of the box in images, x, y are normalized to be in [0,1] so multiply by the number of cells to know the corresponding cell of the box
            x_cell, y_cell = self.S*x - j, self.S*y - i 
            width_cell, height_cell = (width* self.S, height* self.S)
            
            
            if label_matrix[i, j, 20] == 0: # if there is no object in the calculated cell i, j
                label_matrix[i, j, 20] = 1
                box_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                #box_coordinates = torch.tensor([x_cell, y_cell, width, height])
                label_matrix[i, j, 21:25] = box_coordinates
                label_matrix[i, j, class_label] = 1 # probability of the presence of object in that cell is equal to 1

        return image, label_matrix

"""
csv_file= 'data/8examples.csv'
image_dir= 'data/images'
label_dir = 'data/labels'
index= 2

def test():
    dataloader = VOCDataset(csv_file, image_dir, label_dir)
    l = dataloader.__len__()
    image, label_matrix = dataloader.__getitem__(index)

    print('len csv_file :', l, 'label_matrix dim :', label_matrix.shape)
    print('sum label_matrix :', sum(label_matrix.flatten()))
    print(np.array(image).shape)

test()
"""


