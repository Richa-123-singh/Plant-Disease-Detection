import numpy as np
import os
from PIL import Image
img_folder = "Plant_Disease_Dataset"

def load_images_from_folder(folder_path):
    dataset = []
    labelset = []
    counter = 0
    for subfolder in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, subfolder)
        # folder_name = os.path.basename(subfolder_path)
        if os.path.isdir(subfolder_path):
            for filename in os.listdir(subfolder_path):
                if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg") or filename.endswith(".JPG"):
                    image_path = os.path.join(subfolder_path, filename)
                    image = Image.open(image_path)
                    pix = np.array(image.getdata()).reshape(image.size[0], image.size[1], 3)
                    dataset.append(pix)
                    labelset.append(counter)
        counter+=1
    return np.asarray(dataset),np.asarray(labelset)
data,label = load_images_from_folder(img_folder)
np.save("./data_10.npy",data)
np.save("./label_10.npy",label)