import os
from tqdm import tqdm
import numpy as np
import clip
from PIL import Image
import torch
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

# Model
model, preprocess = clip.load("ViT-B/32")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Extract features
def read_image(image_path):
    # load sample image
    raw_image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)

    return raw_image

def extract_features(data_path, bin_path):
    features_list = []
    list_car = sorted(os.listdir(data_path))  # Đảm bảo đúng thứ tự ảnh
    for type_car in tqdm(list_car):
        print(type_car)
        type_car_path = os.path.join(data_path, type_car)
        for car in sorted(os.listdir(type_car_path)):
            # Trích xuất đặc trưng
            image_path = os.path.join(type_car_path, car)
            image = read_image(image_path)
            feature = model.encode_image(image)[0].cpu().detach().numpy()

            # Chuyển về kích thước (1, 512) và lưu vào danh sách
            features_list.append(feature)
    feature_npy = np.array(features_list)
    #Lưu vào file
    np.save(bin_path, feature_npy)

root_dir = './Database'
bin_path = './File_bin/Database.npy'
extract_features(root_dir, bin_path)
