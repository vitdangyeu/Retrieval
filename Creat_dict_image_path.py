from tqdm import tqdm
import os
import json
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

# Create json key to image
def key_to_image(data_path, json_path):
    # Create Dictionary
    dic = {}
    i = 0
    list_car = sorted(os.listdir(data_path))  # Đảm bảo đúng thứ tự ảnh
    for type_car in tqdm(list_car):
        type_car_path = os.path.join(data_path, type_car)
        for car in sorted(os.listdir(type_car_path)):
            image_path = os.path.join(type_car_path, car)
            result = image_path.split("Database/")[-1]  # Lấy phần sau "Keyframes_"
            dic[i] = result
            i += 1

    # Save file json
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dic, f, indent=4, ensure_ascii=False)

root_dir = './Database'
json_path = './Dict/keyframe_id2path.json'
key_to_image(root_dir, json_path)




