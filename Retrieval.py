import clip
from PIL import Image
import torch
import numpy as np
import faiss
import matplotlib.pyplot as plt
import os
import json
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("ignore")

model, preprocess = clip.load("ViT-B/32")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
bin_file = "./File_bin/Database.npy"
dict_id2img_json = "./Dict/keyframe_id2path.json"

# Load Faiss
feature_npy = np.load(bin_file)
m = 8  # number of centroid IDs in final compressed vectors
bits = 8 # number of bits in each centroid
nlist = 50  # how many cells
d = 512
quantizer = faiss.IndexFlatL2(d)  # we keep the same L2 distance flat index
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, bits) 
index.train(feature_npy)
index.add(feature_npy)

# Image
def encode_image(image_path):
    raw_image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    image_feature = model.encode_image(raw_image).cpu().detach().numpy()
    
    return image_feature

# Text
def encode_text(text):
    txt = clip.tokenize(text).to(device)
    text_feature = model.encode_text(txt).cpu().detach().numpy()

    return text_feature

# Return index
def return_image_id(img_path = None, text = None):
    k = 10 # Number of pictures
    if img_path != None:
        image_feature = encode_image(img_path)
        D, I = index.search(image_feature, k)
    else:
        text_feature = encode_text(text)
        D, I = index.search(text_feature, k)

    return I[0]

# Test
def show_image(total_images = 10, input_image=None, input_text = None):
    # Load dict file
    with open(dict_id2img_json, "r", encoding="utf-8") as f:
        dict_id2img = json.load(f)

    # 
    cols = 5
    rows = (total_images+ cols - 1) // cols  # 

    # 
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))
    axes = axes.flatten()  # 

    for i, id in enumerate(return_image_id(img_path=input_image, text=input_text)):
        image = Image.open(os.path.join("./Database", dict_id2img[str(id)]))  # 
        axes[i].imshow(image) 
        axes[i].axis("off")  
        axes[i].set_title(f"Image {i+1}") 

    # 
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()

def get_args():
    parser = ArgumentParser(add_help=False)
    parser.add_argument("--input_image", type=str, help="Path image", required=False)
    parser.add_argument("--input_text", type=str, help="text", required=False)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    show_image(input_image= args.input_image, input_text=args.input_text)