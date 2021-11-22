#Resize original images to n_px by n_px
import cv2
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
from model import ImageGPT2LMHeadModel
from utils import *
import argparse
import os

parser = argparse.ArgumentParser(description='Defines model and inputs for image-gpt')
parser.add_argument('--model_size','--size', type=str, required=True, help="size of the igpt-model")
parser.add_argument('--models_dir','--mdir', type=str, required=True, help="path to the model directory")
parser.add_argument('--color_clusters_dir','--ccdir', type=str, required=True, help="path to the color clusters directory")
parser.add_argument('--image_path','--link',type=str,required=True, help="link to the image to be executed")

args = parser.parse_args()

bs = 8 
n_px = 32

dim=(n_px,n_px)

color_clusters_file = "%s/kmeans_centers.npy"%(args.color_clusters_dir)
clusters = np.load(color_clusters_file) #get color clusters

MODELS={"l":(1536,16,48),"m":(1024,8,36),"s":(512,8,24) } 
n_embd,n_head,n_layer=MODELS[args.model_size] #set model hyperparameters
vocab_size = len(clusters) + 1 #add one for start of sentence token
config = transformers.GPT2Config(vocab_size=vocab_size,n_ctx=n_px*n_px,n_positions=n_px*n_px,n_embd=n_embd,n_layer=n_layer,n_head=n_head)
model_path = "%s/model.ckpt-1000000.index"%(args.models_dir)

model = ImageGPT2LMHeadModel.from_pretrained(model_path,from_tf=True,config=config)

x = np.zeros((bs,n_px,n_px,3),dtype=np.uint8)

# curl args.image_path > sg.jpeg
image_paths = ["sg.jpeg"]*bs

for n,image_path in enumerate(image_paths):
  img_np = cv2.imread(image_path)   # reads an image in the BGR format
  # print(img_np)
  img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)   # BGR -> RGB
  H,W,C = img_np.shape
  D = min(H,W)
  img_np = img_np[:D,:D,:C] #get square piece of image
  x[n] = cv2.resize(img_np,dim, interpolation = cv2.INTER_AREA) #resize to n_px by n_px

#visualize resized images
f, axes = plt.subplots(1,bs,dpi=300)

for img,ax in zip(x,axes):
    ax.axis('off')
    ax.imshow(img)

#use Image-GPT color palette and crop images
x_norm = normalize_img(x) #normalize pixels values to -1 to +1
samples = color_quantize_np(x_norm,clusters).reshape(x_norm.shape[:-1]) #map pixels to closest color cluster

n_px_crop = 16
primers = samples.reshape(-1,n_px*n_px)[:,:n_px_crop*n_px] # crop top n_px_crop rows. These will be the conditioning tokens

#visualize samples and crops with Image-GPT color palette. Should look similar to original resized images
samples_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px, n_px, 3]).astype(np.uint8) for s in samples] # convert color clusters back to pixels
primers_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px_crop,n_px, 3]).astype(np.uint8) for s in primers] # convert color clusters back to pixels


f, axes = plt.subplots(1,bs,dpi=300)
for img,ax in zip(samples_img,axes):
    ax.axis('off')
    ax.imshow(img)

f, axes2 = plt.subplots(1,bs,dpi=300)
for img,ax in zip(primers_img,axes2):
    ax.axis('off')
    ax.imshow(img)

context = np.concatenate( (np.full( (bs,1), vocab_size - 1 ),primers,), axis=1 )
context = torch.tensor(context)
output = model.generate(input_ids=context,max_length= n_px*n_px + 1,temperature=1.0,do_sample=True,top_k=40)

#visualize samples with Image-GPT color palette. 
samples = output[:,1:].cpu().detach().numpy()
samples_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px, n_px, 3]).astype(np.uint8) for s in samples] # convert color cluster tokens back to pixels
f, axes = plt.subplots(1,bs,dpi=300)

for img,ax in zip(samples_img,axes):
    ax.axis('off')
    ax.imshow(img)
    