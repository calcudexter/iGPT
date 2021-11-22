import numpy as np
import torch
import pathlib
import matplotlib.pyplot as plt
from model import ImageGPT2LMHeadModel
import argparse

arser = argparse.ArgumentParser(description='Defines model and inputs for image-gpt')
parser.add_argument('--model_size','--size', type=str, required=True, help="size of the igpt-model")
parser.add_argument('--models_dir','--mdir', type=str, required=True, help="path to the model directory")
parser.add_argument('--color_clusters_dir','--ccdir', type=str, required=True, help="path to the color clusters directory")
parser.add_argument('--image_path','--link',type=str,required=True, help="link to the image to be executed")

args = parser.parse_args()

bs = 8 
n_px = 32

color_clusters_file = "%s/kmeans_centers.npy"%(args.color_clusters_dir)
clusters = np.load(color_clusters_file) #get color clusters

MODELS={"l":(1536,16,48),"m":(1024,8,36),"s":(512,8,24) } 
n_embd,n_head,n_layer=MODELS[args.model_size] #set model hyperparameters
vocab_size = len(clusters) + 1 #add one for start of sentence token
config = transformers.GPT2Config(vocab_size=vocab_size,n_ctx=n_px*n_px,n_positions=n_px*n_px,n_embd=n_embd,n_layer=n_layer,n_head=n_head)
model_path = "%s/model.ckpt-1000000.index"%(args.models_dir)

model = ImageGPT2LMHeadModel.from_pretrained(model_path,from_tf=True,config=config)

context = np.full( (bs,1), vocab_size - 1 ) #initialize with SOS token
context = torch.tensor(context).cuda()
output = model.generate(input_ids=context,max_length= n_px*n_px + 1,temperature=1.0,do_sample=True,top_k=40)

#visualize samples with Image-GPT color palette.
samples = output[:,1:].cpu().detach().numpy()
samples_img = [np.reshape(np.rint(127.5 * (clusters[s] + 1.0)), [n_px, n_px, 3]).astype(np.uint8) for s in samples] # convert color cluster tokens back to pixels
f, axes = plt.subplots(1,bs,dpi=300)

for img,ax in zip(samples_img,axes):
    ax.axis('off')
    ax.imshow(img)