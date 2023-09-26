import os
import clip
import torch
import json

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

text_inputs_0 = torch.cat([clip.tokenize("don't have Page footer, Page header, and formula")]).to(device)
text_inputs_1 = torch.cat([clip.tokenize("lists are divided into small items")]).to(device)

a = torch.zeros(1, 256).to(device)

# Calculate features
with torch.no_grad():
    text_features_0 = model.encode_text(text_inputs_0).tolist()
    text_features_1 = model.encode_text(text_inputs_1).tolist()

mydict = {"publaynet":text_features_0, "doclaynet":text_features_1}

with open("pub_doc_label-style.json", "w") as f:
    f.write(json.dumps(mydict))
