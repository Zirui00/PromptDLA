import os
import clip
import torch
import json

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

text_inputs_0 = torch.cat([clip.tokenize("a Persian document page")]).to(device)
text_inputs_1 = torch.cat([clip.tokenize("a piece of  Khmer paper")]).to(device)
text_inputs_2 = torch.cat([clip.tokenize("a Kazakh page")]).to(device)
text_inputs_3 = torch.cat([clip.tokenize("a document page in Lao")]).to(device)
text_inputs_4 = torch.cat([clip.tokenize("a piece of paper in Turkish")]).to(device)
text_inputs_5 = torch.cat([clip.tokenize("a Hindi paper")]).to(device)
text_inputs_6 = torch.cat([clip.tokenize("a page of Vietnamese document")]).to(device)

a = torch.zeros(1, 256).to(device)

# Calculate features
with torch.no_grad():
    text_features_0 = model.encode_text(text_inputs_0).tolist()
    text_features_1 = model.encode_text(text_inputs_1).tolist()
    text_features_2 = model.encode_text(text_inputs_2).tolist()
    text_features_3 = model.encode_text(text_inputs_3).tolist()
    text_features_4 = model.encode_text(text_inputs_4).tolist()
    text_features_5 = model.encode_text(text_inputs_5).tolist()
    text_features_6 = model.encode_text(text_inputs_6).tolist()

mydict = {"bosi":text_features_0, "gaomian":text_features_1, "hasake":text_features_2, "laowo":text_features_3, "tuerqi":text_features_4, "yindi":text_features_5, "yuenan":text_features_6}

with open("prompts/fanyu_language.json", "w") as f:
    f.write(json.dumps(mydict))
