import os
import clip
import torch
import json

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

text_inputs_0 = torch.cat([clip.tokenize("a document page of Scientific articles")]).to(device)
text_inputs_1 = torch.cat([clip.tokenize("a piece of paper concerning with textbooks")]).to(device)
text_inputs_2 = torch.cat([clip.tokenize("a photo of books")]).to(device)
text_inputs_3 = torch.cat([clip.tokenize("a page comes from text papers")]).to(device)
text_inputs_4 = torch.cat([clip.tokenize("a piece of paper comes from magazines")]).to(device)
text_inputs_5 = torch.cat([clip.tokenize("a document page of newspapers")]).to(device)
text_inputs_6 = torch.cat([clip.tokenize("a page of notes")]).to(device)

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

mydict = {"Scientific articles":text_features_0, "Textbooks":text_features_1, "Books(photo)":text_features_2, "Test papers":text_features_3, "Magazines":text_features_4, "Newspapers":text_features_5, "Notes":text_features_6}

with open("prompts/M6Doc_doc-ty.json", "w") as f:
    f.write(json.dumps(mydict))
