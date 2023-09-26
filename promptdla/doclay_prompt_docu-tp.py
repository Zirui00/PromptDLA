import os
import clip
import torch
import json

# Load the model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load('ViT-B/32', device)

text_inputs_0 = torch.cat([clip.tokenize("a document page of government tenders")]).to(device)
text_inputs_1 = torch.cat([clip.tokenize("a piece of paper concerning with laws and regulations")]).to(device)
text_inputs_2 = torch.cat([clip.tokenize("a page comes from financial reports")]).to(device)
text_inputs_3 = torch.cat([clip.tokenize("a piece of paper comes from patents")]).to(device)
text_inputs_4 = torch.cat([clip.tokenize("a document page of manuals")]).to(device)
text_inputs_5 = torch.cat([clip.tokenize("a page of scientific articles")]).to(device)

a = torch.zeros(1, 256).to(device)

# Calculate features
with torch.no_grad():
    text_features_0 = model.encode_text(text_inputs_0).tolist()
    text_features_1 = model.encode_text(text_inputs_1).tolist()
    text_features_2 = model.encode_text(text_inputs_2).tolist()
    text_features_3 = model.encode_text(text_inputs_3).tolist()
    text_features_4 = model.encode_text(text_inputs_4).tolist()
    text_features_5 = model.encode_text(text_inputs_5).tolist()

mydict = {"government_tenders":text_features_0, "laws_and_regulations":text_features_1, "financial_reports":text_features_2, "patents":text_features_3, "manuals":text_features_4, "scientific_articles":text_features_5}

with open("prompts/Doclay_doc-ty.json", "w") as f:
    f.write(json.dumps(mydict))
