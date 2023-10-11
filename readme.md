# PromptDLA

In this paper, we propose a simple yet effective domain-aware
Prompt Document Layout Analysis (PromptDLA) framework to incorporate domain prior into DLA through domain prompts. The
proposed novel PromptDLA features a unique domain-aware prompter that customizes prompts according to the specific
attributes of the data domain. Then these prompts can guide the DLA toward essential features and structures in the
data, leading to better generalization of the DLA across different domains.

![](architecture.png)

## Setup

Here is a from-scratch script.

```bash
conda create -n PromptDLA python=3.8
conda activate PromptDLA

# install pytorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# install dependency 
pip install -r requirements.txt
python -m pip install -e detectron2
```

## Data preparation

Download dateset into "./data_prepare/*_data", where * is from {doclaynet, publaynet, m6doc}

DocLayNet download url: [DocLayNet Dataset](https://developer.ibm.com/exchanges/data/all/doclaynet/)

```
├── doclaynet_data
    ├── COCO
    │   ├── train.json
    │   │── val.json
    │   └── test.json
    └──  PNG
        ├── xxx.png
        ├── xxx.png
        └── ...
```

Publaynet download url: [PubLayNet Datasest](https://developer.ibm.com/exchanges/data/all/publaynet/)

```
├── publaynet_data
    ├── train.json
    │  
    ├── val.json
    │  
    ├── train
    │   ├── xxx.jpg
    │   └── ...
    └── val
        ├── xxx.jpg
        └── ...
```

M6Doc download url: [M6Doc Dataset](https://github.com/HCIILAB/M6Doc)

```
├── m6doc_data
    ├── annotations
    │   ├── instances_train2017.json
    │   │── instances_val2017.json
    │   └── instances_test.json
    ├── train2017
    │   ├── xxx.jpg
    │   └── ...
    ├── val2017
    │   ├── xxx.jpg
    │   └── ...
    └── test2017
        ├── xxx.jpg
        └── ...

```

You can also change the saving directory, but remember to change the dataset path in "./promptdla/train_net.py
register_coco_instances()"

```bash
cd data_prepare/
```

### document type

```bash
# DocLayNet
python doclaynet_document-type.py --annotations doclaynet_data/COCO/train.json
python doclaynet_document-type.py --annotations doclaynet_data/COCO/val.json
python doclaynet_document-type.py --annotations doclaynet_data/COCO/test.json

# M6Doc
python m6doc_document-type.py --annotations m6doc_data/annotations/instances_train2017.json
python m6doc_document-type.py --annotations m6doc_data/annotations/instances_val2017.json
python m6doc_document-type.py --annotations m6doc_data/annotations/instances_test2017.json
```

Then you will get file "[origin-name]_document-type.json"

### integration of publaynet and doclayent

```bash
# PubLayNet
python publaynet_label-style.py --annotations publaynet_data/train.json
python publaynet_label-style.py --annotations publaynet_data/val.json

# DocLayNet
python doclaynet_label-style.py --annotations doclaynet_data/COCO/train.json
python doclaynet_label-style.py --annotations doclaynet_data/COCO/val.json
python publaynet_label-style.py --annotations doclaynet_data/COCO/test.json
```

Then you will get file "[origin-name]_label-style.json"

Another important thing is that, since the objects in PubLayNet is three times more than those in DocLayNet, to prevent
the bias when training, we down scale the size of PubLayNet into 1/3.

```bash
python mini_publaynet.py --annotations publaynet_data/train_label-style.json
python mini_publaynet.py --annotations publaynet_data/train_label-style.json
```
Then you will get file "[train/val]_mini_label-style.json"
## Obtain Prompt

```bash
cd ../promptdla/
```

```bash
# DocLayNet document type prompt
python doclay_prompt_docu-tp.py

# M6Doc document type prompt
python m6doc_prompt_docu-tp.py

# Integration of DocLayNet and PubLayNet Label style prompt
python doclay_publay_label-style.py
```

the prompt json file will be saved in "./promptdla/prompts/"

You can also try your own prompt template,
by modifying the sentence in clip.tokenize()

## Finetuned Models

| Model                                                   | Model Name (Path)             | 
|---------------------------------------------------------|-------------------------------|
| DocLayNet with document type                            | [doclaynet_prompt_docu-tp]()  |
| DocLayNet without domain                                | [doclaynet_noprompt]()        |
| M6Doc with document type                                | [m6doc_prompt_docu-tp]()      |
| M6Doc without domain                                    | [m6doc_noprompt]()            |
| Integration of PubLayNet and DocLayNet with label style | [doclay_publay_label-style]() |
| Integration of PubLayNet and DocLayNet without domain   | [doclay_publay_noprompt]()    |

## Inference

```bash
python inference.py --dataset <dataset name> --image_dir <path to image dir> --output_dir <path to save dir> --config <path to config file> --use_prompt 1 --prompt_file <path to prompt> MODEL.WEIGHTS <path to finetuned model>

# where, --use_prompt {1: PrompDLA, 0: no prompt}
#        --dataset {doclaynet, m6doc, publay_doclay}

# for example
python inference.py --dataset doclaynet --image_dir images/ --output_dir inferred_images/ --config configs/doclaynet_configs/cascade/cascade_dit_base.yaml --use_prompt 1 --prompt_file prompts/Doclay_doc-ty.json -- MODEL.WEIGHTS weights/doclay_prompt.pth
python inference.py --dataset doclaynet --image_dir images/ --output_dir inferred_images/ --config configs/doclaynet_configs/cascade/cascade_dit_base.yaml --use_prompt 0 -- MODEL.WEIGHTS weights/doclay_prompt.pth
```

## Evaluation

```bash
python train_net.py --dataset <dataset name> --config-file [path to config file] --eval-only --num-gpus 8 --use_prompt 1 --prompt_file <path to prompt> MODEL.WEIGHTS <path to finetuned model> OUTPUT_DIR <your_output_dir> 

# where, --use_prompt {1: PrompDLA, 0: no prompt}
#        --dataset {doclaynet, m6doc, publay_doclay}

# for example
python train_net.py --dataset doclaynet --config-file configs/doclaynet_configs/cascade/cascade_dit_base.yaml --eval-only --num-gpus 8 --use_prompt 1 --prompt_file prompts/Doclay_doc-ty.json MODEL.WEIGHTS weights/doclay_prompt.pth OUTPUT_DIR evaluation/doclaynet/prompt
python train_net.py --dataset doclaynet --config-file configs/doclaynet_configs/cascade/cascade_dit_base.yaml --eval-only --num-gpus 8 --use_prompt 0 MODEL.WEIGHTS weights/doclay_prompt.pth OUTPUT_DIR evaluation/doclaynet/noprompt
```

## Pre-trained Models

| Model                               | Model Name (Path)                                                                                                                                                                                                                          | 
|-------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Pre-trained weights from DiT        | [DiT_base](https://layoutlm.blob.core.windows.net/dit/dit-pts/dit-base-224-p16-500k-62d53a.pth?sv=2022-11-02&ss=b&srt=o&sp=r&se=2033-06-08T16:48:15Z&st=2023-06-08T08:48:15Z&spr=https&sig=a9VXrihTzbWyVfaIDlIT1Z0FoR1073VB0RLQUMuudD4%3D) |
| Pre-trained weights from LayoutLMv3 | [LayoutLMv3_base](https://huggingface.co/microsoft/layoutlmv3-base)                                                                                                                                                                        |

Our methods are primarily implemented on DiT pre-trained model, but we also get a promotion on other pre-trained model
such as LayoutLMv3. If you want to test our method on LayoutLMv3, feel free to contact us.

## Training

```bash
python train_net.py --dataset <dataset name> --config-file [path to config file] --num-gpus 8 --use_prompt 1 --prompt_file <path to prompt> MODEL.WEIGHTS <path to pre-trained model> OUTPUT_DIR <your_output_dir> 

# where, --use_prompt {1: PrompDLA, 0: no prompt}
#        --dataset {doclaynet, m6doc, publay_doclay}

# for example
python train_net.py --dataset doclaynet --config-file configs/doclaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 8 --use_prompt 1 --prompt_file prompts/Doclay_doc-ty.json MODEL.WEIGHTS weights/dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR finetune/doclaynet/prompt
python train_net.py --dataset doclaynet --config-file configs/doclaynet_configs/cascade/cascade_dit_base.yaml --num-gpus 8 --use_prompt 0 MODEL.WEIGHTS weights/dit-base-224-p16-500k-62d53a.pth OUTPUT_DIR finetune/doclaynet/noprompt
```


