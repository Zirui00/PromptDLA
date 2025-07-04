#!/usr/bin/env python
# --------------------------------------------------------------------------------
# MPViT: Multi-Path Vision Transformer for Dense Prediction
# Copyright (c) 2022 Electronics and Telecommunications Research Institute (ETRI).
# All Rights Reserved.
# Written by Youngwan Lee
# --------------------------------------------------------------------------------

"""
Detection Training Script for MPViT.
"""

import os
import itertools

import torch

from typing import Any, Dict, List, Set

from detectron2.data import build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping

from ditod import add_vit_config
from ditod import DetrDatasetMapper

from detectron2.data.datasets import register_coco_instances
import logging
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.engine.defaults import create_ddp_model
import weakref
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from ditod import MyDetectionCheckpointer, ICDAREvaluator
from ditod import MyTrainer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    """
    register publaynet first
    """
    register_coco_instances(
        "doclaynet_train",
        {},
        "../data_prepare/doclaynet_data/COCO/train_document-type.json",
        "../data_prepare/doclaynet_data/PNG"
    )

    register_coco_instances(
        "doclaynet_test",
        {},
        "../data_prepare/doclaynet_data/COCO/test_document-type.json",
        "../data_prepare/doclaynet_data/PNG"
    )

    register_coco_instances(
        "m6doc_train",
        {},
        "../data_prepare/m6doc_data/COCO/instances_train2017_document-type.json",
        "../data_prepare/m6doc_data/train2017"
    )

    register_coco_instances(
        "m6doc_test",
        {},
        "../data_prepare/m6doc_data/COCO/instances_test2017_document-type.json",
        "../data_prepare/m6doc_data/test2017"
    )

    register_coco_instances(
        "publaynet-doc_mini_train",
        {},
        "../data_prepare/publaynet_data/train_mini_label-style.json",
        "../data_prepare/publaynet_data/train"
    )

    register_coco_instances(
        "publaynet-doc_val",
        {},
        "../data_prepare/publaynet_data/val_mini_label-style.json",
        "../data_prepare/publaynet_data/val"
    )

    register_coco_instances(
        "doclaynet-pub_train",
        {},
        "../data_prepare/doclaynet_data/COCO/train_label-style.json",
        "../data_prepare/doclaynet_data/PNG"
    )

    register_coco_instances(
        "doclaynet-pub_test",
        {},
        "../data_prepare/doclaynet_data/COCO/val_label-style.json",
        "../data_prepare/doclaynet_data/PNG"
    )

    cfg = setup(args)

    if args.eval_only:
        model = MyTrainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = MyTrainer.test(cfg, model)
        return res

    trainer = MyTrainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    # args = parser.parse_args(
    #     ["--config-file", "config/doclaynet_configs/cascade/cascade_dit_base.yaml", "--num-gpus", "1",
    #      "MODEL.WEIGHTS", "./finetune/doclaynet_clip_prompt_linear-project_one-weight/model_final.pth", "OUTPUT_DIR",
    #      "./finetuning/"])
    args = parser.parse_args()
    print("Command Line Args:", args)

    if args.debug:
        import debugpy

        print("Enabling attach starts.")
        debugpy.listen(address=('0.0.0.0', 9310))
        debugpy.wait_for_client()
        print("Enabling attach ends.")

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url="tcp://127.0.0.1:12345",
        args=(args,),
    )
