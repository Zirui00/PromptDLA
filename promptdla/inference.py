import argparse

import cv2

from ditod import add_vit_config

import torch
import numpy as np

from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, Visualizer, _create_text_labels, GenericMask
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultPredictor, default_argument_parser
import os


class MyVisualizer(Visualizer):
    def draw_instance_predictions(self, predictions, score_threshold=None):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes.tolist() if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if score_threshold != None:
            top_id = np.where(scores.numpy() > score_threshold)[0].tolist()
            scores = torch.tensor(scores.numpy()[top_id])
            boxes.tensor = torch.tensor(boxes.tensor.numpy()[top_id])
            classes = [classes[ii] for ii in top_id]
            labels = [labels[ii] for ii in top_id]

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks[top_id])
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            colors = [
                self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            ]
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.reset_image(
                self._create_grayscale_image(
                    (predictions.pred_masks.any(dim=0) > 0).numpy()
                    if predictions.has("pred_masks")
                    else None
                )
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output


def main():
    parser = default_argument_parser()

    args = parser.parse_args()

    # Step 1: instantiate config
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)

    # Step 2: add model weights URL to config
    cfg.merge_from_list(args.opts)

    # Step 3: set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.MODEL.DEVICE = device

    # Step 4: define model
    predictor = DefaultPredictor(cfg)

    # Step 5: run inference
    for file in os.listdir(args.image_path_dir):
        img = cv2.imread(args.image_path_dir + '/' + file)
        category = file.split('-')[0]
        md = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        md.set(thing_classes=["Caption", "Footnote", "Formula", "List-item", "Page-footer", "Page-header", "Picture",
                              "Section-header", "Table", "Text", "Title"])

        output = predictor(img, doc_category=category)["instances"]

        v = MyVisualizer(img[:, :, ::-1],
                         md,
                         scale=1.0,
                         instance_mode=ColorMode.SEGMENTATION)
        result = v.draw_instance_predictions(output.to("cpu"), score_threshold=0.3)
        result_image = result.get_image()[:, :, ::-1]

        # step 6: save
        cv2.imwrite(args.output_file_name + '/' + file[:-4] + '_infer.png', result_image)


if __name__ == '__main__':
    main()
