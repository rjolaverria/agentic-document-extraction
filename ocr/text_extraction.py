from collections import defaultdict
from dataclasses import dataclass

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from paddleocr import PaddleOCR  # type: ignore[import-untyped]
from transformers import LayoutLMv3ForTokenClassification
from typing import Any

CLS_TOKEN_ID = 0
UNK_TOKEN_ID = 3
EOS_TOKEN_ID = 2
TMP_DIR = "tmp/"

ocr = PaddleOCR(lang="en")
layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained(
    "hantian/layoutreader"
)


@dataclass
class OCRTextRegion:
    text: str
    poly: list[np.ndarray]
    confidence: float

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        """[x1, y1, x2, y2] bounding box"""
        x_coords = [p[0] for p in self.poly]
        y_coords = [p[1] for p in self.poly]
        return (min(x_coords), min(y_coords), max(x_coords), max(y_coords))


def _store_aligned_bboxes(request_id: str, processed_img, boxes):
    """
    Store OCR processed an image with aligned bounding boxes overlaid for visualization.
    """
    img_plot = processed_img.copy()
    for box in boxes:
        pts = np.array(box, dtype=int)
        cv2.polylines(img_plot, [pts], True, (0, 255, 0), 2)

    plt.figure(figsize=(8, 10))
    plt.imshow(cv2.cvtColor(img_plot, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    path_to_file = f"{TMP_DIR}/{request_id}"
    os.makedirs(path_to_file, exist_ok=True)

    cv2.imwrite(f"{path_to_file}/processed_img.png", processed_img)
    plt.savefig(f"{path_to_file}/bboxes.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def _store_reading_order(request_id: str, ocr_regions, processed_img, reading_order):
    """
    Store image with reading order overlaid for visualization.
    """

    fig, ax = plt.subplots(1, figsize=(10, 14))
    ax.imshow(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))

    order_map = {i: order for i, order in enumerate(reading_order)}

    for i, region in enumerate(ocr_regions):
        bbox = region.bbox
        if bbox and len(bbox) >= 4:
            x1, y1, x2, y2 = bbox
            polygon_points = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            ax.add_patch(
                patches.Polygon(
                    polygon_points,
                    linewidth=2,
                    edgecolor="blue",
                    facecolor="none",
                    alpha=0.7,
                )
            )
            # Calculate center for text
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            ax.text(
                center_x,
                center_y,
                str(order_map.get(i, i)),
                fontsize=13,
                color="red",
                ha="center",
                va="center",
                fontweight="bold",
            )

    ax.axis("off")
    plt.tight_layout()

    path_to_file = f"{TMP_DIR}/{request_id}"
    os.makedirs(path_to_file, exist_ok=True)
    plt.savefig(f"{path_to_file}/reading_order.png", bbox_inches="tight", pad_inches=0)
    plt.close()


def _extract_text(request_id: str, image_path: str) -> tuple[list[OCRTextRegion], Any]:
    result = ocr.predict(image_path)
    page = result[0]
    texts = page["rec_texts"]
    scores = page["rec_scores"]
    polys = page["rec_polys"]
    processed_img = page["doc_preprocessor_res"]["output_img"]

    # storing for debugging/visualization
    _store_aligned_bboxes(request_id, processed_img, polys)

    ocr_regions: list[OCRTextRegion] = []
    for text, score, poly in zip(texts, scores, polys):
        assert isinstance(poly, np.ndarray)
        ocr_regions.append(
            OCRTextRegion(text=text, poly=poly.astype(int).tolist(), confidence=score)
        )

    return (ocr_regions, processed_img)


def _boxes2inputs(boxes: list[list[int]]) -> dict[str, torch.Tensor]:
    bbox = [[0, 0, 0, 0]] + boxes + [[0, 0, 0, 0]]
    input_ids = [CLS_TOKEN_ID] + [UNK_TOKEN_ID] * len(boxes) + [EOS_TOKEN_ID]
    attention_mask = [1] + [1] * len(boxes) + [1]
    return {
        "bbox": torch.tensor([bbox]),
        "attention_mask": torch.tensor([attention_mask]),
        "input_ids": torch.tensor([input_ids]),
    }


def _prepare_inputs(
    inputs: dict[str, torch.Tensor], model: LayoutLMv3ForTokenClassification
) -> dict[str, torch.Tensor]:
    ret = {}
    for k, v in inputs.items():
        v = v.to(model.device)
        if torch.is_floating_point(v):
            v = v.to(model.dtype)
        ret[k] = v
    return ret


def _parse_logits(logits: torch.Tensor, length: int) -> list[int]:
    """
    parse logits to orders

    :param logits: logits from model
    :param length: input length
    :return: orders
    """
    logits = logits[1 : length + 1, :length]
    orders = logits.argsort(descending=False).tolist()
    ret = [o.pop() for o in orders]
    while True:
        order_to_idxes = defaultdict(list)
        for idx, order in enumerate(ret):
            order_to_idxes[order].append(idx)
        # filter idxes len > 1
        order_to_idxes = defaultdict(
            list, {k: v for k, v in order_to_idxes.items() if len(v) > 1}
        )  # type: ignore[arg-type]
        if not order_to_idxes:
            break
        # filter
        for order, idxes in order_to_idxes.items():
            # find original logits of idxes
            idxes_to_logit = {}
            for idx in idxes:
                idxes_to_logit[idx] = logits[idx, order]
            sorted_idxes_to_logit = sorted(
                idxes_to_logit.items(), key=lambda x: x[1], reverse=True
            )
            # keep the highest logit as order, set others to next candidate
            for idx, _ in sorted_idxes_to_logit[1:]:
                ret[idx] = orders[idx].pop()

    return ret


def _get_reading_order(
    request_id: str, ocr_regions: list[OCRTextRegion], processed_img
) -> list[int]:
    max_x = max_y = 0
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox
        max_x = max(max_x, x2)
        max_y = max(max_y, y2)

    image_width = max_x * 1.1  # Add 10% padding
    image_height = max_y * 1.1

    boxes = []
    for region in ocr_regions:
        x1, y1, x2, y2 = region.bbox
        left = int((x1 / image_width) * 1000)
        top = int((y1 / image_height) * 1000)
        right = int((x2 / image_width) * 1000)
        bottom = int((y2 / image_height) * 1000)
        boxes.append([left, top, right, bottom])

    inputs = _boxes2inputs(boxes)
    inputs = _prepare_inputs(inputs, layoutlm_model)

    logits = layoutlm_model(**inputs).logits.cpu().squeeze(0)

    reading_order = _parse_logits(logits, len(boxes))

    _store_reading_order(
        request_id=request_id,
        ocr_regions=ocr_regions,
        processed_img=processed_img,
        reading_order=reading_order,
    )

    return reading_order


def get_ordered_text(request_id: str, image_path: str) -> list[dict]:
    extracted_text, processed_img = _extract_text(request_id, image_path)
    reading_order = _get_reading_order(request_id, extracted_text, processed_img)

    indexed_regions = [
        (reading_order[i], i, extracted_text[i]) for i in range(len(extracted_text))
    ]

    indexed_regions.sort(key=lambda x: x[0])

    ordered_text = []
    for position, original_idx, region in indexed_regions:
        ordered_text.append(
            {
                "position": position,
                "text": region.text,
                "confidence": region.confidence,
                "bbox": region.bbox,
            }
        )

    return ordered_text
