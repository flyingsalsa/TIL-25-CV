import base64
import itertools
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

load_dotenv()

BATCH_SIZE = 16


class COCOPatched(COCO):
    def __init__(self, annotations):
        # The varnames here are disgusting, but they're used by other
        # non-overridden methods so don't touch them.
        self.dataset, self.anns, self.cats, self.imgs = {}, {}, {}, {}
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)

        assert type(annotations) == dict, \
            f"Annotation format {type(annotations)} not supported"
        print("Annotations loaded.")
        self.dataset = annotations
        self.createIndex()


def sample_generator(
        instances: Sequence[Mapping[str, Any]],
        data_dir: Path,
) -> Iterator[Mapping[str, Any]]:
    for instance in instances:
        with open(data_dir / "images" / instance["file_name"], "rb") as img_file:
            img_data = img_file.read()
            yield {
                "key": instance["id"],
                "b64": base64.b64encode(img_data).decode("ascii"),
            }


def score_cv(preds: Sequence[Mapping[str, Any]], ground_truth: Any) -> float:
    if not preds:
        return 0.

    ground_truth = COCOPatched(ground_truth)
    results = ground_truth.loadRes(preds)
    coco_eval = COCOeval(ground_truth, results, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats[0].item()


def main():
    data_dir = Path(f"data/cv")
    results_dir = Path(f"results")
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "annotations.json", "r") as f:
        annotations = json.load(f)
    instances = annotations["images"]

    batch_generator = itertools.batched(
        sample_generator(instances, data_dir), n=BATCH_SIZE)

    results = []
    for batch in tqdm(batch_generator, total=math.ceil(len(instances) / BATCH_SIZE)):
        response = requests.post("http://localhost:5002/cv", data=json.dumps({
            "instances": batch,
        }))

        batch_preds = response.json()["predictions"]
        for instance, single_image_detections in zip(batch, batch_preds):
            for detection in single_image_detections:
                results.append({
                    "image_id": instance["key"],
                    "score": 1.,
                    "bbox": detection["bbox"],
                    "category_id": detection["category_id"],
                })

    results_path = results_dir / \
        f"cv_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json"
    print(f"Saving test results to {str(results_path)}")
    with open(results_path, "w") as results_file:
        json.dump(results, results_file)

    mean_ap = score_cv(results, annotations)
    print("mAP@.5:.05:.95:", mean_ap)


if __name__ == "__main__":
    main()
