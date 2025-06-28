import warnings
warnings.simplefilter("ignore", UserWarning)

import os
import json
import base64
import argparse
import re
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

IMAGE_SIZE_STR = "The width of the image is {width}px and the height is {height}px. "

PLAIN_TEMPLATE = "[{top}, {left}, {width}, {height}, \"{text}\"]"
PLAIN_TEMPLATE_IDX = "[{idx}, [{top}, {left}, {width}, {height}, \"{text}\"]]"
PLAIN_TEMPLATE_IDX_NO_TEXT = "[{idx}, [{top}, {left}, {width}, {height}]]"

# ---------- Prompt wrapping ----------

def _get_question_prompt_plain(
    question: str,
    bbox_all: List[Tuple[int, int, int, int, str]],
    image_width: int,
    image_height: int,
    test_version: str,
) -> str:
    """Construct the plain-text question prompt according to test version."""

    img_size_prompt = IMAGE_SIZE_STR.format(width=image_width, height=image_height)

    if test_version == "1":
        prompt = (
            question
            + "\nPlease first answer the question according to the given image and then generate the grounded text bounding boxes that support your answer. "
            + img_size_prompt
            + "Please generate the bounding boxes in the below format, the coordinates should be integers:\n"
            + PLAIN_TEMPLATE
        )
        return prompt

    if test_version == "2":
        pair_text_all = []
        for idx, (top, left, width, height, text) in enumerate(bbox_all, 1):
            pair_text_all.append(
                PLAIN_TEMPLATE_IDX.format(idx=idx, top=top, left=left, width=width, height=height, text=text)
            )
        bbox_str = ", ".join(pair_text_all)

        prompt = (
            question
            + "\nPlease first answer the question according to the given image and then select the grounded bounding boxes that support your answer. "
            + img_size_prompt
            + "All the bounding boxes are provided below in the below format:\n"
            + bbox_str
        )
        return prompt

    pair_boxes_all = []
    for idx, (top, left, width, height, _) in enumerate(bbox_all, 1):
        pair_boxes_all.append(
            PLAIN_TEMPLATE_IDX_NO_TEXT.format(idx=idx, top=top, left=left, width=width, height=height)
        )
    bbox_str = ", ".join(pair_boxes_all)

    prompt = (
        question
        + "\nPlease first answer the question according to the given image and then select the grounded bounding boxes that support your answer. "
        + img_size_prompt
        + "All the bounding boxes are provided below in the below format:\n"
        + bbox_str
    )
    return prompt


def get_question_prompt(
    question: str,
    bbox_all: List[Tuple[int, int, int, int, str]],
    image_path: str,
    test_version: str,
) -> str:
    from PIL import Image

    with Image.open(image_path) as img:
        width, height = img.size

    return _get_question_prompt_plain(question, bbox_all, width, height, test_version)


# ---------- Bounding-box extraction helpers ----------

def _extract_from_plain_v1(text: str):
    """Extract boxes for generation variant (v1)."""
    pattern = r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*\"([^\"]+)\"\s*\]"
    matches = re.findall(pattern, text)
    return [(int(top), int(left), int(width), int(height)) for top, left, width, height, _ in matches]


def _extract_from_plain_v2(text: str):
    """Extract boxes for selection variant (v2)."""
    pattern = r"\[\s*\d+,\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*\"[^\"]+\"\s*\]\s*\]"
    matches = re.findall(pattern, text)
    return [(int(top), int(left), int(width), int(height)) for top, left, width, height in matches]


def _extract_from_plain_v3(text: str):
    """Extract boxes for selection variant without text (v3)."""
    pattern = r"\[\s*\d+,\s*\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]\s*\]"
    matches = re.findall(pattern, text)
    return [(int(top), int(left), int(width), int(height)) for top, left, width, height in matches]


# ---------- Metric computation ----------

def _calculate_iou(pred_boxes, gt_boxes):
    """IoU of sets (Jaccard index)."""
    intersection = len(set(pred_boxes).intersection(set(gt_boxes)))
    union = len(set(pred_boxes).union(set(gt_boxes)))
    return intersection / union if union else 0.0


def _calculate_precision(pred_boxes, gt_boxes):
    inter = len(set(pred_boxes).intersection(set(gt_boxes)))
    return inter / len(pred_boxes) if pred_boxes else 0.0


def _calculate_recall(pred_boxes, gt_boxes):
    inter = len(set(pred_boxes).intersection(set(gt_boxes)))
    return inter / len(gt_boxes) if gt_boxes else 0.0


def _calculate_f1(precision: float, recall: float):
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def evaluate(model_outputs: str, grounding_gt, test_version: str):
    """Evaluate model output against ground truth."""
    gt_boxes = [(tp[0], tp[1], tp[2], tp[3]) for tp in grounding_gt]

    if test_version == "1":
        pred_boxes = _extract_from_plain_v1(model_outputs)
        iou = _calculate_iou(pred_boxes, gt_boxes)
        return {"iou": iou}, [iou]

    if test_version == "2":
        pred_boxes = _extract_from_plain_v2(model_outputs)
    elif test_version == "3":
        pred_boxes = _extract_from_plain_v3(model_outputs)
    else:
        raise ValueError(f"Unsupported test_version: {test_version}")

    iou = _calculate_iou(pred_boxes, gt_boxes)
    precision = _calculate_precision(pred_boxes, gt_boxes)
    recall = _calculate_recall(pred_boxes, gt_boxes)
    f1 = _calculate_f1(precision, recall)
    return {"iou": iou, "precision": precision, "recall": recall, "f1": f1}, [iou, precision, recall, f1]


def call_gpt(image_path: str, question_prompt: str, model_name: str = "gpt-4o") -> str:
    """Call the GPT-4o vision model.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    question_prompt : str
        The text prompt (already wrapped by `get_question_prompt`).
    model_name : str, optional
        Name of the model to use. Defaults to "gpt-4o".

    Returns
    -------
    str
        The assistant's raw text response.
    """
    if openai.api_key is None:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()

    response = openai.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant in answering questions about a given image and finding bounding boxes that support the answer.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                ],
            },
        ],
    )
    return response.choices[0].message.content

# ---------- CLI ----------

def parse_args():
    parser = argparse.ArgumentParser(description="Run GPT-4o on grounding benchmark.")
    parser.add_argument("--testset", type=str, default="ChartQA", help="All, ChartQA, DocVQA, InfographicsVQA, TrinsQA")
    parser.add_argument("--end_idx", type=int, default=0, help="Only evaluate the first N samples if >0.")
    parser.add_argument("--save_name", type=str, default="try_results_gpt4o_simple.json")
    parser.add_argument("--test_version", type=str, default="1", choices=["1", "2", "3"], help="Prompt variant (1: generation, 2: selection with text, 3: selection without text).")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="OpenAI model name.")
    return parser.parse_args()

# ---------- Main ----------

def main():
    args = parse_args()

    if args.testset == "All":
        testsets: List[str] = ["ChartQA", "DocVQA", "InfographicsVQA", "TrinsQA"]
    else:
        testsets = [args.testset]

    wrap_format = "plain"
    all_results_to_save = [{"test_version": args.test_version, "wrap_format": wrap_format, "model": args.model_name}]

    for testset in testsets:
        meta_path = os.path.join("benchmark", testset, "meta.json")
        with open(meta_path, "r") as f:
            meta_data = json.load(f)

        per_sample_metrics = []
        details_to_save = []

        meta_data_to_run = meta_data[: args.end_idx] if args.end_idx > 0 else meta_data
        total_samples = len(meta_data_to_run)

        for i, sample in tqdm(enumerate(meta_data_to_run), total=total_samples, desc=f"{testset}"):
            image_path = os.path.join("benchmark", testset, sample["image_path"])
            question = sample["question"]
            answers = sample["answers"]
            bbox_all = sample["bbox_all"]
            grounding_gt = sample["grounding"]

            prompt = get_question_prompt(question, bbox_all, image_path, args.test_version)

            try:
                model_output = call_gpt(image_path, prompt, model_name=args.model_name)
            except Exception as e:
                print(f"[Warning] GPT call failed on sample {i} (testset {testset}): {e}")
                model_output = ""

            metrics_dict, metrics_list = evaluate(model_output, grounding_gt, args.test_version)
            per_sample_metrics.append(metrics_list)

            sample_save = {
                **sample,
                "question_prompt": prompt,
                "model_output": model_output,
                "metrics": metrics_dict,
            }
            details_to_save.append(sample_save)

        metrics_array = np.array(per_sample_metrics)
        avg_metrics = metrics_array.mean(axis=0) if len(metrics_array) > 0 else np.zeros(metrics_array.shape[1])

        if args.test_version == "1":
            final_res = {"iou": float(avg_metrics[0])}
        else:
            final_res = {
                "iou": float(avg_metrics[0]),
                "precision": float(avg_metrics[1]),
                "recall": float(avg_metrics[2]),
                "f1": float(avg_metrics[3]),
            }

        print("\n===", testset, "===")
        for k, v in final_res.items():
            print(f"{k}: {v:.5f}")

        all_results_to_save.append({testset: {"final_results": final_res, "results": details_to_save}})

    with open(args.save_name, "w") as f:
        json.dump(all_results_to_save, f, indent=4)
    print(f"Results written to {args.save_name}")


if __name__ == "__main__":
    main() 