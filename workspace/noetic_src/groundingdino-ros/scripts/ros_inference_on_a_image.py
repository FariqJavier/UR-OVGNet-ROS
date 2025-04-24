#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import os
import sys

from PIL import Image as PILImage, ImageDraw, ImageFont
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_PATH = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'src'))
GROUNDINGDINO_PATH = os.path.abspath(os.path.join(SRC_PATH, '..', 'groundingdino'))
sys.path.insert(0, GROUNDINGDINO_PATH)

import torch
import datasets.transforms as T
from models import build_model
from util.slconfig import SLConfig
from util.utils import clean_state_dict, get_phrases_from_posmap
from util.vl_utils import create_positive_map_from_span

def load_image(image_path):
    # load image
    # image_pil = img.convert("RGB")  # load image
    image_pil = PILImage.open(image_path).convert("RGB")  # load image

    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image


def load_model(model_config_path, model_checkpoint_path, cpu_only=False):
    args = SLConfig.fromfile(model_config_path)
    args.device = "cuda" if not cpu_only else "cpu"
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    print(load_res)
    _ = model.eval()
    return model


def get_grounding_output(model, image, caption, box_threshold, text_threshold=None, with_logits=True, cpu_only=False, token_spans=None):
    assert text_threshold is not None or token_spans is not None, "text_threshould and token_spans should not be None at the same time!"
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    device = "cuda" if not cpu_only else "cpu"
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"][0]  # (nq, 4)

    # filter output
    if token_spans is None:
        logits_filt = logits.cpu().clone()
        boxes_filt = boxes.cpu().clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4 #

        # get phrase
        tokenlizer = model.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        max_logits = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
            if with_logits:
                a = logit.max().item()
                max_logits.append(a)
            else:
                pred_phrases.append(pred_phrase)
        max_logit = max(max_logits)
        index = max_logits.index(max_logit)
        boxes_filt = boxes_filt[index]


    else:
        # given-phrase mode
        positive_maps = create_positive_map_from_span(
            model.tokenizer(text_prompt),
            token_span=token_spans
        ).to(image.device) # n_phrase, 256

        logits_for_phrases = positive_maps @ logits.T # n_phrase, nq
        all_logits = []
        all_phrases = []
        all_boxes = []
        for (token_span, logit_phr) in zip(token_spans, logits_for_phrases):
            # get phrase
            phrase = ' '.join([caption[_s:_e] for (_s, _e) in token_span])
            # get mask
            filt_mask = logit_phr > box_threshold
            # filt box
            all_boxes.append(boxes[filt_mask])
            # filt logits
            all_logits.append(logit_phr[filt_mask])
            if with_logits:
                logit_phr_num = logit_phr[filt_mask]
                all_phrases.extend([phrase + f"({str(logit.item())[:4]})" for logit in logit_phr_num])
            else:
                all_phrases.extend([phrase for _ in range(len(filt_mask))])
        boxes_filt = torch.cat(all_boxes, dim=0).cpu()
        pred_phrases = all_phrases


    return boxes_filt, pred_phrases

def plot_boxes_to_image(image_pil, tgt):
    H, W = tgt["size"]
    boxes = tgt["boxes"]
    labels = tgt["labels"]
    assert len(boxes) == len(labels), "boxes and labels must have same length"

    draw = ImageDraw.Draw(image_pil)
    mask = Image.new("L", image_pil.size, 0)
    mask_draw = ImageDraw.Draw(mask)

    # draw boxes and masks
    for box, label in zip(boxes, labels):
        # from 0..1 to 0..W, 0..H
        box = box * torch.Tensor([W, H, W, H])
        # from xywh to xyxy
        box[:2] -= box[2:] / 2
        box[2:] += box[:2]
        # random color
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        # draw
        x0, y0, x1, y1 = box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

        draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
        # draw.text((x0, y0), str(label), fill=color)

        font = ImageFont.load_default()
        if hasattr(font, "getbbox"):
            bbox = draw.textbbox((x0, y0), str(label), font)
        else:
            w, h = draw.textsize(str(label), font)
            bbox = (x0, y0, w + x0, y0 + h)
        # bbox = draw.textbbox((x0, y0), str(label))
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0), str(label), fill="white")

        mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=6)

    return image_pil, mask

class GroundingDinoROSNode:
    def __init__(self):
        rospy.init_node("grounding_dino_node")

        self.image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw")
        self.text_prompt = rospy.get_param("~text_prompt", "a bottle")
        self.config_path = rospy.get_param("~config_path")
        self.checkpoint_path = rospy.get_param("~checkpoint_path")
        self.output_dir = rospy.get_param("~output_dir", "/tmp/dino_ros_output")
        os.makedirs(self.output_dir, exist_ok=True)

        self.bridge = CvBridge()

        # Load GroundingDINO model
        self.model = load_model(self.config_path, self.checkpoint_path)

        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1)
        rospy.loginfo("GroundingDINO node ready. Waiting for image...")

    def image_callback(self, msg):
        try:
            # Convert to OpenCV and then PIL
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            image_pil = PILImage.fromarray(cv_image[..., ::-1])  # Convert BGR to RGB

            # Save temp image for reuse
            temp_path = os.path.join(self.output_dir, "input.png")
            image_pil.save(temp_path)

            # Run model (reuse existing code)
            image_pil_loaded, image_tensor = load_image(temp_path)
            boxes, labels = get_grounding_output(
                self.model, image_tensor, self.text_prompt,
                box_threshold=0.3, text_threshold=0.25, cpu_only=True
            )

            # Draw results
            output_image, mask = plot_boxes_to_image(image_pil_loaded, {
                "boxes": boxes,
                "labels": labels,
                "size": [image_pil.height, image_pil.width]
            })
            output_path = os.path.join(self.output_dir, "result.jpg")
            output_image.save(output_path)
            rospy.loginfo(f"Inference complete. Result saved to {output_path}")

        except Exception as e:
            rospy.logerr(f"Error during inference: {e}")

class GroundingDinoImageFileNode:
    def __init__(self):
        rospy.init_node("grounding_dino_file_node")

        self.image_path = rospy.get_param("~image_path")
        self.text_prompt = rospy.get_param("~text_prompt", "a bottle")
        self.config_path = rospy.get_param("~config_path")
        self.checkpoint_path = rospy.get_param("~checkpoint_path")
        self.box_threshold = rospy.get_param("~box_threshold", 0.3)
        self.text_threshold = rospy.get_param("~text_threshold", 0.25)
        self.token_spans = rospy.get_param("~token_spans", None)
        self.cpu_only = rospy.get_param("~cpu-only", False)
        self.output_dir = rospy.get_param("~output_dir", "/tmp/dino_ros_output")
        os.makedirs(self.output_dir, exist_ok=True)

        # Load GroundingDINO model
        self.model = load_model(self.config_path, self.checkpoint_path, self.cpu_only)

        self.process_image_file()

    def process_image_file(self):
        try:
            if not os.path.isfile(self.image_path):
                raise FileNotFoundError(f"Image file '{self.image_path}' not found.")

            # Load the image using PIL
            image_pil = PILImage.open(self.image_path)

            # Save a temporary copy (optional)
            temp_path = os.path.join(self.output_dir, "input_groundingdino.png")
            image_pil.save(temp_path)

            if self.token_spans is not None:
                self.text_threshold = None
                print("Using token_spans. Set the text_threshold to None.")

            # Run model (reuse existing code)
            image_pil_loaded, image_tensor = load_image(temp_path)
            
            boxes, labels = get_grounding_output(
                self.model, image_tensor, self.text_prompt,
                self.box_threshold, self.text_threshold, cpu_only=self.cpu_only, token_spans=eval(f"{self.token_spans}")
            )

            # Draw results
            size = image_pil.size
            output_image, mask = plot_boxes_to_image(image_pil_loaded, {
                "boxes": boxes,
                "labels": labels,
                "size": [size[1], size[0]],  # H,W
            })
            output_path = os.path.join(self.output_dir, "result_groundingdino.jpg")
            image_with_box = plot_boxes_to_image(image_pil, output_image)[0]
            image_with_box.save(output_path)
            rospy.loginfo(f"Inference complete. Result saved to {output_path}")

        except Exception as e:
            rospy.logerr(f"Error during inference: {e}")

if __name__ == "__main__":
    try:
        # GroundingDinoROSNode()
        # rospy.spin()
        GroundingDinoImageFileNode()
    except rospy.ROSInterruptException:
        pass
