from typing import Tuple
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from mtcnn_pytorch.src.get_nets import PNet, RNet, ONet
from mtcnn_pytorch.src.box_utils import nms, calibrate_box, get_image_boxes, convert_to_square
from mtcnn_pytorch.src.first_stage import run_first_stage
from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face


class MTCNN():
    def __init__(self, device: str = 'cuda:0', crop_size: Tuple[int, int] = (112, 112)):

        assert device in ['cuda:0', 'cpu']
        self.device = torch.device(device)
        assert crop_size in [(112, 112), (96, 112)]
        self.crop_size = crop_size

        self.pnet = PNet().to(self.device)
        self.rnet = RNet().to(self.device)
        self.onet = ONet().to(self.device)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.refrence = get_reference_facial_points(default_square=crop_size[0] == crop_size[1])

        self.min_face_size = 20
        self.thresholds =  [0.6,0.7,0.9]
        self.nms_thresholds = [0.7, 0.7, 0.7]
        self.factor = 0.85


    def align(self, img):
        _, landmarks = self.detect_faces(img, self.min_face_size, self.thresholds, self.nms_thresholds, self.factor)
        if landmarks is None or len(landmarks) == 0:
            return None
        facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
        warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=self.crop_size)
        return Image.fromarray(warped_face)

    def align_multi(self, img, limit=None):
        boxes, landmarks = self.detect_faces(img, self.min_face_size, self.thresholds, self.nms_thresholds, self.factor)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=self.crop_size)
            faces.append(Image.fromarray(warped_face))
        return boxes, faces

    def align_batch(self, imgs):
        """
        Batched version of align(): detect + align the largest face in each
        of *imgs* with ONE forward pass through RNet and ONE through ONet
        covering every image's candidate boxes together, instead of one
        RNet/ONet pass per image (see detect_faces_batch). PNet still runs
        per-image since its scan scales depend on each image's own size.

        Returns a list aligned to imgs, each entry a PIL.Image or None if no
        face was found in that image.
        """
        results = self.detect_faces_batch(
            imgs, self.min_face_size, self.thresholds, self.nms_thresholds, self.factor
        )
        aligned = []
        for img, (boxes, landmarks) in zip(imgs, results):
            if landmarks is None or len(landmarks) == 0:
                aligned.append(None)
                continue
            facial5points = [[landmarks[0][j], landmarks[0][j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(np.array(img), facial5points, self.refrence, crop_size=self.crop_size)
            aligned.append(Image.fromarray(warped_face))
        return aligned

    def detect_faces_batch(self, images, min_face_size, thresholds, nms_thresholds, factor):
        """
        Batched version of detect_faces() for multiple images at once.

        Stage 1 (PNet) still runs per-image: its image pyramid scales are
        derived from each image's own size, so different images can't share
        one PNet batch without resizing/padding away that size information.

        Stages 2 (RNet) and 3 (ONet) DO batch across images: both operate on
        fixed-size crops (24x24 / 48x48) cut out of the candidate boxes
        regardless of the source image's size, so every image's candidate
        crops are concatenated into one tensor and run through RNet/ONet in
        a single forward pass, instead of one pass per image. Per-image
        filtering (score threshold, NMS, calibration) after each stage is
        unchanged — only the network call itself is batched.

        Returns a list of (boxes, landmarks) tuples, one per input image, in
        the same format detect_faces() returns for a single image.
        """
        if not images:
            return []

        empty = (np.empty((0, 5), dtype=np.float32), np.empty((0, 10), dtype=np.float32))

        with torch.no_grad():
            # ── STAGE 1 — per image ──────────────────────────────────────────
            # Wrapped per-image: one malformed/edge-case image (e.g. a crop
            # too small for the pyramid) must degrade to "no candidates" for
            # that image, not abort every other image sharing this batch —
            # matches the isolation the old per-image align() call got for
            # free from its own try/except in adaface.py.
            per_image_boxes = []
            for image in images:
                try:
                    width, height = image.size
                    min_length = min(height, width)
                    min_detection_size = 12

                    scales = []
                    m = min_detection_size / min_face_size
                    scaled_min_length = min_length * m
                    factor_count = 0
                    while scaled_min_length > min_detection_size:
                        scales.append(m * factor ** factor_count)
                        scaled_min_length *= factor
                        factor_count += 1

                    bounding_boxes = []
                    for s in scales:
                        boxes = run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
                        bounding_boxes.append(boxes)
                    bounding_boxes = [b for b in bounding_boxes if b is not None]

                    if len(bounding_boxes) == 0:
                        per_image_boxes.append(np.empty((0, 5), dtype=np.float32))
                        continue

                    bounding_boxes = np.vstack(bounding_boxes)
                    keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
                    bounding_boxes = bounding_boxes[keep]
                    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
                    bounding_boxes = convert_to_square(bounding_boxes)
                    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
                    per_image_boxes.append(bounding_boxes)
                except Exception:
                    per_image_boxes.append(np.empty((0, 5), dtype=np.float32))

            if sum(len(b) for b in per_image_boxes) == 0:
                return [empty for _ in images]

            # ── STAGE 2 — RNet, batched across all images ────────────────────
            batch_crops = []
            for image, boxes in zip(images, per_image_boxes):
                if len(boxes) > 0:
                    batch_crops.append(get_image_boxes(boxes, image, size=24))
            stacked = torch.FloatTensor(np.concatenate(batch_crops, axis=0)).to(self.device)
            offsets_out, probs_out = self.rnet(stacked)
            offsets_all = offsets_out.cpu().data.numpy()
            probs_all   = probs_out.cpu().data.numpy()

            stage2_boxes = []
            pos = 0
            for boxes in per_image_boxes:
                k = len(boxes)
                if k == 0:
                    stage2_boxes.append(np.empty((0, 5), dtype=np.float32))
                    continue
                offsets = offsets_all[pos:pos + k]
                probs   = probs_all[pos:pos + k]
                pos += k

                try:
                    keep = np.where(probs[:, 1] > thresholds[1])[0]
                    b = boxes[keep]
                    if len(b) == 0:
                        stage2_boxes.append(np.empty((0, 5), dtype=np.float32))
                        continue
                    b[:, 4] = probs[keep, 1].reshape((-1,))
                    offs = offsets[keep]

                    keep2 = nms(b, nms_thresholds[1])
                    b = b[keep2]
                    b = calibrate_box(b, offs[keep2])
                    b = convert_to_square(b)
                    b[:, 0:4] = np.round(b[:, 0:4])
                    stage2_boxes.append(b)
                except Exception:
                    stage2_boxes.append(np.empty((0, 5), dtype=np.float32))

            if sum(len(b) for b in stage2_boxes) == 0:
                return [empty for _ in images]

            # ── STAGE 3 — ONet, batched across all images ────────────────────
            batch_crops2 = []
            for image, boxes in zip(images, stage2_boxes):
                if len(boxes) > 0:
                    batch_crops2.append(get_image_boxes(boxes, image, size=48))
            stacked2 = torch.FloatTensor(np.concatenate(batch_crops2, axis=0)).to(self.device)
            landmarks_out, offsets_out2, probs_out2 = self.onet(stacked2)
            landmarks_all = landmarks_out.cpu().data.numpy()
            offsets_all2  = offsets_out2.cpu().data.numpy()
            probs_all2    = probs_out2.cpu().data.numpy()

            results = []
            pos = 0
            for boxes in stage2_boxes:
                k = len(boxes)
                if k == 0:
                    results.append(empty)
                    continue
                landmarks = landmarks_all[pos:pos + k]
                offsets   = offsets_all2[pos:pos + k]
                probs     = probs_all2[pos:pos + k]
                pos += k

                try:
                    keep = np.where(probs[:, 1] > thresholds[2])[0]
                    b  = boxes[keep]
                    if len(b) == 0:
                        results.append(empty)
                        continue
                    b[:, 4] = probs[keep, 1].reshape((-1,))
                    offs = offsets[keep]
                    lm   = landmarks[keep]

                    bw = b[:, 2] - b[:, 0] + 1.0
                    bh = b[:, 3] - b[:, 1] + 1.0
                    xmin, ymin = b[:, 0], b[:, 1]
                    lm[:, 0:5]  = np.expand_dims(xmin, 1) + np.expand_dims(bw, 1) * lm[:, 0:5]
                    lm[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(bh, 1) * lm[:, 5:10]

                    b = calibrate_box(b, offs)
                    keep3 = nms(b, nms_thresholds[2], mode='min')
                    b  = b[keep3]
                    lm = lm[keep3]

                    results.append((b, lm))
                except Exception:
                    results.append(empty)

        return results

    def detect_faces(self, image, min_face_size, thresholds, nms_thresholds, factor):
        """
        Arguments:
            image: an instance of PIL.Image.
            min_face_size: a float number.
            thresholds: a list of length 3.
            nms_thresholds: a list of length 3.

        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # BUILD AN IMAGE PYRAMID
        width, height = image.size
        min_length = min(height, width)

        min_detection_size = 12
        # factor = 0.707  # sqrt(0.5)

        # scales for scaling the image
        scales = []

        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor**factor_count)
            min_length *= factor
            factor_count += 1

        # STAGE 1

        # it will be returned
        bounding_boxes = []

        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
                bounding_boxes.append(boxes)

            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            if len(bounding_boxes) == 0:
                return [], []
            bounding_boxes = np.vstack(bounding_boxes)

            keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
            bounding_boxes = bounding_boxes[keep]

            # use offsets predicted by pnet to transform bounding boxes
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            # shape [n_boxes, 5]

            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 2

            img_boxes = get_image_boxes(bounding_boxes, image, size=24)
            img_boxes = torch.FloatTensor(img_boxes).to(self.device)

            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
            offsets = offsets[keep]

            keep = nms(bounding_boxes, nms_thresholds[1])
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # STAGE 3

            img_boxes = get_image_boxes(bounding_boxes, image, size=48)
            if len(img_boxes) == 0:
                return [], []
            img_boxes = torch.FloatTensor(img_boxes).to(self.device)
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1, ))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1) * landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1) * landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]

        return bounding_boxes, landmarks
