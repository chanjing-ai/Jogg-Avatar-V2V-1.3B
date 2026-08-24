# import os
import cv2
import logging
import numpy as np

from video_preprocess.scrfd_crop_face.files.scrfd import SCRFD

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s: %(message)s")


class ScrfdAPI():

    def __init__(self, model_path, provider='gpu'):
        self.detector = SCRFD(model_file=model_path)
        ctx_id = 0 if provider == 'gpu' else -1
        self.detector.prepare(ctx_id)
        self.image = None
        self.bboxes = None
        self.kpss = None
        self.bboxes_list = []

    def infer(self,
              image_in,
              nms_thresh=0.5,
              input_size=(640, 640)):

        # self.image = cv2.cvtColor(image_in, cv2.COLOR_BGR2RGB)

        self.bboxes, self.kpss = self.detector.detect(image_in,
                                                      thresh=nms_thresh,
                                                      input_size=input_size,
                                                      max_num=1)

        return self.bboxes, self.kpss

    def infer_batch(self,
                    image_in,
                    nms_thresh=0.5,
                    input_size=(640, 640)):

        self.bboxes, self.kpss = self.detector.detect_batch(
            image_in, thresh=nms_thresh, input_size=input_size, max_num=1)

        return self.bboxes, self.kpss

    def _extract_image_face(self, frame, wh=1, img_size=256):
        """
        Extracts the face region and keypoints from an image frame.

        Parameters:
        frame (numpy.ndarray): The input image frame.
        wh (int): Weighting factor for the face box calculation. Default is 1.
        img_size (int): The desired size of the output image. Default is 256.

        Returns:
        tuple: A tuple containing:
            - list: Coordinates of the cropped face box [Xmin, Ymin, Xmax, Ymax].
            - list: Original face box coordinates [x1, y1, x2, y2].
            - list: Keypoints of the face.
        """
        bboxes, kpss = self.infer(frame)
        if not len(bboxes):
            return [], []
        borderpad = 0

        x1, y1, x2, y2, _ = bboxes[0].astype(int)

        face_box = [int(x1), int(y1), int(x2), int(y2)]

        # Calculate the eye center and face box dimensions
        eye_y = (kpss[0][0][1] + kpss[0][1][1]) // 2
        xmin, w = x1, min(x2 - x1, y2 - y1)
        x_c = xmin + w / 2
        Xmin = int(x_c - w / wh * 0.72)
        Xmax = int(x_c + w / wh * 0.72)
        Ymin = int(eye_y - w / wh * 1.44 * 0.2)
        Ymax = int(eye_y + w / wh * 1.44 * 0.8)

        # # Handle border padding if necessary
        # if Xmin <= 0 or Ymin <= 0 or Xmax >= frame.shape[1] or Ymax >= frame.shape[0]:
        #     borderpad = int(np.max([np.max(frame.shape[:2]) * 0.2, 25]))
        #     frame = np.pad(frame,
        #                    ((borderpad, borderpad),
        #                     (borderpad, borderpad),
        #                     (0, 0)),
        #                    'constant',
        #                    constant_values=(0, 0))

        #     Xmin = Xmin + borderpad
        #     Xmax = Xmax + borderpad
        #     Ymin = Ymin + borderpad
        #     Ymax = Ymax + borderpad
        #     if Xmin <= 0 or Ymin <= 0 or Xmax >= frame.shape[1] or Ymax >= frame.shape[0]:
        #         print(f'Face pad crop fail')
        #         return [], [], [], 0

        #     kpss = kpss[0]
        #     for j in range(5):
        #         kpss[j] = [
        #             int(kpss[j][0] + borderpad),
        #             int(kpss[j][1] + borderpad)
        #         ]
        # else:
        #     kpss = kpss[0]
        #     for j in range(5):
        #         kpss[j] = [
        #             int(kpss[j][0]),
        #             int(kpss[j][1])
        #         ]

        if Xmax - Xmin < 180 or Ymax - Ymin < 180:
            return [], []
        # else:
        #     # Convert NumPy array to list and ensure all values are Python integers
        #     kpss_list = [
        #         [int(x) for x in kp]
        #         for kp in kpss.tolist()
        #     ]
        #     return [Xmin, Ymin, Xmax, Ymax], face_box, kpss_list, borderpad
        kpss = kpss[0]
        for j in range(5):
            kpss[j] = [
                int(kpss[j][0]),
                int(kpss[j][1])
            ]
        kpss_list = [
            [int(x) for x in kp]
            for kp in kpss.tolist()
        ]

        return [Xmin, Ymin, Xmax, Ymax], kpss_list