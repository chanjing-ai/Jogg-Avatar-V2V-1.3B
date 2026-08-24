import cv2
import os
import glob
import logging
import numpy as np
from more_itertools import chunked
from typing import Iterator
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s")


def img_padding(img, target_width=512, target_height=512):
    try:
        new_image = cv2.resize(img, (target_width, target_height))

        # 图片填充至512x512 -------------------------------------------------
        # height, width, channels = img.shape
        #
        # # 计算需要在水平或垂直方向上添加黑色边界的像素值
        # horizontal_padding = (target_width - width) // 2
        # vertical_padding = (target_height - height) // 2
        #
        # new_image = np.zeros((target_height, target_width, channels),
        #                      dtype=np.uint8)
        #
        # new_image[vertical_padding:vertical_padding + height,
        #           horizontal_padding:horizontal_padding + width] = img
        # ----------------------------------------------------------------
    except Exception as e:
        logging.info(f"图片resize/padding失败：{e}")

    return new_image


def get_video_path(src_dir: str) -> list:
    video_extensions = ['*.mp4', '*.avi', '*.mkv', '*.mov']

    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(src_dir, ext)))

    return video_files


def video2frames(video_path: str) -> list:
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened()

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frames.append(frame)
    cap.release()

    return frames


def iter_frame(path: str) -> Iterator[np.ndarray]:
    cap = cv2.VideoCapture(path)
    assert cap.isOpened()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def iter_batch(path: str, batch_size: int) -> Iterator[np.ndarray]:
    for batch in chunked(tqdm(iter_frame(path)), batch_size):
        yield np.stack(batch)
