# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import subprocess
import tqdm
import cv2
import glob
from typing import List, Tuple
from multiprocessing import Pool


def get_video_fps(video_path: str):
    cam = cv2.VideoCapture(video_path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    cam.release()
    return fps


def resample_fps_hz(video_input, video_output):
    os.makedirs(os.path.dirname(video_output), exist_ok=True)
    fps = get_video_fps(video_input)
    if fps == 25:
        shutil.copy(video_input, video_output)
    elif fps >= 24:
        command = f"ffmpeg -loglevel error -y -i {video_input} -r 25 -crf 18 -ar 16000 -q:a 0 {video_output}"
        subprocess.run(command, shell=True)


def multi_run_wrapper(args):
    return resample_fps_hz(*args)


def resample_fps_hz_multiprocessing(paths, num_workers):
    print(f"Resampling FPS and Hz of {len(paths)} videos ...")
    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(multi_run_wrapper, paths), total=len(paths)):
            pass
