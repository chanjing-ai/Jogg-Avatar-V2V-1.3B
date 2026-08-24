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
import subprocess
import tqdm
from multiprocessing import Pool
import json
import math


def segment_video(_, video_input):
    if not os.path.exists(video_input):
        return
    video_duration = get_video_duration(video_input)
    if video_duration is None:
        os.remove(video_input)
        return
    if video_duration < 9:
        return
    video_basename = video_input[:-4]
    video_output = os.path.join(os.path.dirname(video_input), f"{video_basename}_%03d.mp4")
    os.makedirs(os.path.dirname(video_output), exist_ok=True)
    command = f"ffmpeg -loglevel error -y -i {video_input} -map 0 -c:v copy -c:a copy -segment_time 6 -f segment -reset_timestamps 1 {video_output}"
    subprocess.run(command, shell=True)

    first_segment_file = os.path.join(os.path.dirname(video_input), f"{video_basename}_000.mp4")
    if not os.path.exists(first_segment_file):
        # 没有生成任何分段，视为失败，保留原视频
        return
    else:
        os.remove(video_input)
        return



def get_video_duration(video_path):
    """
    使用ffprobe获取视频精确时长（毫秒级）

    Args:
        video_path: 视频文件路径

    Returns:
        float: 视频时长（秒）
    """
    try:
        # 使用ffprobe获取视频时长，精确到毫秒
        command = [
            'ffprobe',
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'json',
            video_path
        ]

        result = subprocess.run(command, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        duration = float(data['format']['duration'])

        return duration
    except Exception as e:
        print(f"获取视频时长失败 {video_path}: {str(e)}")
        return None



def multi_run_wrapper(args):
    return segment_video(*args)


def segment_videos_multiprocessing(paths, num_workers):

    with Pool(num_workers) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(multi_run_wrapper, paths), total=len(paths)):
            pass




# ffmpeg -ss 00:05:00 -i input.mp4 -t 00:02:00 -c copy -reset_timestamps 1 segment1.mp4
