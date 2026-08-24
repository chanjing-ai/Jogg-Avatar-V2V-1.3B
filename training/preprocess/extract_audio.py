#!/usr/bin/env python3
"""
视频音频提取脚本
遍历指定目录下的所有视频文件，使用ffmpeg提取音频并保存为WAV格式
"""

import os
import subprocess
import argparse
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp

def extract_audio_from_video(video_path, output_path):
    """
    使用ffmpeg从视频中提取音频

    Args:
        video_path: 输入视频文件路径
        output_path: 输出音频文件路径
    """
    try:
        # 构建ffmpeg命令
        cmd = [
            'ffmpeg',
            '-hide_banner',
            '-loglevel', 'panic',
            '-i', str(video_path),
            '-ac', '1',  # 单声道
            '-ar', '16000',  # 采样率16kHz
            '-acodec', 'pcm_s16le',
            str(output_path)
        ]

        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode == 0:
            return True
        else:
            print(f"提取音频失败: {video_path}")
            print(f"错误信息: {result.stderr}")
            return False

    except Exception as e:
        print(f"处理文件 {video_path} 时发生错误: {str(e)}")
        return False


def extract_audio_from_videos(paths, proc_idx, data_root=None, aux_root=None):
    print(f"进程 {proc_idx} 开始处理 {len(paths)} 个视频")
    for path in tqdm(paths):
        if data_root is not None and aux_root is not None:
            rel = path.resolve().relative_to(data_root)
            output_path = (aux_root / rel).with_suffix('.wav')
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = path.with_suffix('.wav')
        if output_path.exists():
            continue
        extract_audio_from_video(path, output_path)


def extract_audio_process_videos(data_dir, num_procs, aux_dir=None):
    video_files = list(Path(data_dir).rglob("*.mp4")) + \
                  list(Path(data_dir).rglob("*.MP4")) + \
                  list(Path(data_dir).rglob("*.avi")) + \
                  list(Path(data_dir).rglob("*.mov")) + \
                  list(Path(data_dir).rglob("*.mkv"))

    print(f"找到 {len(video_files)} 个视频文件")
    video_chunks = [[] for _ in range(num_procs)]
    for i in range(len(video_files)):
        video_chunks[i % num_procs].append(video_files[i])
    processes = []
    data_root = Path(data_dir).expanduser().resolve()
    aux_root = Path(aux_dir).expanduser().resolve() if aux_dir else None
    for i in range(num_procs):
        video_chunk = video_chunks[i]
        p = mp.Process(
            target=extract_audio_from_videos,
            args=(video_chunk, i, data_root, aux_root),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("所有视频处理完成！")
