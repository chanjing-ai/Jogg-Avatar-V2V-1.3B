import os
import cv2
import json
import numpy as np
import multiprocessing as mp
import argparse
from pathlib import Path
from tqdm import tqdm


MODEL_ROOT = Path(os.environ.get("JOGG_AVATAR_MODEL_DIR", "models"))

def create_landmark_runner():
    """
    为指定GPU创建LandmarkRunner实例
    """
    from face_landmark.lp_lmk.human_landmark_runner import LandmarkRunner

    runner = LandmarkRunner(
        ckpt_path=str(MODEL_ROOT / "landmark.onnx"),
        onnx_provider='cuda',
        device_id=0)
    runner.warmup()
    return runner

def is_point_in_bbox(point, bbox):
    """
    判断点是否在边界框内
    point: [x, y] 坐标
    bbox: [xmin, ymin, xmax, ymax]
    """
    x, y = point
    xmin, ymin, xmax, ymax = bbox
    return xmin <= x <= xmax and ymin <= y <= ymax

def is_landmarks_in_face_box(landmarks, face_box):
    """
    判断关键点是否全部在人脸框内
    landmarks: 203个关键点的数组，每个点为 [x, y]
    face_box: [xmin, ymin, xmax, ymax]
    """
    if not face_box:  # 如果没有人脸框
        return False

    # 检查所有关键点是否都在人脸框内
    for point in landmarks:
        if not is_point_in_bbox(point, face_box):
            return False
    return True

def process_video(video_path, lmk_runner):
    """
    处理单个视频
    """
    try:
        # 读取视频
        cap = cv2.VideoCapture(str(video_path))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lmk = None
        landmarks = []

        # 预热
        for _ in range(10):
            lmk = lmk_runner.run(frame, lmk)

        while True:
            lmk203 = lmk_runner.run(frame, lmk)
            # lmk5 = []
            # lmk5.append(lmk203[48])
            # lmk5.append(lmk203[67])
            # lmk5.append(lmk203[197])
            # lmk5.append(lmk203[198])
            # lmk5.append(lmk203[201])
            landmarks.append(lmk203)

            # 转换BGR到RGB色彩空间
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            lmk = lmk203.copy()


        # 释放视频捕获对象
        cap.release()
        return landmarks

    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {e}")
        print(len(landmarks))
        return None

def process_videos_on_gpu(gpu_id, video_chunk, proc_idx, data_root=None, aux_root=None):
    """
    在指定GPU上处理视频列表
    """
    print(f"进程 {proc_idx} 在GPU {gpu_id} 上启动，处理 {len(video_chunk)} 个视频")

    # 关键：在导入任何CUDA相关库之前设置环境变量
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # 现在创建runner实例
    lmk_runner = create_landmark_runner()

    success_count = 0
    error_count = 0

    for video_path in tqdm(video_chunk, desc=f"进程{proc_idx}(GPU{gpu_id})"):
        try:
            if data_root is not None and aux_root is not None:
                rel = video_path.resolve().relative_to(data_root)
                landmarks_path = (aux_root / rel).with_suffix('.npy')
                landmarks_path.parent.mkdir(parents=True, exist_ok=True)
            else:
                landmarks_path = video_path.with_suffix('.npy')

            # 检查是否已经存在结果文件
            if landmarks_path.exists():
                print(f"进程{proc_idx}: 跳过已存在的文件: {landmarks_path}")
                continue

            landmarks = process_video(video_path, lmk_runner)
            if landmarks is not None:
                np.save(landmarks_path, np.array(landmarks))
                success_count += 1
                print(f"进程{proc_idx}: 成功处理 {video_path}")
            else:
                error_count += 1
                print(f"进程{proc_idx}: 处理失败 {video_path}")

        except Exception as e:
            error_count += 1
            print(f"进程{proc_idx}: 处理视频 {video_path} 时出错: {e}")
            continue

    print(f"进程 {proc_idx} 在GPU {gpu_id} 上完成，成功: {success_count}, 失败: {error_count}")


def landmark_process_videos(data_dir, num_gpus, procs_per_gpu, aux_dir=None):
    video_files = list(Path(data_dir).rglob("*.mp4")) + \
                  list(Path(data_dir).rglob("*.avi")) + \
                  list(Path(data_dir).rglob("*.mov")) + \
                  list(Path(data_dir).rglob("*.mkv"))
    video_chunks = [[] for _ in range(num_gpus * procs_per_gpu)]
    for idx, video_path in enumerate(video_files):
        video_chunks[idx % (num_gpus * procs_per_gpu)].append(video_path)
    processes = []
    data_root = Path(data_dir).expanduser().resolve()
    aux_root = Path(aux_dir).expanduser().resolve() if aux_dir else None
    for proc_idx in range(num_gpus * procs_per_gpu):
        gpu_id = (proc_idx // procs_per_gpu)
        p = mp.Process(
            target=process_videos_on_gpu,
            args=(gpu_id, video_chunks[proc_idx], proc_idx, data_root, aux_root),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("所有视频处理完成！")
