import repo_paths

repo_paths.ensure_project_src_path()

import os
import argparse
import cv2
import json
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm


MODEL_ROOT = Path(os.environ.get("CHANJING_AVATAR_MODEL_DIR", "models"))


def create_landmark_runner(device_id=0):
    """
    为指定GPU创建LandmarkRunner实例
    """
    from video_preprocess.face_landmark.lp_lmk.human_landmark_runner import LandmarkRunner

    runner = LandmarkRunner(
        ckpt_path=str(MODEL_ROOT / "landmark.onnx"),
        onnx_provider='cuda',
        device_id=device_id)
    runner.warmup()
    return runner


def create_scrfd_runner():
    from video_preprocess.scrfd_crop_face.files.scrfd_api import ScrfdAPI
    runner = ScrfdAPI(
        model_path=str(MODEL_ROOT / "scrfd_500m_bnkps.onnx"),
        provider='gpu',
    )
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
    if not face_box:
        return False
    for point in landmarks:
        if not is_point_in_bbox(point, face_box):
            return False
    return True


def process_video(video_path, lmk_runner, scrfd_runner):
    """
    处理单个视频
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            return None

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lmk = None

        # 使用 SCRFD 检测人脸，获取初始关键点
        crop_face_box, lmk_scrfd = scrfd_runner._extract_image_face(frame)
        if lmk_scrfd:
            lmk = np.array(lmk_scrfd)

        landmarks = []
        frame_idx = 0
        last_scrfd_frame = 0
        reinit_interval = int(fps * 2)  # 每 2 秒重新初始化一次

        for _ in range(10):
            lmk = lmk_runner.run(frame, lmk)

        while True:
            lmk203 = lmk_runner.run(frame, lmk)
            landmarks.append(lmk203)
            frame_idx += 1

            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # 定期重新使用 SCRFD 初始化，避免错误传播
            if frame_idx - last_scrfd_frame >= reinit_interval:
                crop_face_box_new, lmk_scrfd_new = scrfd_runner._extract_image_face(frame)
                if lmk_scrfd_new:
                    lmk = np.array(lmk_scrfd_new)
                    last_scrfd_frame = frame_idx
                    # 快速迭代几次以稳定关键点
                    for _ in range(5):
                        lmk = lmk_runner.run(frame, lmk)
                else:
                    # 如果 SCRFD 检测失败，继续使用上一帧的关键点
                    lmk = lmk203.copy()
            else:
                lmk = lmk203.copy()

        cap.release()
        return landmarks

    except Exception as e:
        print(f"处理视频 {video_path} 时出错: {e}")
        return None


def process_videos_on_gpu(gpu_id, video_chunk, proc_idx):
    """
    在指定GPU上处理视频列表
    """
    print(f"进程 {proc_idx} 在GPU {gpu_id} 上启动，处理 {len(video_chunk)} 个视频")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    lmk_runner = create_landmark_runner(device_id=0)
    scrfd_runner = create_scrfd_runner()

    success_count = 0
    error_count = 0

    for video_path in tqdm(video_chunk, desc=f"进程{proc_idx}(GPU{gpu_id})"):
        try:
            landmarks_path = video_path.with_suffix('.npy')

            if landmarks_path.exists():
                print(f"进程{proc_idx}: 跳过已存在的文件: {landmarks_path}")
                continue

            landmarks = process_video(video_path, lmk_runner, scrfd_runner)
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


def landmark_process_videos(data_dir, num_gpus, procs_per_gpu):
    video_files = list(Path(data_dir).rglob("*.mp4")) + \
                  list(Path(data_dir).rglob("*.avi")) + \
                  list(Path(data_dir).rglob("*.mov")) + \
                  list(Path(data_dir).rglob("*.mkv"))
    video_chunks = [[] for _ in range(num_gpus * procs_per_gpu)]
    for idx, video_path in enumerate(video_files):
        video_chunks[idx % (num_gpus * procs_per_gpu)].append(video_path)
    processes = []
    for proc_idx in range(num_gpus * procs_per_gpu):
        gpu_id = proc_idx // procs_per_gpu
        p = mp.Process(target=process_videos_on_gpu, args=(gpu_id, video_chunks[proc_idx], proc_idx))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("所有视频处理完成！")


def main():
    parser = argparse.ArgumentParser(description='Pipeline')
    parser.add_argument('--data_dir', type=str, default="", help='预处理数据目录')
    parser.add_argument('--num_gpus', type=int, default=1, help='使用的GPU数量')
    parser.add_argument('--procs_per_gpu', type=int, default=1, help='每个GPU上的进程数')
    parser.add_argument('--Use_Landmark', type=bool, default=True, help='是否使用Landmark')
    args = parser.parse_args()

    data_dir_list = [args.data_dir]
    for data_dir in data_dir_list:
        print(f"Processing {data_dir}")
        if args.data_dir:
            if args.Use_Landmark:
                landmark_process_videos(data_dir, args.num_gpus, args.procs_per_gpu)


if __name__ == "__main__":
    main()
