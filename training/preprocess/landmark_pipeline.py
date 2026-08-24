import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocess.extract_audio import extract_audio_process_videos
from preprocess.landmark import landmark_process_videos


def main():
    parser = argparse.ArgumentParser(description="InfiniteTalk preprocessing pipeline")
    parser.add_argument("--data_dir", type=str, default="", help="预处理数据目录")
    parser.add_argument("--aux_dir", type=str, default="", help="存放 wav/npy 的镜像目录")
    parser.add_argument("--num_gpus", type=int, default=1, help="使用的 GPU 数量")
    parser.add_argument("--procs_per_gpu", type=int, default=1, help="每个 GPU 的进程数")
    parser.add_argument("--num_procs", type=int, default=8, help="提取音频时的 CPU 进程数")
    parser.add_argument("--no_landmark", action="store_true", help="跳过 landmark，仅提取音频")
    parser.add_argument("--no_extract_audio", action="store_true", help="跳过音频提取")
    args = parser.parse_args()

    print(f"Processing {args.data_dir}")

    if not args.no_landmark:
        landmark_process_videos(args.data_dir, args.num_gpus, args.procs_per_gpu, args.aux_dir or None)

    if not args.no_extract_audio:
        extract_audio_process_videos(args.data_dir, args.num_procs, args.aux_dir or None)


if __name__ == "__main__":
    main()
