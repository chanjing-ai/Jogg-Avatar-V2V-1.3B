"""Create the shared text embedding consumed by the training dataset."""

import argparse
from pathlib import Path

import torch

from wan.configs.wan_t2v_1_3B_train import t2v_1_3B_train
from wan.modules.t5 import T5EncoderModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="A person is talking.")
    parser.add_argument(
        "--checkpoint",
        default="models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth",
    )
    parser.add_argument(
        "--tokenizer",
        default="models/Wan2.1-T2V-1.3B/google/umt5-xxl",
    )
    parser.add_argument("--output", default="data/share/context.pt")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device)
    encoder = T5EncoderModel(
        text_len=t2v_1_3B_train.text_len,
        dtype=t2v_1_3B_train.t5_dtype,
        device=device,
        checkpoint_path=args.checkpoint,
        tokenizer_path=args.tokenizer,
    )
    context = encoder([args.prompt], device)[0].detach().cpu().clone()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(context, output)
    print(f"Saved {tuple(context.shape)} context tensor to {output}")


if __name__ == "__main__":
    main()
