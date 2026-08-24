import json
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from release_utils import BASE_MODEL_FILES, split_model_paths, validate_inference_inputs


class ReleaseTest(unittest.TestCase):
    def test_pyproject_is_valid_and_pins_cuda_torch(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        dependencies = project["project"]["dependencies"]
        self.assertIn("torch==2.8.0", dependencies)
        self.assertEqual(
            project["tool"]["uv"]["sources"]["torch"]["index"],
            "pytorch-cu128",
        )

    def test_model_paths_are_trimmed(self):
        self.assertEqual(
            split_model_paths("one.safetensors, two.safetensors"),
            [Path("one.safetensors"), Path("two.safetensors")],
        )

    def test_huggingface_index_matches_two_shard_release(self):
        index = json.loads(
            (ROOT / "huggingface" / "model.safetensors.index.json").read_text(
                encoding="utf-8"
            )
        )
        shard_names = set(index["weight_map"].values())
        self.assertEqual(
            shard_names,
            {
                "model-00001-of-00002.safetensors",
                "model-00002-of-00002.safetensors",
            },
        )
        self.assertGreater(index["metadata"]["total_size"], 7_000_000_000)

    def test_inference_preflight_accepts_complete_layout(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            base = root / "base"
            for relative in BASE_MODEL_FILES:
                path = base / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            audio_encoder = root / "wav2vec"
            audio_encoder.mkdir()
            checkpoint = root / "model.safetensors"
            checkpoint.touch()
            video = root / "source.mp4"
            audio = root / "drive.wav"
            video.touch()
            audio.touch()
            video.with_suffix(".npy").touch()
            video.with_name("source_bbox.json").write_text("{}", encoding="utf-8")
            input_json = root / "input.json"
            input_json.write_text(
                json.dumps(
                    [
                        {
                            "prompt": "A person is talking.",
                            "cond_video": str(video),
                            "cond_audio": {"person1": str(audio)},
                        }
                    ]
                ),
                encoding="utf-8",
            )
            args = SimpleNamespace(
                ckpt_dir=str(base),
                infinitetalk_dir=str(checkpoint),
                wav2vec_dir=str(audio_encoder),
                input_json=str(input_json),
                scene_seg=False,
            )
            jobs = validate_inference_inputs(args)
            self.assertEqual(len(jobs), 1)

    def test_release_contains_no_internal_absolute_paths(self):
        forbidden = ("/home/jingyan", "/nas-hdd/", "/traindata-")
        for path in ROOT.rglob("*"):
            if path == Path(__file__).resolve():
                continue
            relative_parts = path.relative_to(ROOT).parts
            if any(part in {".git", ".venv", "data", "models"} for part in relative_parts):
                continue
            if path.is_file() and path.suffix in {".py", ".md", ".json", ".toml"}:
                text = path.read_text(encoding="utf-8")
                for value in forbidden:
                    self.assertNotIn(value, text, str(path))

    def test_flashhead_package_is_not_part_of_this_release(self):
        self.assertFalse((ROOT / "src" / "flash_head").exists())
        multitalk = (ROOT / "src" / "wan" / "multitalk.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("from flash_head", multitalk)


if __name__ == "__main__":
    unittest.main()
