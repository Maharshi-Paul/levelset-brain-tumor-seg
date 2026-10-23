"""Tests for the headless CLI interface."""

import json
from pathlib import Path

import cv2
import numpy as np

from levelset_seg.cli import build_parser, run_cli


def _make_synthetic_image(path: Path, size: int = 80) -> None:
    img = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 4, 220, -1)
    noise = np.random.RandomState(42).randint(0, 30, (size, size), dtype=np.uint8)
    img = cv2.add(img, noise)
    cv2.imwrite(str(path), img)


def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["-i", "input.jpg", "-o", "output.png"])
    assert args.input == "input.jpg"
    assert args.output == "output.png"
    assert args.iterations is None
    assert args.convergence_tol is None
    assert args.seed is None
    assert args.save_panels is False
    assert args.json is None


def test_cli_parser_custom_args():
    parser = build_parser()
    args = parser.parse_args([
        "--input", "in.png",
        "--output", "out.png",
        "--iterations", "10",
        "--convergence-tol", "0.01",
        "--seed", "42",
        "--save-panels",
        "--json", "metrics.json",
    ])
    assert args.iterations == 10
    assert args.convergence_tol == 0.01
    assert args.seed == 42
    assert args.save_panels is True
    assert args.json == "metrics.json"


def test_cli_end_to_end(tmp_path):
    in_file = tmp_path / "test_input.png"
    out_file = tmp_path / "test_output.png"
    _make_synthetic_image(in_file, size=80)

    parser = build_parser()
    args = parser.parse_args([
        "-i", str(in_file),
        "-o", str(out_file),
        "--iterations", "2",
        "--seed", "123",
    ])

    run_cli(args)

    assert out_file.exists()
    out_img = cv2.imread(str(out_file))
    assert out_img is not None
    assert out_img.shape == (80, 80, 3)


def test_cli_save_panels_and_json(tmp_path):
    in_file = tmp_path / "sample.png"
    out_file = tmp_path / "sample_seg.png"
    json_file = tmp_path / "metrics.json"
    _make_synthetic_image(in_file, size=60)

    parser = build_parser()
    args = parser.parse_args([
        "-i", str(in_file),
        "-o", str(out_file),
        "--iterations", "2",
        "--seed", "42",
        "--save-panels",
        "--json", str(json_file),
    ])

    run_cli(args)

    assert out_file.exists()
    panels_file = tmp_path / "sample_seg_panels.png"
    assert panels_file.exists()
    panels_img = cv2.imread(str(panels_file))
    assert panels_img is not None
    # Panels are 3 images stacked horizontally: (H, 3*W, 3)
    assert panels_img.shape[0] == 60
    assert panels_img.shape[1] == 180
    assert panels_img.shape[2] == 3

    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert "input" in data
    assert "output" in data
    assert "roi_solidity" in data
    assert "duration_seconds" in data
    assert data["iterations"] == 2


def test_cli_directory_batch(tmp_path):
    in_dir = tmp_path / "inputs"
    out_dir = tmp_path / "outputs"
    in_dir.mkdir()

    _make_synthetic_image(in_dir / "slice1.png", size=50)
    _make_synthetic_image(in_dir / "slice2.jpg", size=50)

    json_file = tmp_path / "batch_metrics.json"
    parser = build_parser()
    args = parser.parse_args([
        "-i", str(in_dir),
        "-o", str(out_dir),
        "--iterations", "2",
        "--seed", "99",
        "--save-panels",
        "--json", str(json_file),
    ])

    run_cli(args)

    assert (out_dir / "slice1_seg.png").exists()
    assert (out_dir / "slice1_panels.png").exists()
    assert (out_dir / "slice2_seg.png").exists()
    assert (out_dir / "slice2_panels.png").exists()

    assert json_file.exists()
    data = json.loads(json_file.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) == 2
    assert data[0]["iterations"] == 2
