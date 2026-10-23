"""Tests for the headless CLI interface."""

import cv2
import numpy as np

from levelset_seg.cli import build_parser, run_cli


def test_cli_parser_defaults():
    parser = build_parser()
    args = parser.parse_args(["-i", "input.jpg", "-o", "output.png"])
    assert args.input == "input.jpg"
    assert args.output == "output.png"
    assert args.iterations is None
    assert args.convergence_tol is None
    assert args.seed is None


def test_cli_parser_custom_args():
    parser = build_parser()
    args = parser.parse_args([
        "--input", "in.png",
        "--output", "out.png",
        "--iterations", "10",
        "--convergence-tol", "0.01",
        "--seed", "42",
    ])
    assert args.iterations == 10
    assert args.convergence_tol == 0.01
    assert args.seed == 42


def test_cli_end_to_end(tmp_path):
    # Create a small synthetic image with a bright circle in the center
    img = np.zeros((80, 80), dtype=np.uint8)
    cv2.circle(img, (40, 40), 15, 220, -1)
    noise = np.random.RandomState(42).randint(0, 30, (80, 80), dtype=np.uint8)
    img = cv2.add(img, noise)

    in_file = tmp_path / "test_input.png"
    out_file = tmp_path / "test_output.png"
    cv2.imwrite(str(in_file), img)

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
    assert out_img.shape[0] == 80
    assert out_img.shape[1] == 80
    assert out_img.shape[2] == 3
