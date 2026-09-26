"""Checkpoint and gradient contracts without downloading a full vision tower."""
import sys
from types import SimpleNamespace

import torch
from skyball_gameplay.modeling import create_gameplay_model, load_checkpoint_model
from skyball_gameplay.training import build_parser


class StubQwenEncoder(torch.nn.Module):
    def __init__(self, snapshot, image_size):
        super().__init__()
        self.project = torch.nn.Linear(3, 2048)

    def forward(self, images):
        return self.project(images.mean(dim=(-2, -1)))


def test_qwen_frozen_temporal_head_and_checkpoint_roundtrip(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, 'skyball_gameplay.qwen_vision', SimpleNamespace(QwenFrameEncoder=StubQwenEncoder))
    model = create_gameplay_model(arch='qwen_vision', timm_model='', input_frames=4,
                                 num_classes=2, qwen_snapshot='fixture', image_size=[32, 32])
    model.encoder.requires_grad_(False)
    images = torch.rand(2, 4, 3, 32, 32)
    logits = model(images)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
    assert all(p.grad is None for p in model.encoder.parameters())
    assert all(p.grad is not None and p.grad.isfinite().all() for p in model.classifier.parameters())
    model.eval()
    torch.testing.assert_close(model(images), model(images.flip(1)))
    checkpoint = tmp_path / 'qwen.pt'
    torch.save(dict(arch='qwen_vision', timm_model='', input_frames=4, num_classes=2,
                    qwen_snapshot='fixture', image_size=[32, 32], normalization_mean=[.5]*3,
                    normalization_std=[.5]*3, resize_mode='letterbox', state_dict=model.state_dict()), checkpoint)
    loaded, metadata = load_checkpoint_model(checkpoint, 'cpu', require_input_frames=4)
    torch.testing.assert_close(loaded(images), model(images), rtol=0, atol=0)
    assert metadata['qwen_snapshot'] == 'fixture'
    assert metadata['normalization_mean'] == [.5]*3
    assert metadata['resize_mode'] == 'letterbox'


def test_letterbox_preserves_full_frame_and_native_ratio():
    import numpy as np
    from PIL import Image
    from skyball_gameplay.preprocessing import LetterboxResize
    image = Image.new('RGB', (1024, 576), (255, 0, 0))
    actual = np.asarray(LetterboxResize((384, 672))(image))
    assert actual.shape == (384, 672, 3)
    assert (actual[:3] == 128).all() and (actual[-3:] == 128).all()
    assert (actual[3:-3] == [255, 0, 0]).all()
    # A portrait source gets side padding, preserving both top and bottom.
    actual = np.asarray(LetterboxResize((384, 672))(Image.new('RGB', (100, 200), (0,255,0))))
    assert (actual[:, :240] == 128).all() and (actual[:, -240:] == 128).all()
    assert (actual[:, 240:-240] == [0,255,0]).all()


def test_frozen_qwen_cli_and_legacy_precision_default():
    parser = build_parser()
    args = parser.parse_args(['--arch', 'qwen_vision', '--qwen-snapshot', '/snapshot',
                              '--freeze-encoder', '--amp', '--amp-dtype', 'bfloat16'])
    assert args.freeze_encoder and args.amp_dtype == 'bfloat16'
    assert parser.parse_args([]).amp_dtype == 'float16'
