"""Verificações do fluxo local que não exigem GPU ou downloads."""
import contextlib
import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from PIL import Image

from embedding_adapter import train


class WorkflowTests(unittest.TestCase):
    def test_predict_preserves_conditioning_shape(self):
        rank = 2
        model = train.ResidualAdapter(768, rank)
        checkpoint = {
            'rank': rank,
            'state_dict': model.state_dict(),
            'x_mean': torch.zeros(768),
            'x_std': torch.ones(768),
            'linear': torch.zeros(768, rank),
            'scale': torch.ones(rank),
            'basis': torch.zeros(77 * 768, rank),
            'y_mean': torch.ones(77 * 768),
        }
        with patch.object(train, 'DEVICE', 'cpu'):
            actual = train.predict(checkpoint, torch.zeros(3, 768))
        self.assertEqual(tuple(actual.shape), (3, 77, 768))
        self.assertTrue(torch.all(actual == 1))

    def test_evaluation_and_render_without_old_checkpoints(self):
        class FakePipeline:
            def __init__(self):
                self.vae = SimpleNamespace(enable_tiling=lambda: None)

            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                return cls()

            def to(self, *args):
                return self

            def __call__(self, **kwargs):
                return SimpleNamespace(images=[Image.new('RGB', (16, 16), 'white')])

        dataset = {'train': [{}], 'validation': [{}], 'test': [{'caption': 'a test scene'}]}
        cache = {'x': torch.zeros(8, 768), 'y': torch.zeros(8, 77, 768, dtype=torch.float16)}
        fake_prediction = lambda cp, x, nonlinear=True: torch.zeros(len(x), 77, 768)
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                output = Path('resultados_v3')
                output.mkdir()
                with patch.object(train, 'OUT', output), patch.object(train, 'DEVICE', 'cpu'), \
                     patch.object(train, 'predict', fake_prediction), \
                     patch.object(train, 'DiffusionPipeline', FakePipeline), \
                     contextlib.redirect_stdout(io.StringIO()):
                    target, predictions = train.evaluate(dataset, cache, {'best_epoch': 0, 'rank': 2})
                    self.assertEqual(set(predictions), {'linear_coco', 'mlp_coco'})
                    train.render(dataset, cache, target, predictions, SimpleNamespace(images=1))
                with Image.open(output / 'comparacao.png') as image:
                    self.assertEqual(image.size, (768, 424))
                self.assertFalse((output / '00_v2.png').exists())
            finally:
                os.chdir(original_cwd)

    def test_cli_help_does_not_create_results(self):
        with tempfile.TemporaryDirectory() as directory:
            for module in ('embedding_adapter.train', 'embedding_adapter.generate'):
                result = subprocess.run(
                    [sys.executable, '-m', module, '--help'],
                    cwd=directory, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)
            self.assertFalse((Path(directory) / 'resultados_v3').exists())


if __name__ == '__main__':
    unittest.main()
