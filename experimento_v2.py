#!/usr/bin/env python3
"""Melhora controlada: mais dados e regularização escolhida em validação separada."""
import argparse
import gc
import json
import random
import time
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from diffusers import DiffusionPipeline
from sentence_transformers import SentenceTransformer
from transformers import CLIPTextModel, CLIPTokenizer

MODEL = 'segmind/tiny-sd'
GEMMA = 'google/embeddinggemma-2'
OUT = Path('resultados_v2')
OUT.mkdir(exist_ok=True)
DEVICE = 'cuda:0'


def dataset():
    objects = ['cat', 'dog', 'bird', 'fox', 'rabbit', 'robot', 'bicycle', 'teapot', 'car', 'boat', 'house', 'tree', 'flower', 'mushroom', 'astronaut', 'dragon']
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'black']
    places = ['in a garden', 'on a beach', 'in a forest', 'on a mountain', 'in the snow', 'beside a lake', 'in a city', 'in a field']
    styles = ['a photograph of', 'a watercolor painting of', 'a drawing of', 'a 3d render of']
    values = [f'{s} a {c} {o} {p}' for s in styles for c in colors for o in objects for p in places]
    random.Random(3407).shuffle(values)
    # Estes são exatamente os 96 testes do piloto. Nunca entram no treino ou validação.
    test = values[768:864]
    extra_styles = ['a studio photograph of', 'an oil painting of', 'a digital illustration of', 'a pencil sketch of']
    extra = [f'{s} a {c} {o} {p}' for s in extra_styles for c in colors for o in objects for p in places]
    tail = values[864:] + extra
    random.Random(9411).shuffle(tail)
    remaining = values[:768] + tail
    free = ['a red bicycle parked beside a brick wall', 'a small yellow rubber duck floating in a bathtub']
    return remaining[:6144], remaining[6144:6528], test, free


def collect():
    cache = OUT / 'representacoes.pt'
    if cache.exists():
        print('Usando representações em cache.', flush=True)
        return torch.load(cache, map_location='cpu', weights_only=True)
    train, val, test, free = dataset()
    texts = train + val + test + free + ['']
    print(f'Codificando {len(texts)} frases com EmbeddingGemma 2...', flush=True)
    gemma = SentenceTransformer(GEMMA, device=DEVICE, model_kwargs={'torch_dtype': torch.bfloat16}, config_kwargs={'vision_config': None, 'audio_config': None})
    x = gemma.encode(texts, batch_size=16, show_progress_bar=True, normalize_embeddings=False, convert_to_tensor=True).float().cpu()
    del gemma
    gc.collect()
    torch.cuda.empty_cache()
    print('Codificando os alvos CLIP...', flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder='tokenizer')
    teacher = CLIPTextModel.from_pretrained(MODEL, subfolder='text_encoder').to(DEVICE).eval()
    y = torch.empty((len(texts), 77, 768), dtype=torch.float16)
    with torch.inference_mode():
        for start in range(0, len(texts), 32):
            tokens = tokenizer(texts[start:start+32], padding='max_length', max_length=77, truncation=True, return_tensors='pt')
            y[start:start+32] = teacher(tokens.input_ids.to(DEVICE))[0].half().cpu()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    data = {'x': x, 'y': y, 'train': train, 'validation': val, 'test': test, 'free': free}
    torch.save(data, cache)
    return data


def predict(checkpoint, x):
    return ((x.to(DEVICE) - checkpoint['x_mean'].to(DEVICE)) @ checkpoint['weight'].to(DEVICE) + checkpoint['y_mean'].to(DEVICE)).reshape(-1, 77, 768).cpu()


def fit(data):
    n, nv, nt = map(len, (data['train'], data['validation'], data['test']))
    x = data['x'].to(DEVICE)
    train_y = data['y'][:n].to(device=DEVICE, dtype=torch.float32).flatten(1)
    xm, ym = x[:n].mean(0), train_y.mean(0)
    xc = x[:n] - xm
    gram = xc.T @ xc
    rhs = xc.T @ (train_y - ym)
    del train_y
    torch.cuda.empty_cache()
    val_x = x[n:n+nv] - xm
    val_y = data['y'][n:n+nv].to(device=DEVICE, dtype=torch.float32).flatten(1)
    candidates = {}
    best = float('inf')
    checkpoint = None
    for alpha in [0.001, 0.01, 0.1, 1.0, 10.0]:
        w = torch.linalg.solve(gram + alpha * torch.eye(gram.shape[0], device=DEVICE), rhs)
        error = float((val_x @ w + ym - val_y).square().mean())
        candidates[str(alpha)] = error
        print(f'Ridge {alpha}: MSE de validação {error:.6f}', flush=True)
        if error < best:
            best = error
            checkpoint = {'weight': w.cpu(), 'x_mean': xm.cpu(), 'y_mean': ym.cpu(), 'shape': (77, 768), 'ridge': alpha}
        del w
    assert checkpoint is not None
    torch.save(checkpoint, OUT / 'adaptador.pt')
    del x, xc, gram, rhs, val_x, val_y
    gc.collect()
    torch.cuda.empty_cache()
    old = torch.load('resultados/adaptador.pt', map_location='cpu', weights_only=True)
    eval_x = data['x'][n+nv:-1]
    target = data['y'][n+nv:-1].float()
    new_y, old_y = predict(checkpoint, eval_x), predict(old, eval_x)
    metrics = {'train_prompts': n, 'validation_prompts': nv, 'test_prompts': nt,
               'selected_ridge': checkpoint['ridge'], 'validation_mse': candidates,
               'test_old_mse': float((old_y[:nt]-target[:nt]).square().mean()),
               'test_new_mse': float((new_y[:nt]-target[:nt]).square().mean()),
               'free_old_mse': float((old_y[nt:]-target[nt:]).square().mean()),
               'free_new_mse': float((new_y[nt:]-target[nt:]).square().mean())}
    (OUT / 'metricas.json').write_text(json.dumps(metrics, indent=2))
    print('Métricas finais:', metrics, flush=True)
    return target, old_y, new_y


def render(data, predictions, args):
    original, old, new = predictions
    nt = len(data['test'])
    indices = [0, 1, nt, nt+1]
    prompts = data['test'][:2] + data['free']
    (OUT / 'prompts.json').write_text(json.dumps(prompts, indent=2))
    pipe = DiffusionPipeline.from_pretrained(MODEL, dtype=torch.float16).to(DEVICE)
    # Decodificação em blocos de 256 px: evita o caminho muito lento do VAE em 512 px nesta GPU.
    pipe.vae.enable_tiling()
    pipe.vae.tile_sample_min_size = 256
    pipe.vae.tile_latent_min_size = 32
    timings = {}
    for row, idx in enumerate(indices[:args.images]):
        for name, values in [('original', original), ('v1', old), ('v2', new)]:
            path = OUT / f'{row:02d}_{name}.png'
            if path.exists():
                continue
            print(f'{row+1}: {name}: {prompts[row]}', flush=True)
            started = time.monotonic()
            result = pipe(prompt_embeds=values[idx:idx+1].to(device=DEVICE, dtype=torch.float16),
                          negative_prompt_embeds=data['y'][-1:].to(device=DEVICE, dtype=torch.float16),
                          generator=torch.Generator(device=DEVICE).manual_seed(1000+row),
                          num_inference_steps=args.steps, height=512, width=512)
            result.images[0].save(path)
            timings[f'{row}_{name}'] = time.monotonic() - started
    (OUT / 'tempos.json').write_text(json.dumps(timings, indent=2))
    nrows = min(args.images, len(indices))
    canvas = Image.new('RGB', (3*384, nrows*424), '#eeeeee')
    draw = ImageDraw.Draw(canvas)
    for row in range(nrows):
        for col, name in enumerate(('original', 'v1', 'v2')):
            im = Image.open(OUT / f'{row:02d}_{name}.png').resize((384,384))
            canvas.paste(im, (col*384, row*424+40))
            draw.text((col*384+8,row*424+12),f'Prompt {row+1} | {name}', fill='black')
    canvas.save(OUT / 'comparacao.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=int, default=4)
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--fit-only', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('A GPU não está disponível para o PyTorch.')
    torch.set_grad_enabled(False)
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    data = collect()
    predictions = fit(data)
    if not args.fit_only:
        render(data, predictions, args)
    print('Concluído:', OUT.resolve(), flush=True)


if __name__ == '__main__':
    main()
