#!/usr/bin/env python3
"""Piloto: EmbeddingGemma 2 -> projeção ridge -> condicionamento do Tiny-SD."""
import argparse
import json
import random
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from sentence_transformers import SentenceTransformer
from transformers import CLIPTextModel, CLIPTokenizer

MODEL = 'segmind/tiny-sd'
GEMMA = 'google/embeddinggemma-2'
OUT = Path('resultados')
OUT.mkdir(exist_ok=True)


def prompts():
    objects = ['cat', 'dog', 'bird', 'fox', 'rabbit', 'robot', 'bicycle', 'teapot', 'car', 'boat', 'house', 'tree', 'flower', 'mushroom', 'astronaut', 'dragon']
    colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple', 'white', 'black']
    places = ['in a garden', 'on a beach', 'in a forest', 'on a mountain', 'in the snow', 'beside a lake', 'in a city', 'in a field']
    styles = ['a photograph of', 'a watercolor painting of', 'a drawing of', 'a 3d render of']
    values = [f'{s} a {c} {o} {p}' for s in styles for c in colors for o in objects for p in places]
    random.Random(3407).shuffle(values)
    return values[:768], values[768:864]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--images', type=int, default=3)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU ROCm indisponível para o PyTorch; nenhuma geração foi iniciada.')
    device = 'cuda:0'
    print('GPU:', torch.cuda.get_device_name(0), flush=True)
    train, test = prompts()
    all_prompts = train + test + ['']
    print('Carregando EmbeddingGemma 2...', flush=True)
    gemma = SentenceTransformer(GEMMA, device=device, model_kwargs={'torch_dtype': torch.bfloat16}, config_kwargs={'vision_config': None, 'audio_config': None})
    x = torch.tensor(gemma.encode(all_prompts, batch_size=8, show_progress_bar=True, normalize_embeddings=False), dtype=torch.float32)
    del gemma
    torch.cuda.empty_cache()
    print('Carregando o codificador CLIP original...', flush=True)
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder='tokenizer')
    teacher = CLIPTextModel.from_pretrained(MODEL, subfolder='text_encoder').to(device).eval()
    ys = []
    with torch.no_grad():
        for start in range(0, len(all_prompts), 16):
            tokens = tokenizer(all_prompts[start:start+16], padding='max_length', max_length=tokenizer.model_max_length, truncation=True, return_tensors='pt')
            ys.append(teacher(tokens.input_ids.to(device))[0].float().cpu())
    y = torch.cat(ys)
    del teacher
    torch.cuda.empty_cache()
    n = len(train)
    # A regularização e a interceptação são ajustadas apenas ao conjunto de treino.
    x_train, y_train = x[:n].to(device), y[:n].flatten(1).to(device)
    x_mean, y_mean = x_train.mean(0), y_train.mean(0)
    xc, yc = x_train - x_mean, y_train - y_mean
    ridge = 1.0
    gram = xc.T @ xc
    w = torch.linalg.solve(gram + ridge * torch.eye(gram.shape[0], device=device), xc.T @ yc)
    predictions = ((x[n:-1].to(device) - x_mean) @ w + y_mean).reshape(-1, *y.shape[1:]).cpu()
    direct = x[n:-1, None, :].expand(-1, y.shape[1], -1)
    actual = y[n:-1]
    metrics = {'train_prompts': n, 'test_prompts': len(test), 'ridge': ridge,
               'adapted_mse': float((predictions-actual).square().mean()),
               'direct_mse': float((direct-actual).square().mean()),
               'mean_baseline_mse': float((y_mean.cpu().reshape(1,*y.shape[1:])-actual).square().mean())}
    (OUT / 'metricas.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    torch.save({'weight': w.cpu(), 'x_mean': x_mean.cpu(), 'y_mean': y_mean.cpu(),
                'shape': y.shape[1:], 'ridge': ridge}, OUT / 'adaptador.pt')
    print('Métricas:', metrics, flush=True)
    del xc, yc, gram, w, x_train, y_train
    torch.cuda.empty_cache()
    print('Carregando Tiny-SD para comparar imagens...', flush=True)
    pipe = DiffusionPipeline.from_pretrained(MODEL, torch_dtype=torch.float16).to(device)
    for i in range(min(args.images, len(test))):
        print(f'Prompt {i}: {test[i]}', flush=True)
        for name, positive in [('original', actual[i]), ('direto', direct[i]), ('adaptado', predictions[i])]:
            generator = torch.Generator(device=device).manual_seed(1000 + i)
            image = pipe(prompt_embeds=positive.unsqueeze(0).to(device=device,dtype=torch.float16), negative_prompt_embeds=y[-1:].to(device=device,dtype=torch.float16), generator=generator, num_inference_steps=args.steps, height=args.size, width=args.size).images[0]
            image.save(OUT / f'{i:02d}_{name}.png')
    (OUT / 'prompts.json').write_text(json.dumps(test[:args.images], indent=2), encoding='utf-8')
    print('Pronto:', OUT.resolve(), flush=True)


if __name__ == '__main__':
    main()
