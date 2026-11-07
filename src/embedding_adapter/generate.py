#!/usr/bin/env python3
"""Gera usando EmbeddingGemma 2 + adaptador; dispensa o CLIP original em execução."""
import argparse
import gc
from pathlib import Path

import torch
from diffusers import DiffusionPipeline
from sentence_transformers import SentenceTransformer
from .train import GEMMA, MODEL, predict


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('prompt')
    parser.add_argument('--seed',type=int,default=1002)
    parser.add_argument('--output',default='resultados_v3/geracao_avulsa.png')
    args=parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('GPU indisponível para o PyTorch.')
    cp=torch.load('resultados_v3/adaptador.pt',map_location='cpu',weights_only=True)
    gemma=SentenceTransformer(GEMMA,device='cuda:0',model_kwargs={'torch_dtype':torch.bfloat16},config_kwargs={'vision_config':None,'audio_config':None})
    with torch.inference_mode():
        x=gemma.encode([args.prompt],convert_to_tensor=True,show_progress_bar=False).float().cpu()
        positive=predict(cp,x)
    del gemma
    gc.collect()
    torch.cuda.empty_cache()
    # O condicionamento negativo vazio é pré-calculado; não há text_encoder nesta pipeline.
    pipe=DiffusionPipeline.from_pretrained(MODEL,dtype=torch.float16,text_encoder=None,tokenizer=None).to('cuda:0')
    assert pipe.text_encoder is None and pipe.tokenizer is None
    pipe.vae.enable_tiling()
    pipe.vae.tile_sample_min_size=256
    pipe.vae.tile_latent_min_size=32
    image=pipe(prompt_embeds=positive.to(device='cuda:0',dtype=torch.float16),
               negative_prompt_embeds=cp['negative_prompt_embeds'].to(device='cuda:0',dtype=torch.float16),
               generator=torch.Generator(device='cuda:0').manual_seed(args.seed),
               num_inference_steps=30,height=512,width=512).images[0]
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    image.save(path)
    print('Gerado sem carregar o CLIP original:',path.resolve())


if __name__=='__main__':
    main()
