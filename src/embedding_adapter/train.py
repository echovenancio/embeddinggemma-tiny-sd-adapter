#!/usr/bin/env python3
"""Legendas COCO: projeção linear e MLP residual de embedding global para estados CLIP."""
import argparse
import gc
import hashlib
import json
import random
import re
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torch import nn
from huggingface_hub import hf_hub_download, HfApi
from sentence_transformers import SentenceTransformer
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DiffusionPipeline
from PIL import Image, ImageDraw

OUT = Path('resultados_v3')
MODEL = 'segmind/tiny-sd'
GEMMA = 'google/embeddinggemma-2'
SOURCE = 'phiyodr/coco2017'
DEVICE = 'cuda:0'
REFS = ['a photograph of a purple cat in a city', 'a photograph of a blue teapot in a field',
        'a red bicycle parked beside a brick wall', 'a small yellow rubber duck floating in a bathtub']


def clean(text):
    return re.sub(r'[^a-z0-9]+', ' ', text.lower()).strip()


def make_dataset():
    path = OUT / 'dataset.json'
    if path.exists():
        return json.loads(path.read_text())
    sha = HfApi().dataset_info(SOURCE).sha
    names = {'train': 'data/train-00000-of-00001-0084e041f1902997.parquet',
             'validation': 'data/validation-00000-of-00001-e3c37e369512a3aa.parquet'}
    tables = {}
    for split, filename in names.items():
        file = hf_hub_download(SOURCE, filename, repo_type='dataset', revision=sha)
        tables[split] = pq.read_table(file, columns=['image_id', 'captions']).to_pylist()
    rng = random.Random(39171)
    for rows in tables.values():
        rng.shuffle(rows)
    used = {clean(s) for s in REFS}
    def choose(rows, count):
        selected = []
        for row in rows:
            captions = list(row['captions'])
            rng.shuffle(captions)
            for caption in captions:
                key = clean(caption)
                if key not in used and 4 <= len(key.split()) <= 40:
                    selected.append({'image_id': row['image_id'], 'caption': caption.strip()})
                    used.add(key)
                    break
            if len(selected) == count:
                break
        if len(selected) != count:
            raise RuntimeError('Número insuficiente de legendas elegíveis.')
        return selected
    train = choose(tables['train'], 16000)
    val = choose(tables['validation'][:2500], 512)
    test = choose(tables['validation'][2500:], 256)
    for a,b in [(train,val),(train,test),(val,test)]:
        assert not ({r['image_id'] for r in a} & {r['image_id'] for r in b})
        assert not ({clean(r['caption']) for r in a} & {clean(r['caption']) for r in b})
    data = {'source': SOURCE, 'revision': sha, 'train': train, 'validation': val, 'test': test, 'references': REFS}
    path.write_text(json.dumps(data, indent=2))
    return data


def collect(dataset):
    path = OUT / 'representacoes.pt'
    digest = hashlib.sha256(json.dumps(dataset, sort_keys=True).encode()).hexdigest()
    if path.exists():
        cache = torch.load(path, map_location='cpu', weights_only=True)
        assert cache['dataset_sha256'] == digest
        return cache
    texts = [r['caption'] for split in ['train','validation','test'] for r in dataset[split]] + REFS + ['']
    print(f'EmbeddingGemma 2: {len(texts)} frases', flush=True)
    gemma = SentenceTransformer(GEMMA, device=DEVICE, model_kwargs={'torch_dtype': torch.bfloat16}, config_kwargs={'vision_config':None, 'audio_config':None})
    x = gemma.encode(texts, batch_size=16, convert_to_tensor=True, show_progress_bar=True).float().cpu()
    del gemma
    gc.collect()
    torch.cuda.empty_cache()
    print('Codificando alvos CLIP...', flush=True)
    teacher = CLIPTextModel.from_pretrained(MODEL, subfolder='text_encoder').to(DEVICE).eval()
    tokenizer = CLIPTokenizer.from_pretrained(MODEL, subfolder='tokenizer')
    y = torch.empty((len(texts), 77, 768), dtype=torch.float16)
    with torch.inference_mode():
        for start in range(0,len(texts),32):
            ids = tokenizer(texts[start:start+32], padding='max_length', max_length=77, truncation=True, return_tensors='pt').input_ids.to(DEVICE)
            y[start:start+32] = teacher(ids)[0].half().cpu()
    del teacher
    gc.collect()
    torch.cuda.empty_cache()
    cache = {'x':x, 'y':y, 'dataset_sha256':digest}
    torch.save(cache,path)
    return cache


class ResidualAdapter(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.residual = nn.Sequential(nn.Linear(input_dim,1024), nn.GELU(), nn.Dropout(0.1),
                                      nn.Linear(1024,1024), nn.GELU(), nn.Dropout(0.1), nn.Linear(1024,output_dim))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, x, linear):
        return linear + self.residual(x)


def cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def train(dataset, cache, args):
    checkpoint = OUT / 'adaptador.pt'
    if checkpoint.exists():
        print('Usando adaptador já treinado.', flush=True)
        saved = torch.load(checkpoint,map_location='cpu',weights_only=True)
        assert saved['dataset_sha256'] == cache['dataset_sha256']
        assert saved['rank'] == args.rank, 'O cache foi treinado com outra dimensão PCA.'
        return saved
    n = len(dataset['train'])
    nv = len(dataset['validation'])
    torch.manual_seed(7241)
    with torch.no_grad():
        print('Comprimindo os alvos CLIP por PCA (somente treino)...', flush=True)
        y = cache['y'][:n].flatten(1).to(device=DEVICE,dtype=torch.float32)
        ym = y.mean(0)
        y.sub_(ym)
        # Base treinada só com alvos de treino; a reconstrução preserva a sequência de 77 tokens.
        _,_,basis = torch.pca_lowrank(y,q=args.rank,center=False,niter=1)
        coeff = y @ basis
        del y
        gc.collect()
        torch.cuda.empty_cache()
        validation_y = cache['y'][n:n+nv].flatten(1).to(device=DEVICE,dtype=torch.float32)
        val_coeff = (validation_y-ym) @ basis
        floor = float(((val_coeff @ basis.T + ym)-validation_y).square().mean())
        del validation_y
        scale = coeff.std(0).clamp_min(0.01)
        target = coeff/scale
        val_target = val_coeff/scale
        x = cache['x'][:n+nv].to(DEVICE)
        xm, xs = x[:n].mean(0), x[:n].std(0).clamp_min(0.001)
        x = (x-xm)/xs
        xt, xv = x[:n], x[n:]
        gram = xt.T @ xt
        rhs = xt.T @ target
        weights = (scale/scale.mean()).square()
        best = float('inf')
        linear = None
        ridge_scores = {}
        for alpha in [1.0,10.0,100.0,1000.0]:
            w = torch.linalg.solve(gram + alpha*torch.eye(768,device=DEVICE),rhs)
            score = float(((xv@w-val_target).square()*weights).mean())
            ridge_scores[str(alpha)] = score
            if score < best:
                best, linear, alpha_best = score,w,alpha
        base_train, base_val = xt@linear, xv@linear
        del gram,rhs,coeff,val_coeff
    print(f'PCA: piso MSE de validação {floor:.5f}; ridge selecionada {alpha_best}',flush=True)
    model = ResidualAdapter(768,args.rank).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0006,weight_decay=0.01)
    best_state = cpu_state(model)
    best_loss = best
    best_epoch = 0
    stale = 0
    curve = []
    began = time.monotonic()
    for epoch in range(1,args.epochs+1):
        model.train()
        order = torch.randperm(n,device=DEVICE)
        loss_sum = 0.0
        for start in range(0,n,256):
            idx = order[start:start+256]
            pred = model(xt[idx],base_train[idx])
            loss = ((pred-target[idx]).square()*weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            loss_val = float(((model(xv,base_val)-val_target).square()*weights).mean())
        curve.append({'epoch':epoch,'train_loss':loss_sum/n,'validation_loss':loss_val})
        print(f'Época {epoch}: treino {loss_sum/n:.5f}; validação {loss_val:.5f}; {time.monotonic()-began:.1f}s',flush=True)
        if loss_val < best_loss:
            best_loss,best_epoch,stale = loss_val,epoch,0
            best_state = cpu_state(model)
        else:
            stale += 1
        if stale >= 10:
            break
    result = {'state_dict':best_state,'x_mean':xm.cpu(),'x_std':xs.cpu(),'y_mean':ym.cpu(),
              'basis':basis.cpu(),'scale':scale.cpu(),'linear':linear.cpu(),'rank':args.rank,
              'best_epoch':best_epoch,'validation_linear_loss':best,'validation_mlp_loss':best_loss,
              'validation_pca_floor_mse':floor,'ridge':alpha_best,'ridge_validation_scores':ridge_scores,
              'dataset_sha256':cache['dataset_sha256'],'epochs':len(curve),
              'negative_prompt_embeds':cache['y'][-1:].clone()}
    torch.save(result,checkpoint)
    (OUT/'treino.json').write_text(json.dumps(curve,indent=2))
    del model,optimizer,x,xt,xv,target,val_target,base_train,base_val,basis,linear
    gc.collect()
    torch.cuda.empty_cache()
    return result


@torch.inference_mode()
def predict(cp,x,nonlinear=True):
    model = ResidualAdapter(768,cp['rank']).to(DEVICE).eval()
    model.load_state_dict(cp['state_dict'])
    x = (x.to(DEVICE)-cp['x_mean'].to(DEVICE))/cp['x_std'].to(DEVICE)
    z = x@cp['linear'].to(DEVICE)
    if nonlinear:
        z = model(x,z)
    result = ((z*cp['scale'].to(DEVICE))@cp['basis'].to(DEVICE).T+cp['y_mean'].to(DEVICE)).reshape(-1,77,768).cpu()
    del model
    return result


@torch.inference_mode()
def evaluate(dataset,cache,cp):
    start = len(dataset['train']) + len(dataset['validation'])
    nt = len(dataset['test'])
    x = cache['x'][start:-1]
    target = cache['y'][start:-1].float()
    linear,mlp = predict(cp,x,False),predict(cp,x,True)
    olds = {}
    for name,path in [('v1','resultados/adaptador.pt'),('v2','resultados_v2/adaptador.pt')]:
        if not Path(path).is_file():
            print(f'Comparação opcional {name} ignorada: checkpoint não disponível.', flush=True)
            continue
        old = torch.load(path,map_location='cpu',weights_only=True)
        olds[name] = ((x.to(DEVICE)-old['x_mean'].to(DEVICE))@old['weight'].to(DEVICE)+old['y_mean'].to(DEVICE)).reshape(-1,77,768).cpu()
    preds = {**olds,'linear_coco':linear,'mlp_coco':mlp}
    metrics = {'train':len(dataset['train']),'validation':len(dataset['validation']),'test':nt,
               'best_epoch':cp['best_epoch'],'rank':cp['rank'],
               'parameters_mlp':sum(p.numel() for p in ResidualAdapter(768,cp['rank']).parameters()),
               'test_mse':{name:float((p[:nt]-target[:nt]).square().mean()) for name,p in preds.items()},
               'references_mse':{name:float((p[nt:]-target[nt:]).square().mean()) for name,p in preds.items()},
               'reference_per_prompt':{name:[float((p[nt+i]-target[nt+i]).square().mean()) for i in range(len(REFS))] for name,p in preds.items()}}
    (OUT/'metricas.json').write_text(json.dumps(metrics,indent=2))
    print('Avaliação final:',json.dumps(metrics),flush=True)
    return target,preds


@torch.inference_mode()
def render(dataset,cache,target,preds,args):
    nt = len(dataset['test'])
    # Referências já conhecidas e os dois primeiros testes COCO, definidos antes de ver imagens.
    indices = [nt,nt+1,nt+2,nt+3,0,1][:args.images]
    texts = [r['caption'] for r in dataset['test']] + REFS
    titles = [texts[i] for i in indices]
    (OUT/'prompts.json').write_text(json.dumps(titles,indent=2))
    pipe = DiffusionPipeline.from_pretrained(MODEL,dtype=torch.float16).to(DEVICE)
    pipe.vae.enable_tiling()
    pipe.vae.tile_sample_min_size=256
    pipe.vae.tile_latent_min_size=32
    conditions = [('original',target)]
    if 'v2' in preds:
        conditions.append(('v2',preds['v2']))
    conditions.append(('v3',preds['mlp_coco']))
    for row,idx in enumerate(indices):
        for name,values in conditions:
            path = OUT/f'{row:02d}_{name}.png'
            if path.exists():
                continue
            print(f'Imagem {row+1}/{len(indices)} {name}: {texts[idx]}',flush=True)
            image = pipe(prompt_embeds=values[idx:idx+1].to(device=DEVICE,dtype=torch.float16),
                         negative_prompt_embeds=cache['y'][-1:].to(device=DEVICE,dtype=torch.float16),
                         generator=torch.Generator(device=DEVICE).manual_seed(1000+row),
                         num_inference_steps=30,height=512,width=512).images[0]
            image.save(path)
    canvas = Image.new('RGB',(384*len(conditions),len(indices)*424),'#eeeeee')
    draw = ImageDraw.Draw(canvas)
    for row in range(len(indices)):
        for col,(name,_) in enumerate(conditions):
            im = Image.open(OUT/f'{row:02d}_{name}.png').resize((384,384))
            canvas.paste(im,(384*col,row*424+40))
            draw.text((384*col+8,row*424+12),f'Prompt {row+1} | {name}',fill='black')
    canvas.save(OUT/'comparacao.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank',type=int,default=192)
    parser.add_argument('--epochs',type=int,default=70)
    parser.add_argument('--images',type=int,default=6)
    parser.add_argument('--fit-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(6)
    if not torch.cuda.is_available():
        raise RuntimeError('GPU indisponível.')
    print('GPU:',torch.cuda.get_device_name(0),flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    dataset = make_dataset()
    cache = collect(dataset)
    cp = train(dataset,cache,args)
    target,preds = evaluate(dataset,cache,cp)
    if not args.fit_only:
        render(dataset,cache,target,preds,args)
    print('Concluído:',OUT.resolve(),flush=True)


if __name__ == '__main__':
    main()
