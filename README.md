# EmbeddingGemma 2 → Tiny-SD

Experimentos para usar um embedding global do **EmbeddingGemma 2** como condicionamento de texto de um modelo de difusão **Tiny-SD**. Um adaptador aprende a converter o vetor do prompt em representações que o gerador consiga usar.

O projeto é uma prova de conceito: uma representação mais próxima da saída do CLIP não garante que a imagem preserve todos os objetos, atributos e relações descritos no prompt.

## Como funciona

```text
Prompt → EmbeddingGemma 2 → adaptador → Tiny-SD → imagem
                                 ↑
                 supervisionado pelo CLIP durante o treino
```

Na versão atual, o adaptador combina uma projeção linear, uma base PCA e uma MLP residual. EmbeddingGemma, CLIP e Tiny-SD permanecem congelados. Após o treino, a geração avulsa não carrega o CLIP: ela reutiliza um condicionamento negativo vazio salvo no checkpoint.

## Estrutura

```text
src/embedding_adapter/  Treinamento e geração da versão atual
experiments/            Experimentos históricos v1 e v2
docs/                   Método, partições e reprodução
tests/                  Verificações locais do fluxo
pyproject.toml          Pacote, dependências e comandos
LICENSE                 Licença MIT do código
```

O repositório contém código e documentação. Pesos, checkpoints, datasets, embeddings, imagens e resultados são obtidos ou gerados localmente e ficam fora do Git.

## Instalação

Requer Python 3.13 ou superior e uma GPU acessível pelo PyTorch. A execução foi testada com uma GPU AMD e ROCm; os scripts usam a interface `torch.cuda`, também utilizada pelo PyTorch com ROCm.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Instale primeiro PyTorch e torchvision adequados à sua GPU, seguindo o [guia oficial do PyTorch](https://pytorch.org/get-started/locally/). Depois, instale o projeto:

```bash
python -m pip install -e .
```

## Treinar

Execute na raiz do repositório:

```bash
embedding-adapter-train --fit-only
```

O comando prepara os dados, treina o adaptador e calcula as métricas. Para também gerar imagens de comparação:

```bash
embedding-adapter-train --images 6
```

Na primeira execução, são baixados modelos do Hugging Face e as legendas e metadados do COCO2017. As imagens do COCO não são baixadas. Os arquivos produzidos ficam em `resultados_v3/` e são ignorados pelo Git.

## Gerar uma imagem

Depois de treinar a versão atual:

```bash
embedding-adapter-generate 'a red bicycle parked beside a brick wall' \
  --seed 1002 --output resultados_v3/minha_imagem.png
```

O comando espera o checkpoint local em `resultados_v3/adaptador.pt`. Nenhum checkpoint é distribuído neste repositório.

## Reproduzir as versões históricas

A v2 usa o checkpoint local produzido pela v1. Para executar as duas versões na ordem:

```bash
python experiments/v1.py
python experiments/v2.py
```

A versão atual funciona por conta própria. Se os checkpoints v1 e v2 existirem localmente, o treinamento também os inclui na avaliação. Veja [método e reprodução](docs/experiments.md) para detalhes sobre dados, métricas e caches.

## Testes

Com o projeto e suas dependências instalados:

```bash
python -m unittest discover -s tests -v
```

Os testes não baixam modelos, datasets ou pesos.

## Licença

O código deste repositório está disponível sob a [licença MIT](LICENSE). Os modelos e o dataset externos possuem termos próprios e são obtidos separadamente.
