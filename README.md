# EmbeddingGemma 2 + Tiny-SD

Experimento para usar um embedding global do [EmbeddingGemma 2](https://huggingface.co/google/embeddinggemma-2) como condicionamento de texto para o [Tiny-SD](https://huggingface.co/segmind/tiny-sd). Um adaptador aprende a aproximar as representações do CLIP usado pelo gerador.

```text
prompt → EmbeddingGemma 2 → adaptador → Tiny-SD → imagem
                                 ↑
                      CLIP original (treino)
```

O adaptador combina projeção ridge, PCA e uma MLP residual. O EmbeddingGemma, o CLIP e o Tiny-SD ficam congelados. A métrica de treino compara representações: um erro menor não garante que a imagem preserve todos os detalhes do prompt.

## Instalação

Requer Python 3.13 ou superior e uma GPU compatível com PyTorch. O experimento foi executado com ROCm; os scripts usam a interface `torch.cuda`, compartilhada pelo PyTorch com CUDA e ROCm.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Instale [PyTorch e torchvision](https://pytorch.org/get-started/locally/) para sua GPU e depois instale o projeto:

```bash
python -m pip install -e .
```

## Treinamento

Na raiz do repositório:

```bash
embedding-adapter-train --fit-only
```

O comando baixa os modelos e as legendas do [COCO2017](https://huggingface.co/datasets/phiyodr/coco2017), prepara os dados, treina o adaptador e calcula métricas. As imagens do COCO não são baixadas. Para gerar também as comparações visuais:

```bash
embedding-adapter-train --images 6
```

Os resultados e checkpoints ficam em `resultados_v3/`. Como o código reutiliza caches e imagens existentes, use um diretório de trabalho separado para uma configuração nova. Consulte [método e reprodução](docs/experiments.md) para detalhes sobre os dados, as partições e os caches.

## Geração

Após o treinamento:

```bash
embedding-adapter-generate 'a red bicycle parked beside a brick wall' \
  --seed 1002 --output resultados_v3/minha_imagem.png
```

O comando usa `resultados_v3/adaptador.pt`. O CLIP original não é carregado durante a geração: o checkpoint contém o condicionamento negativo vazio calculado no preparo.

## Estrutura

```text
src/embedding_adapter/  Treinamento e geração
experiments/            Versões anteriores do experimento
docs/                   Método e reprodução
tests/                  Testes sem GPU ou downloads
```

As versões anteriores podem ser executadas com `python experiments/v1.py` e depois `python experiments/v2.py`; a v2 depende do checkpoint da v1. A versão atual não depende deles.

## Testes

```bash
python -m unittest discover -s tests -v
```

## Licença

O código está sob a [licença MIT](LICENSE). Os modelos e o dataset externos têm termos próprios.
