# EmbeddingGemma 2 + Tiny-SD: adaptador experimental

Código de experimentos que convertem um embedding global do EmbeddingGemma 2 em condicionamento para o modelo de difusão Tiny-SD. O codificador e o gerador ficam congelados; o treinamento ajusta apenas o adaptador.

O repositório contém apenas código e documentação de uso. Pesos, checkpoints, datasets, embeddings, legendas extraídas, resultados, imagens, relatórios, logs e ambientes locais não são distribuídos. A lista de arquivos permitidos está definida no `.gitignore`.

## Arquivos

- `experimento.py`: piloto com frases sintéticas e regressão ridge.
- `experimento_v2.py`: mais frases sintéticas e regularização escolhida em validação.
- `experimento_v3.py`: legendas COCO, projeção linear, PCA e MLP residual.
- `gerar_v3.py`: inferência com um checkpoint v3 treinado localmente.
- `requirements.txt`: dependências Python, sem URLs locais ou credenciais.

## Ambiente

Use Python 3.13 e uma GPU compatível com PyTorch. Os scripts usam a interface `torch.cuda`, também usada pelo backend ROCm da AMD. A primeira geração pode levar mais tempo por causa da preparação de kernels.

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

Instale primeiro PyTorch e torchvision para sua GPU seguindo o [guia oficial do PyTorch](https://pytorch.org/get-started/locally/). Em seguida:

```bash
python -m pip install -r requirements.txt
```

O experimento foi executado com PyTorch 2.11.0 e ROCm. A instalação de CUDA ou ROCm depende da máquina. O arquivo de dependências não fixa uma distribuição de GPU específica.

## Treinar a versão mais recente

Execute os comandos na raiz do repositório:

```bash
python experimento_v3.py --fit-only
```

Esse comando prepara os dados, treina o adaptador e grava as métricas. As comparações com v1 e v2 são opcionais e só são feitas se seus checkpoints existirem localmente. Não é necessário baixar ou executar as versões anteriores para treinar a v3.

Para também gerar a comparação visual:

```bash
python experimento_v3.py --images 6
```

Na primeira execução, o código baixa automaticamente modelos e legendas de fontes públicas:

- [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2);
- [`segmind/tiny-sd`](https://huggingface.co/segmind/tiny-sd);
- [`phiyodr/coco2017`](https://huggingface.co/datasets/phiyodr/coco2017), apenas os metadados e as legendas; as imagens COCO não são baixadas.

Esses downloads ficam no cache local do Hugging Face. Os artefatos de treino ficam em `resultados_v3/`, incluindo `adaptador.pt`, `dataset.json`, `representacoes.pt`, métricas e imagens. Todos ficam fora do Git.

## Gerar com o adaptador

Treine a v3 antes de usar a inferência. Nenhum checkpoint é incluído neste repositório.

```bash
python gerar_v3.py 'a red bicycle parked beside a brick wall' --seed 1002 --output resultados_v3/minha_imagem.png
```

A geração usa EmbeddingGemma 2 + adaptador. O CLIP original não é carregado na pipeline de inferência; seu condicionamento negativo vazio, calculado durante o preparo, é reutilizado a partir do checkpoint. O CLIP original é necessário durante o treinamento para fornecer os alvos.

## Método e avaliação

A v3 usa 16.000 legendas de treino, 512 de validação e 256 de teste, com uma legenda por imagem. As partições não compartilham IDs de imagem nem legendas normalizadas idênticas. A revisão do dataset utilizada é registrada localmente em `dataset.json`.

O alvo CLIP é uma sequência de 77 vetores com 768 valores cada. Uma PCA ajustada somente no treino reduz esse alvo para 192 componentes. Uma projeção ridge prevê os coeficientes; uma MLP residual com duas camadas ocultas de 1.024 unidades aprende uma correção. A regularização e a época são escolhidas com a validação.

A métrica principal é o erro quadrático médio de reconstrução dos estados CLIP. Ela não equivale a qualidade visual: objetos, cores, materiais e relações podem ser perdidos mesmo quando o erro diminui. Este é um protótipo experimental.

## Reexecutar ou mudar a configuração

Os scripts reutilizam representações e checkpoints existentes e pulam imagens já geradas. Para um novo experimento com outra configuração, use outro diretório de trabalho com os scripts, ou preserve os resultados atuais e altere o diretório `OUT` antes de executar. Não misture checkpoints de configurações diferentes.

Para reproduzir a sequência histórica completa, execute `experimento.py`, depois `experimento_v2.py` e por fim `experimento_v3.py`. A v2 depende do checkpoint local da v1.

## Conteúdo compartilhável

O histórico inicial foi criado do zero a partir dos sete arquivos listados no `.gitignore`. Não contém o histórico do ambiente de trabalho, caminhos pessoais, configurações de autenticação, dados de treino ou modelos. Para adicionar um novo arquivo ao repositório, é necessário incluí-lo explicitamente na lista permitida do `.gitignore`.
