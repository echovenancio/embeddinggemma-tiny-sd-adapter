# Método e reprodução

## Versão atual

O experimento parte de uma questão: até que ponto um vetor global do EmbeddingGemma 2 pode orientar um gerador treinado com representações do CLIP?

1. O EmbeddingGemma 2 produz um vetor global para cada prompt.
2. O CLIP do Tiny-SD produz a sequência de referência de 77 vetores de 768 valores.
3. Uma PCA ajustada apenas no conjunto de treino comprime os estados do CLIP em 192 componentes por padrão.
4. Uma projeção ridge prevê os coeficientes PCA a partir do vetor global. Uma MLP residual aprende uma correção.
5. O Tiny-SD recebe a sequência reconstruída para gerar a imagem.

O codificador de embeddings, o CLIP e o gerador não são ajustados. A MLP é treinada para aproximar as representações do CLIP, e não diretamente pelo resultado visual da difusão.

## Dados e partições

A versão atual usa legendas da distribuição [`phiyodr/coco2017`](https://huggingface.co/datasets/phiyodr/coco2017): 16.000 legendas de treino, 512 de validação e 256 de teste, com uma legenda por imagem. As partições locais não compartilham IDs de imagem nem legendas normalizadas idênticas. A revisão do dataset é registrada no arquivo local `resultados_v3/dataset.json`.

As outras dependências de dados são os checkpoints públicos [`google/embeddinggemma-2`](https://huggingface.co/google/embeddinggemma-2) e [`segmind/tiny-sd`](https://huggingface.co/segmind/tiny-sd). Os downloads usam o cache local do Hugging Face. O código não baixa as imagens COCO.

## Avaliação

O erro quadrático médio entre os estados originais do CLIP e as representações reconstruídas mede aproximação numérica. Ele não substitui a inspeção das imagens: cores, objetos, materiais e relações podem divergir mesmo com erro menor.

A v3 compara a projeção linear e a MLP residual no mesmo conjunto de teste. Se os checkpoints anteriores estiverem presentes, também calcula as métricas das versões v1 e v2. A geração visual usa os mesmos prompts e seeds entre condições.

## Caches e saídas locais

A versão atual salva partições, representações, checkpoint, curva de treino, métricas e imagens em `resultados_v3/`. Os scripts reutilizam caches e checkpoints existentes e pulam imagens já geradas. Para alterar a configuração sem misturar resultados, trabalhe em uma cópia separada ou preserve os arquivos anteriores e altere `OUT` em `src/embedding_adapter/train.py`.

O `.gitignore` usa uma lista explícita de caminhos permitidos; todo arquivo novo fica ignorado até ser incluído nessa lista. Isso evita adicionar artefatos gerados por acidente. A lista deve ser atualizada conscientemente ao adicionar novos arquivos de código ou documentação.
