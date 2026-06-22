# Sistema de Recomendação de Filmes — Filtragem Colaborativa vs. iRSVD

Comparação entre duas abordagens de sistemas de recomendação aplicadas ao conjunto de dados **MovieLens 100k**: **Filtragem Colaborativa Baseada em Itens** (abordagem baseada em memória) e **Improved Regularized SVD — iRSVD** (abordagem baseada em modelo).
O projeto inclui implementação completa em **Julia**, ajuste de hiperparâmetros via validação cruzada.

## Resultados

| Modelo | Hiperparâmetros | RMSE |
|---|---|---|
| Item-based | k=20, β=0 | 0,9769 |
| **iRSVD** | k=3, λ=0,001, γ=0,001 | **0,9407** |

O iRSVD superou a filtragem colaborativa baseada em itens em aproximadamente **3,7%** de RMSE, confirmando a vantagem de abordagens baseadas em modelo (fatoração de matriz) sobre abordagens baseadas em memória neste conjunto de dados.

## Dataset

O projeto utiliza o **MovieLens 100k**, disponível em [grouplens.org/datasets/movielens](https://grouplens.org/datasets/movielens/100k/).

## Como executar

1. Clone o repositório:
   ```bash
   git clone https://github.com/fllpo/item-based-vs-iRSVD.git
   cd item-based-vs-iRSVD
   
   ```

2. Execute o pipeline desejado:
   ```bash
   julia metodologia_item-based.jl
   julia metodologia_iRSVD.jl
   ```
   
## Metodologia

- **Filtragem Colaborativa Baseada em Itens**: similaridade entre itens calculada a partir das avaliações dos usuários, com seleção dos *k* vizinhos mais próximos.
- **iRSVD**: fatoração de matriz com regularização, otimizada via descida de gradiente estocástico, com termos de viés (*bias*) por usuário e item.
- **Validação**: ajuste de hiperparâmetros (número de fatores latentes, taxas de regularização e aprendizado) via validação cruzada, com avaliação final por RMSE.
