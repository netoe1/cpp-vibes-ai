# Etapa 1 — Modelo Bigram: Do Estatístico à Rede Neural

Nesta primeira etapa, construímos o modelo de linguagem mais simples possível: um **Bigram**.

## 1. O que é um Modelo de Linguagem?

Por trás de qualquer IA conversacional existe uma tarefa surpreendentemente simples:

> Dado o texto até agora, **qual é a probabilidade de cada token ser o próximo?**

No modelo **Bigram**, simplificamos ao extremo: para prever o próximo caractere $y$, olhamos **apenas** o caractere imediatamente anterior $x$:

$$P(y \mid x)$$

## 2. Tokenização UTF-8

Computadores operam sobre números, não letras. O [tokenizador.hpp](../src/comum/tokenizador.hpp) percorre o arquivo `dados/dialogos.txt`, coleta todos os caracteres UTF-8 únicos (o **vocabulário** de tamanho $V$) e atribui um número inteiro de $0$ até $V - 1$ a cada um.

Usamos leitura compatível com UTF-8 para que caracteres acentuados do português (`á`, `ç`, `ã`, etc.) sejam tratados como **um único token**, mesmo ocupando 2 bytes em memória.

## 3. A Ponte entre Estatística e Rede Neural

No arquivo [main.cpp](../src/etapa1_bigram/main.cpp), resolvemos o mesmo problema de **duas formas**:

1. **Forma Estatística (Contagem)**: contamos quantas vezes cada par $(x, y)$ ocorre no texto e dividimos pelo total de vezes que $x$ aparece.
2. **Forma Neural (Gradiente Descendente)**: criamos uma matriz de pesos $W$ de tamanho $V \times V$ inicializada com números aleatórios.
   - A linha $W[x]$ contém os **logits** (pontuações brutas) para cada possível próximo token.
   - Aplicamos a função **Softmax** para transformar os logits em probabilidades que somam $1$:

$$p_j = \frac{e^{W_{x,j}}}{\sum_{k=0}^{V-1} e^{W_{x,k}}}$$

## 4. Função de Perda (Entropia Cruzada)

Como sabemos se o modelo está errando muito ou pouco? Medimos a **Entropia Cruzada** (*Cross-Entropy Loss*, também chamada de *Negative Log-Likelihood*):

$$\mathcal{L}(x, y) = -\ln(p_y)$$

- Se o modelo atribui probabilidade $p_y = 1{,}0$ ao caractere correto, a perda é $-\ln(1) = 0$.
- Se atribui probabilidade baixa (ex.: $0{,}01$), a perda explode ($-\ln(0{,}01) \approx 4{,}6$).
- No início (pesos aleatórios), a perda fica próxima de $\ln(V)$.

## 5. O Gradiente Analítico (Sem Mágica!)

Para reduzir a perda, precisamos saber como cada peso $W_{x,j}$ influencia $\mathcal{L}$. Derivando $\mathcal{L} = -\ln(\text{softmax}(W_x)_y)$ em relação aos logits da linha $x$, obtemos uma fórmula elegante:

$$\frac{\partial \mathcal{L}}{\partial W_{x,j}} = \begin{cases} p_j - 1 & \text{se } j = y \text{ (token correto)} \\ p_j & \text{se } j \neq y \text{ (outros tokens)} \end{cases}$$

A intuição é direta:
- Para o token correto ($j = y$), o gradiente é **negativo**, então ao subtrair $\eta \cdot \text{grad}$, o peso **aumenta**.
- Para os tokens errados ($j \neq y$), o gradiente é **positivo**, então os pesos **diminuem**.

Após algumas centenas de épocas, **a rede neural converge para exatamente a mesma perda da tabela estatística!** Isso prova que o gradiente descendente realmente encontrou o ótimo global desse modelo.

## 6. Paralelismo com OpenMP

No laço de treino, cada par $(x_i, y_i)$ é independente dentro de uma época. Usando `#pragma omp parallel`, cada thread da CPU cria sua própria matriz de gradientes local (`grad_local`), processa sua fatia de tokens sem travar as outras threads e, ao final, soma tudo em `gradiente_total` dentro de uma pequena seção crítica (`#pragma omp critical`).

## 7. Experimento Sugerido

Depois de rodar `./build/etapa1_bigram dados/dialogos.txt`, teste no chat:
- `/temp 0.2` — o texto fica repetitivo e preso nas transições mais frequentes.
- `/temp 1.5` — o texto fica caótico e inventa combinações improváveis.
- Observe **por que** o Bigram não consegue responder perguntas: ele não lembra o que você escreveu antes do último caractere. Esse é exatamente o gancho para a **Etapa 2 (MLP)**!
