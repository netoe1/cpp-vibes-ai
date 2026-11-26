#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <span>
#include <vector>

#ifdef USAR_OPENMP
#include <omp.h>
#endif

namespace vibes {

// Matriz 2D simples armazenada de forma contigua na memoria (row-major).
//
// Em C++, guardar os dados em um unico std::vector<float> (em vez de
// vector<vector<float>>) melhora muito o uso de cache da CPU e facilita
// a paralelizacao com OpenMP.
class Matriz {
public:
    Matriz() = default;

    Matriz(std::size_t linhas, std::size_t colunas, float valor_inicial = 0.0f)
        : linhas_(linhas), colunas_(colunas), dados_(linhas * colunas, valor_inicial) {}

    float& operator()(std::size_t linha, std::size_t coluna) {
        assert(linha < linhas_ && coluna < colunas_);
        return dados_[linha * colunas_ + coluna];
    }

    float operator()(std::size_t linha, std::size_t coluna) const {
        assert(linha < linhas_ && coluna < colunas_);
        return dados_[linha * colunas_ + coluna];
    }

    std::span<float> linha(std::size_t i) {
        assert(i < linhas_);
        return {dados_.data() + i * colunas_, colunas_};
    }

    std::span<const float> linha(std::size_t i) const {
        assert(i < linhas_);
        return {dados_.data() + i * colunas_, colunas_};
    }

    void preencher(float valor) { std::fill(dados_.begin(), dados_.end(), valor); }

    std::size_t linhas() const { return linhas_; }
    std::size_t colunas() const { return colunas_; }
    std::size_t total_elementos() const { return dados_.size(); }

    std::vector<float>& dados() { return dados_; }
    const std::vector<float>& dados() const { return dados_; }

private:
    std::size_t linhas_ = 0;
    std::size_t colunas_ = 0;
    std::vector<float> dados_;
};

// Aplica a funcao Softmax sobre um vetor de logits com estabilidade numerica.
//
// Formula: softmax(z_i) = exp((z_i - max_z) / T) / sum_j exp((z_j - max_z) / T)
//
// Subtrair o maior logit (max_z) evita overflow de exp() sem alterar o resultado
// matematico. A temperatura T > 0 controla o "foco" da distribuicao:
//   - T < 1.0: deixa o modelo mais confiante / previsivel
//   - T = 1.0: distribuicao original aprendida
//   - T > 1.0: deixa o modelo mais aleatorio / "criativo"
inline void softmax(std::span<const float> logits, std::span<float> saida_probs,
                    float temperatura = 1.0f) {
    assert(logits.size() == saida_probs.size());
    assert(temperatura > 0.0f);

    float maior = logits[0];
    for (float v : logits) {
        if (v > maior) {
            maior = v;
        }
    }

    float soma = 0.0f;
    for (std::size_t i = 0; i < logits.size(); ++i) {
        const float valor = std::exp((logits[i] - maior) / temperatura);
        saida_probs[i] = valor;
        soma += valor;
    }

    const float inv_soma = 1.0f / soma;
    for (float& p : saida_probs) {
        p *= inv_soma;
    }
}

// Multiplicacao de matrizes C = A * B (usada a partir da Etapa 2 - MLP).
// Paralelizada nas linhas de A usando OpenMP.
inline Matriz multiplicar(const Matriz& a, const Matriz& b) {
    assert(a.colunas() == b.linhas());
    Matriz c(a.linhas(), b.colunas(), 0.0f);

#ifdef USAR_OPENMP
#pragma omp parallel for schedule(static)
#endif
    for (std::size_t i = 0; i < a.linhas(); ++i) {
        for (std::size_t k = 0; k < a.colunas(); ++k) {
            const float aik = a(i, k);
            for (std::size_t j = 0; j < b.colunas(); ++j) {
                c(i, j) += aik * b(k, j);
            }
        }
    }
    return c;
}

}  // namespace vibes
