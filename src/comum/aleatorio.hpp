#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <random>
#include <span>
#include <vector>

namespace vibes {

// Gerador de numeros pseudoaleatorios com semente fixa.
// Em Machine Learning, fixar a semente (seed) e fundamental para reproduzir
// experimentos e comparar mudancas de hiperparametros de forma justa.
class GeradorAleatorio {
public:
    explicit GeradorAleatorio(std::uint32_t semente = 42) : motor_(semente) {}

    // Gera numero com distribuicao normal N(media, desvio_padrao).
    // Muito usado para inicializar pesos de redes neurais.
    float normal(float media = 0.0f, float desvio_padrao = 1.0f) {
        std::normal_distribution<float> dist(media, desvio_padrao);
        return dist(motor_);
    }

    // Gera numero uniforme no intervalo [minimo, maximo).
    float uniforme(float minimo = 0.0f, float maximo = 1.0f) {
        std::uniform_real_distribution<float> dist(minimo, maximo);
        return dist(motor_);
    }

    // Amostra um indice a partir de um vetor de probabilidades que somam ~1.0.
    // Exemplo: se probs = {0.1, 0.7, 0.2}, o indice 1 tem 70% de chance de sair.
    std::size_t amostrar_categorica(std::span<const float> probabilidades) {
        const float r = uniforme(0.0f, 1.0f);
        float acumulado = 0.0f;
        for (std::size_t i = 0; i < probabilidades.size(); ++i) {
            acumulado += probabilidades[i];
            if (r <= acumulado) {
                return i;
            }
        }
        return probabilidades.empty() ? 0 : probabilidades.size() - 1;
    }

    // Amostragem Top-K: mantem apenas os K tokens mais provaveis, renormaliza
    // suas probabilidades e sorteia entre eles. Evita tokens muito improvaiveis
    // sem perder variedade. Se k == 0 ou k >= N, equivale a amostragem normal.
    std::size_t amostrar_top_k(std::span<const float> probabilidades, std::size_t k) {
        const std::size_t n = probabilidades.size();
        if (n == 0) {
            return 0;
        }
        if (k == 0 || k >= n) {
            return amostrar_categorica(probabilidades);
        }

        std::vector<std::size_t> indices(n);
        for (std::size_t i = 0; i < n; ++i) {
            indices[i] = i;
        }
        std::partial_sort(indices.begin(),
                          indices.begin() + static_cast<std::ptrdiff_t>(k),
                          indices.end(), [&](std::size_t a, std::size_t b) {
                              return probabilidades[a] > probabilidades[b];
                          });

        float soma = 0.0f;
        for (std::size_t i = 0; i < k; ++i) {
            soma += probabilidades[indices[i]];
        }
        if (soma <= 0.0f) {
            return indices[0];
        }

        const float r = uniforme(0.0f, soma);
        float acumulado = 0.0f;
        for (std::size_t i = 0; i < k; ++i) {
            acumulado += probabilidades[indices[i]];
            if (r <= acumulado) {
                return indices[i];
            }
        }
        return indices[k - 1];
    }

    std::mt19937& motor() { return motor_; }

private:
    std::mt19937 motor_;
};

}  // namespace vibes
