#include "comum/aleatorio.hpp"
#include "comum/bpe.hpp"
#include "comum/matriz.hpp"
#include "comum/tokenizador.hpp"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <string>
#include <vector>

namespace {

int falhas = 0;

void verificar(bool condicao, const std::string& nome) {
    if (!condicao) {
        std::cerr << "[FALHA] " << nome << "\n";
        ++falhas;
    } else {
        std::cout << "[OK] " << nome << "\n";
    }
}

void testar_tokenizador_utf8() {
    vibes::TokenizadorUTF8 tok;
    const std::string texto = "olá, ação! coração C++";
    tok.construir_vocabulario(texto);

    const auto caracteres = vibes::TokenizadorUTF8::dividir_utf8("ação");
    verificar(caracteres.size() == 4, "UTF-8 divide 'ação' em exatamente 4 caracteres");
    verificar(caracteres[1] == "ç" && caracteres[2] == "ã",
              "UTF-8 preserva caracteres multibyte 'ç' e 'ã'");

    const auto ids = tok.codificar(texto);
    const auto reconstruido = tok.decodificar(ids);
    verificar(reconstruido == texto, "Codificar e decodificar preservam texto UTF-8 exato");

    // Determinismo do vocabulario
    vibes::TokenizadorUTF8 tok2;
    tok2.construir_vocabulario(texto);
    verificar(tok.codificar(texto) == tok2.codificar(texto),
              "Tokenizador produz IDs deterministicos");
}

void testar_softmax_estabilidade() {
    // Valores grandes testam se a subtracao do maximo evita overflow em exp()
    const std::vector<float> logits = {1000.0f, 1001.0f, 1002.0f};
    std::vector<float> probs(3, 0.0f);
    vibes::softmax(logits, probs, 1.0f);

    float soma = 0.0f;
    bool finito = true;
    for (float p : probs) {
        soma += p;
        if (!std::isfinite(p) || p <= 0.0f) {
            finito = false;
        }
    }
    verificar(finito, "Softmax numericamente estavel para logits grandes (1000..1002)");
    verificar(std::abs(soma - 1.0f) < 1e-5f, "Softmax soma aproximadamente 1.0");
    verificar(probs[2] > probs[1] && probs[1] > probs[0],
              "Softmax preserva a ordenacao dos logits");

    // Temperatura baixa deve concentrar mais massa no maior logit
    std::vector<float> probs_frio(3, 0.0f);
    vibes::softmax(logits, probs_frio, 0.2f);
    verificar(probs_frio[2] > probs[2], "Temperatura menor aumenta confianca no maior logit");
}

void testar_multiplicacao_matrizes() {
    vibes::Matriz a(2, 3);
    a(0, 0) = 1; a(0, 1) = 2; a(0, 2) = 3;
    a(1, 0) = 4; a(1, 1) = 5; a(1, 2) = 6;

    vibes::Matriz b(3, 2);
    b(0, 0) = 7;  b(0, 1) = 8;
    b(1, 0) = 9;  b(1, 1) = 10;
    b(2, 0) = 11; b(2, 1) = 12;

    const auto c = vibes::multiplicar(a, b);
    verificar(c.linhas() == 2 && c.colunas() == 2, "Dimensao correta em C = A * B");
    verificar(std::abs(c(0, 0) - 58.0f) < 1e-5f && std::abs(c(0, 1) - 64.0f) < 1e-5f &&
                  std::abs(c(1, 0) - 139.0f) < 1e-5f && std::abs(c(1, 1) - 154.0f) < 1e-5f,
              "Valores corretos na multiplicacao de matrizes (OpenMP)");
}

void testar_aleatorio_deterministico() {
    vibes::GeradorAleatorio rng1(123);
    vibes::GeradorAleatorio rng2(123);
    bool iguais = true;
    for (int i = 0; i < 20; ++i) {
        if (rng1.normal() != rng2.normal()) {
            iguais = false;
            break;
        }
    }
    verificar(iguais, "GeradorAleatorio com mesma seed produz sequencia identica");

    // Top-K = 1 deve sempre retornar o indice de maior probabilidade
    const std::vector<float> probs = {0.05f, 0.15f, 0.70f, 0.10f};
    bool sempre_max = true;
    for (int i = 0; i < 20; ++i) {
        if (rng1.amostrar_top_k(probs, 1) != 2) {
            sempre_max = false;
            break;
        }
    }
    verificar(sempre_max, "amostrar_top_k com k=1 seleciona sempre o token mais provavel");
}

void testar_tokenizador_bpe() {
    const std::string corpus =
        "Usuario: olá, como funciona um transformer?\n"
        "IA: um transformer usa atenção para entender o contexto da frase.\n"
        "Usuario: o que é atenção no transformer?\n"
        "IA: atenção conecta tokens distantes no transformer!\n";

    vibes::TokenizadorBPE bpe;
    bpe.treinar(corpus, 300);

    verificar(bpe.tamanho_vocabulario() > 256 && bpe.tamanho_vocabulario() <= 300,
              "BPE aprendeu fusoes de subpalavras (vocab > 256)");

    const auto ids = bpe.codificar(corpus);
    verificar(ids.size() < corpus.size(),
              "BPE comprime o texto em menos tokens do que o total de bytes");
    verificar(bpe.decodificar(ids) == corpus,
              "BPE codifica e decodifica texto UTF-8 com acentos sem nenhuma perda");

    // Testar salvar e carregar regras BPE
    const std::string caminho_tmp = "/tmp/teste_cpp_vibes.bpe";
    bpe.salvar(caminho_tmp);
    vibes::TokenizadorBPE bpe2;
    bpe2.carregar(caminho_tmp);
    std::remove(caminho_tmp.c_str());

    verificar(bpe2.codificar(corpus) == ids,
              "Tokenizador BPE salvo e recarregado produz exatamente os mesmos IDs");
}

}  // namespace

int main() {
    testar_tokenizador_utf8();
    testar_softmax_estabilidade();
    testar_multiplicacao_matrizes();
    testar_aleatorio_deterministico();
    testar_tokenizador_bpe();

    if (falhas > 0) {
        std::cerr << "\nTotal de falhas: " << falhas << "\n";
        return 1;
    }
    std::cout << "\nTodos os testes de src/comum/ passaram!\n";
    return 0;
}
