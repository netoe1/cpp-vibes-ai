#pragma once

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace vibes {

// Tokenizador Byte-Level BPE (Byte Pair Encoding, estilo GPT-2) feito do zero.
//
// Como funciona:
//   - Os IDs 0..255 representam diretamente os 256 bytes possiveis (nunca ha token
//     desconhecido, e qualquer texto UTF-8 com acentos funciona sem perda).
//   - Durante o treino, encontramos repetidamente o par de tokens adjacentes mais
//     frequente (sem cruzar quebras de linha '\n') e fundimos em um novo token
//     256, 257, ..., ate atingir `tamanho_vocab_alvo`.
//   - Assim, subpalavras e palavras frequentes do portugues viram 1 unico token,
//     multiplicando o alcance efetivo da janela de contexto do Transformer!
class TokenizadorBPE {
public:
    TokenizadorBPE() { inicializar_base(); }

    // Treina as fusoes BPE a partir de um texto de exemplo.
    // `max_bytes_amostra` limita quantos bytes sao usados para descobrir fusoes
    // (200 KB ja bastam para capturar subpalavras do portugues em < 1 segundo).
    void treinar(std::string_view texto, std::size_t tamanho_vocab_alvo = 512,
                 std::size_t max_bytes_amostra = 200000) {
        inicializar_base();
        if (tamanho_vocab_alvo <= 256 || texto.empty()) {
            return;
        }

        const std::size_t num_fusoes = tamanho_vocab_alvo - 256;
        const std::size_t limite = std::min(texto.size(), max_bytes_amostra);

        std::vector<int> ids;
        ids.reserve(limite);
        for (std::size_t i = 0; i < limite; ++i) {
            ids.push_back(static_cast<unsigned char>(texto[i]));
        }

        constexpr int kByteQuebra = static_cast<int>('\n');

        for (std::size_t m = 0; m < num_fusoes; ++m) {
            if (ids.size() < 2) break;

            // Conta frequencia de cada par adjacente (evitando cruzar '\n')
            std::unordered_map<std::uint64_t, std::size_t> contagens;
            contagens.reserve(4096);
            for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
                const int a = ids[i];
                const int b = ids[i + 1];
                if (a == kByteQuebra || b == kByteQuebra) continue;
                const std::uint64_t chave = empacotar_par(a, b);
                ++contagens[chave];
            }

            if (contagens.empty()) break;

            // Escolhe o par mais frequente (desempate deterministico pela menor chave)
            std::uint64_t melhor_chave = 0;
            std::size_t melhor_freq = 0;
            for (const auto& [chave, freq] : contagens) {
                if (freq > melhor_freq || (freq == melhor_freq && chave < melhor_chave)) {
                    melhor_freq = freq;
                    melhor_chave = chave;
                }
            }
            if (melhor_freq < 2) break;

            const int a = static_cast<int>(melhor_chave >> 32);
            const int b = static_cast<int>(melhor_chave & 0xFFFFFFFFULL);
            const int novo_id = static_cast<int>(vocab_.size());

            registrar_fusao(a, b, novo_id);

            // Substitui todas as ocorrencias do par (a, b) por novo_id
            std::vector<int> proximos;
            proximos.reserve(ids.size());
            std::size_t i = 0;
            while (i < ids.size()) {
                if (i + 1 < ids.size() && ids[i] == a && ids[i + 1] == b) {
                    proximos.push_back(novo_id);
                    i += 2;
                } else {
                    proximos.push_back(ids[i]);
                    ++i;
                }
            }
            ids.swap(proximos);
        }
    }

    // Codifica texto UTF-8 em sequencia de IDs BPE.
    // Divide por linhas/segmentos para aplicar fusoes por menor rank rapidamente.
    std::vector<int> codificar(std::string_view texto) const {
        std::vector<int> saida;
        saida.reserve(texto.size() / 2 + 16);

        std::size_t inicio = 0;
        while (inicio < texto.size()) {
            std::size_t pos = texto.find('\n', inicio);
            const bool tem_quebra = (pos != std::string_view::npos);
            const std::size_t fim = tem_quebra ? pos : texto.size();

            if (fim > inicio) {
                codificar_segmento(texto.substr(inicio, fim - inicio), saida);
            }
            if (tem_quebra) {
                saida.push_back(static_cast<unsigned char>('\n'));
                inicio = pos + 1;
            } else {
                break;
            }
        }
        return saida;
    }

    // Decodifica sequencia de IDs BPE de volta para a string UTF-8 original.
    std::string decodificar(const std::vector<int>& ids) const {
        std::string saida;
        for (int id : ids) {
            if (id >= 0 && static_cast<std::size_t>(id) < vocab_.size()) {
                saida += vocab_[static_cast<std::size_t>(id)];
            }
        }
        return saida;
    }

    std::size_t tamanho_vocabulario() const { return vocab_.size(); }
    int id_quebra_linha() const { return static_cast<unsigned char>('\n'); }

    // Salva as regras de fusao em arquivo binario simples (.bpe).
    void salvar(const std::string& caminho) const {
        std::ofstream arq(caminho, std::ios::binary);
        if (!arq.is_open()) {
            throw std::runtime_error("Nao foi possivel salvar tokenizador BPE: " + caminho);
        }
        const char magico[8] = {'V', 'B', 'B', 'P', 'E', '0', '1', '\0'};
        arq.write(magico, 8);
        const std::uint64_t n = fusoes_.size();
        arq.write(reinterpret_cast<const char*>(&n), sizeof(n));
        for (const auto& [a, b] : fusoes_) {
            const std::int32_t pa = a;
            const std::int32_t pb = b;
            arq.write(reinterpret_cast<const char*>(&pa), sizeof(pa));
            arq.write(reinterpret_cast<const char*>(&pb), sizeof(pb));
        }
        if (!arq) {
            throw std::runtime_error("Erro ao gravar tokenizador BPE: " + caminho);
        }
    }

    // Carrega regras de fusao de arquivo binario (.bpe).
    void carregar(const std::string& caminho) {
        std::ifstream arq(caminho, std::ios::binary);
        if (!arq.is_open()) {
            throw std::runtime_error("Nao foi possivel carregar tokenizador BPE: " + caminho);
        }
        char magico[8]{};
        arq.read(magico, 8);
        if (std::string(magico, 7) != "VBBPE01") {
            throw std::runtime_error("Formato de tokenizador BPE invalido: " + caminho);
        }
        std::uint64_t n = 0;
        arq.read(reinterpret_cast<char*>(&n), sizeof(n));

        inicializar_base();
        for (std::uint64_t i = 0; i < n; ++i) {
            std::int32_t pa = 0, pb = 0;
            arq.read(reinterpret_cast<char*>(&pa), sizeof(pa));
            arq.read(reinterpret_cast<char*>(&pb), sizeof(pb));
            registrar_fusao(pa, pb, static_cast<int>(vocab_.size()));
        }
        if (!arq) {
            throw std::runtime_error("Arquivo de tokenizador BPE corrompido: " + caminho);
        }
    }

private:
    static std::uint64_t empacotar_par(int a, int b) {
        return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32) |
               static_cast<std::uint64_t>(static_cast<std::uint32_t>(b));
    }

    void inicializar_base() {
        vocab_.clear();
        vocab_.reserve(256);
        for (int b = 0; b < 256; ++b) {
            vocab_.emplace_back(1, static_cast<char>(b));
        }
        fusoes_.clear();
        par_para_novo_id_.clear();
    }

    void registrar_fusao(int a, int b, int novo_id) {
        fusoes_.emplace_back(a, b);
        par_para_novo_id_[empacotar_par(a, b)] = novo_id;
        vocab_.push_back(vocab_[static_cast<std::size_t>(a)] +
                         vocab_[static_cast<std::size_t>(b)]);
    }

    void codificar_segmento(std::string_view seg, std::vector<int>& destino) const {
        std::vector<int> ids;
        ids.reserve(seg.size());
        for (char c : seg) {
            ids.push_back(static_cast<unsigned char>(c));
        }

        if (fusoes_.empty() || ids.size() < 2) {
            destino.insert(destino.end(), ids.begin(), ids.end());
            return;
        }

        // Aplica iterativamente o par presente com menor ID (menor rank de fusao)
        while (ids.size() >= 2) {
            int melhor_id = std::numeric_limits<int>::max();
            int melhor_a = -1;
            int melhor_b = -1;

            for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
                const auto it = par_para_novo_id_.find(empacotar_par(ids[i], ids[i + 1]));
                if (it != par_para_novo_id_.end() && it->second < melhor_id) {
                    melhor_id = it->second;
                    melhor_a = ids[i];
                    melhor_b = ids[i + 1];
                }
            }

            if (melhor_a < 0) break;

            std::vector<int> proximos;
            proximos.reserve(ids.size());
            std::size_t i = 0;
            while (i < ids.size()) {
                if (i + 1 < ids.size() && ids[i] == melhor_a && ids[i + 1] == melhor_b) {
                    proximos.push_back(melhor_id);
                    i += 2;
                } else {
                    proximos.push_back(ids[i]);
                    ++i;
                }
            }
            ids.swap(proximos);
        }

        destino.insert(destino.end(), ids.begin(), ids.end());
    }

    std::vector<std::string> vocab_;
    std::vector<std::pair<int, int>> fusoes_;
    std::unordered_map<std::uint64_t, int> par_para_novo_id_;
};

}  // namespace vibes
