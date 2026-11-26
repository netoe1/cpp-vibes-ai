#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace vibes {

// Tokenizador a nivel de caractere com suporte a UTF-8.
//
// Por que UTF-8?
// Em portugues temos acentos ("á", "ç", "ã") que ocupam 2 bytes em UTF-8.
// Se tratassemos a string byte a byte, um caractere acentuado seria quebrado em
// dois tokens. Aqui agrupamos os bytes de cada caractere UTF-8 em um unico token.
class TokenizadorUTF8 {
public:
    // Separa uma string UTF-8 em uma lista de caracteres (cada um ainda em UTF-8).
    static std::vector<std::string> dividir_utf8(std::string_view texto) {
        std::vector<std::string> caracteres;
        std::size_t i = 0;
        while (i < texto.size()) {
            const auto byte = static_cast<unsigned char>(texto[i]);
            std::size_t tamanho = 1;
            if ((byte & 0x80) == 0x00) {
                tamanho = 1;  // ASCII (0xxxxxxx)
            } else if ((byte & 0xE0) == 0xC0) {
                tamanho = 2;  // 2 bytes (110xxxxx) - acentos comuns em PT-BR
            } else if ((byte & 0xF0) == 0xE0) {
                tamanho = 3;  // 3 bytes (1110xxxx)
            } else if ((byte & 0xF8) == 0xF0) {
                tamanho = 4;  // 4 bytes (11110xxx) - emojis, etc.
            }
            if (i + tamanho > texto.size()) {
                tamanho = texto.size() - i;
            }
            caracteres.emplace_back(texto.substr(i, tamanho));
            i += tamanho;
        }
        return caracteres;
    }

    // Constroi o vocabulario a partir de um texto de treinamento.
    void construir_vocabulario(std::string_view texto) {
        const auto caracteres = dividir_utf8(texto);
        std::set<std::string> unicos(caracteres.begin(), caracteres.end());

        id_para_token_.clear();
        token_para_id_.clear();

        // Ordenado lexicograficamente (via std::set) para determinismo.
        for (const auto& ch : unicos) {
            const int id = static_cast<int>(id_para_token_.size());
            id_para_token_.push_back(ch);
            token_para_id_[ch] = id;
        }
    }

    // Converte texto em sequencia de IDs inteiros.
    // Caracteres fora do vocabulario sao ignorados (ou substituidos por espaco se existir).
    std::vector<int> codificar(std::string_view texto) const {
        const auto caracteres = dividir_utf8(texto);
        std::vector<int> ids;
        ids.reserve(caracteres.size());

        const auto it_espaco = token_para_id_.find(" ");
        for (const auto& ch : caracteres) {
            const auto it = token_para_id_.find(ch);
            if (it != token_para_id_.end()) {
                ids.push_back(it->second);
            } else if (it_espaco != token_para_id_.end()) {
                ids.push_back(it_espaco->second);
            }
        }
        return ids;
    }

    // Converte sequencia de IDs inteiros de volta para texto UTF-8.
    std::string decodificar(const std::vector<int>& ids) const {
        std::string saida;
        for (int id : ids) {
            if (id >= 0 && static_cast<std::size_t>(id) < id_para_token_.size()) {
                saida += id_para_token_[static_cast<std::size_t>(id)];
            }
        }
        return saida;
    }

    std::string decodificar_um(int id) const {
        if (id >= 0 && static_cast<std::size_t>(id) < id_para_token_.size()) {
            return id_para_token_[static_cast<std::size_t>(id)];
        }
        return "";
    }

    std::size_t tamanho_vocabulario() const { return id_para_token_.size(); }

    bool contem(std::string_view token) const {
        return token_para_id_.find(std::string(token)) != token_para_id_.end();
    }

    int obter_id(std::string_view token) const {
        const auto it = token_para_id_.find(std::string(token));
        return it == token_para_id_.end() ? -1 : it->second;
    }

private:
    std::vector<std::string> id_para_token_;
    std::unordered_map<std::string, int> token_para_id_;
};

}  // namespace vibes
