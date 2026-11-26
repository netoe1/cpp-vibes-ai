#include "comum/bpe.hpp"
#include "etapa3_transformer/transformer.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#ifdef USAR_OPENMP
#include <omp.h>
#endif

namespace {

std::string ler_arquivo(const std::string& caminho) {
    std::ifstream arquivo(caminho);
    if (!arquivo.is_open()) {
        throw std::runtime_error("Nao foi possivel abrir o arquivo: " + caminho);
    }
    std::ostringstream buffer;
    buffer << arquivo.rdbuf();
    return buffer.str();
}

// Interface simples para alternar entre TokenizadorUTF8 (caractere) e TokenizadorBPE (subpalavras).
struct TokenizadorUnificado {
    bool usar_bpe = false;
    vibes::TokenizadorUTF8 utf8;
    vibes::TokenizadorBPE bpe;

    std::size_t tamanho_vocabulario() const {
        return usar_bpe ? bpe.tamanho_vocabulario() : utf8.tamanho_vocabulario();
    }

    std::vector<int> codificar(std::string_view texto) const {
        return usar_bpe ? bpe.codificar(texto) : utf8.codificar(texto);
    }

    std::string decodificar(const std::vector<int>& ids) const {
        return usar_bpe ? bpe.decodificar(ids) : utf8.decodificar(ids);
    }

    int id_quebra_linha() const {
        return usar_bpe ? bpe.id_quebra_linha() : utf8.obter_id("\n");
    }
};

// Constroi a sequencia de tokens e, se `modo_sft` estiver ativo, um vetor paralelo
// `mascara_tokens` (1.0 nos tokens das respostas apos "IA: ", 0.0 nos prompts do Usuario).
void codificar_corpus_com_sft(const std::string& texto, const TokenizadorUnificado& tok,
                              bool modo_sft, std::vector<int>& tokens,
                              std::vector<float>& mascara_tokens) {
    if (!modo_sft) {
        tokens = tok.codificar(texto);
        mascara_tokens.clear();
        return;
    }

    tokens.clear();
    mascara_tokens.clear();
    std::istringstream iss(texto);
    std::string linha;
    const std::string prefixo_ia = "IA: ";

    while (std::getline(iss, linha)) {
        const bool e_resposta = (linha.rfind(prefixo_ia, 0) == 0);
        if (e_resposta) {
            const auto ids_pref = tok.codificar(prefixo_ia);
            const auto ids_resp = tok.codificar(linha.substr(prefixo_ia.size()) + "\n");
            tokens.insert(tokens.end(), ids_pref.begin(), ids_pref.end());
            mascara_tokens.insert(mascara_tokens.end(), ids_pref.size(), 0.0f);
            tokens.insert(tokens.end(), ids_resp.begin(), ids_resp.end());
            mascara_tokens.insert(mascara_tokens.end(), ids_resp.size(), 1.0f);
        } else {
            const auto ids_linha = tok.codificar(linha + "\n");
            tokens.insert(tokens.end(), ids_linha.begin(), ids_linha.end());
            mascara_tokens.insert(mascara_tokens.end(), ids_linha.size(), 0.0f);
        }
    }
}

// Amostra um mini-batch aleatorio de janelas de comprimento T (respeitando mascara SFT).
std::vector<vibes::SequenciaTreino> amostrar_batch(const std::vector<int>& tokens,
                                                   const std::vector<float>& mascara_tokens,
                                                   std::size_t tamanho_batch,
                                                   std::size_t comprimento_seq,
                                                   vibes::GeradorAleatorio& rng) {
    std::vector<vibes::SequenciaTreino> batch;
    if (tokens.size() < 2) return batch;
    const std::size_t T = std::min(comprimento_seq, tokens.size() - 1);
    const std::size_t max_inicio = tokens.size() - T - 1;
    const bool tem_mascara = (mascara_tokens.size() == tokens.size());

    std::uniform_int_distribution<std::size_t> dist(0, max_inicio);
    batch.reserve(tamanho_batch);
    for (std::size_t b = 0; b < tamanho_batch; ++b) {
        std::size_t ini = dist(rng.motor());
        // Em modo SFT, tenta escolher uma janela que contenha pelo menos 1 token de resposta
        if (tem_mascara) {
            for (int tent = 0; tent < 8; ++tent) {
                float soma = 0.0f;
                for (std::size_t t = 0; t < T; ++t) soma += mascara_tokens[ini + 1 + t];
                if (soma > 0.0f) break;
                ini = dist(rng.motor());
            }
        }
        vibes::SequenciaTreino seq;
        seq.entrada.assign(tokens.begin() + static_cast<std::ptrdiff_t>(ini),
                           tokens.begin() + static_cast<std::ptrdiff_t>(ini + T));
        seq.alvo.assign(tokens.begin() + static_cast<std::ptrdiff_t>(ini + 1),
                        tokens.begin() + static_cast<std::ptrdiff_t>(ini + T + 1));
        if (tem_mascara) {
            seq.mascara.assign(mascara_tokens.begin() + static_cast<std::ptrdiff_t>(ini + 1),
                               mascara_tokens.begin() + static_cast<std::ptrdiff_t>(ini + T + 1));
        }
        batch.push_back(std::move(seq));
    }
    return batch;
}

// Constroi ate `max_janelas` janelas deterministas espacadas ao longo do split.
std::vector<vibes::SequenciaTreino> construir_janelas_avaliacao(
    const std::vector<int>& tokens, const std::vector<float>& mascara_tokens,
    std::size_t comprimento_seq, std::size_t max_janelas = 32) {
    std::vector<vibes::SequenciaTreino> seqs;
    if (tokens.size() < 2) return seqs;
    const std::size_t T = std::min(comprimento_seq, tokens.size() - 1);
    const std::size_t total_possiveis = (tokens.size() - 1) / T;
    if (total_possiveis == 0) return seqs;

    const bool tem_mascara = (mascara_tokens.size() == tokens.size());
    const std::size_t passo_janelas =
        std::max<std::size_t>(1, total_possiveis / std::max<std::size_t>(1, max_janelas));

    for (std::size_t j = 0; j < total_possiveis && seqs.size() < max_janelas; j += passo_janelas) {
        const std::size_t ini = j * T;
        if (ini + T >= tokens.size()) break;
        vibes::SequenciaTreino seq;
        seq.entrada.assign(tokens.begin() + static_cast<std::ptrdiff_t>(ini),
                           tokens.begin() + static_cast<std::ptrdiff_t>(ini + T));
        seq.alvo.assign(tokens.begin() + static_cast<std::ptrdiff_t>(ini + 1),
                        tokens.begin() + static_cast<std::ptrdiff_t>(ini + T + 1));
        if (tem_mascara) {
            seq.mascara.assign(mascara_tokens.begin() + static_cast<std::ptrdiff_t>(ini + 1),
                               mascara_tokens.begin() + static_cast<std::ptrdiff_t>(ini + T + 1));
        }
        seqs.push_back(std::move(seq));
    }
    return seqs;
}

// Extrai as perguntas ("Usuario: ...") presentes no corpus para o comando /temas do chat.
std::vector<std::string> extrair_perguntas_corpus(const std::string& texto) {
    std::vector<std::string> perguntas;
    std::istringstream iss(texto);
    std::string linha;
    const std::string pref = "Usuario: ";
    while (std::getline(iss, linha)) {
        if (linha.rfind(pref, 0) == 0) {
            std::string p = linha.substr(pref.size());
            if (!p.empty()) {
                perguntas.push_back(std::move(p));
            }
        }
    }
    return perguntas;
}

std::string para_minusculas_ascii(std::string s) {
    for (char& c : s) {
        if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
    }
    return s;
}

}  // namespace

int main(int argc, char* argv[]) {
    std::string caminho_dados = "dados/dialogos.txt";
    std::string caminho_salvar;
    std::string caminho_carregar;
    std::string caminho_continuar;
    bool modo_chat = true;
    bool rodar_grad_check = false;
    bool modo_sft = false;
    bool salvar_melhor_val = false;
    int paciencia = 0;  // 0 = desativado; >0 = para apos N avaliacoes sem melhora na validacao
    std::size_t vocab_bpe = 0;  // 0 = caractere UTF-8; > 256 = Byte-Level BPE
    int total_passos = 250;
    std::size_t tamanho_batch = 16;
    float lr_max = 3e-3f;

    vibes::ConfigTransformer cfg;
    cfg.contexto_max = 64;
    cfg.dim_modelo = 64;
    cfg.num_cabecas = 4;
    cfg.num_camadas = 2;
    cfg.dim_ff = 256;
    cfg.weight_tying = true;
    cfg.dropout = 0.1f;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--sem-chat") {
            modo_chat = false;
        } else if (arg == "--verificar-gradiente") {
            rodar_grad_check = true;
        } else if (arg == "--sft") {
            modo_sft = true;
        } else if (arg == "--melhor-val") {
            salvar_melhor_val = true;
        } else if (arg == "--paciencia" && i + 1 < argc) {
            paciencia = std::stoi(argv[++i]);
        } else if (arg == "--dropout" && i + 1 < argc) {
            cfg.dropout = std::stof(argv[++i]);
        } else if (arg == "--bpe" && i + 1 < argc) {
            vocab_bpe = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--passos" && i + 1 < argc) {
            total_passos = std::stoi(argv[++i]);
        } else if (arg == "--dim" && i + 1 < argc) {
            cfg.dim_modelo = static_cast<std::size_t>(std::stoul(argv[++i]));
            cfg.dim_ff = 4 * cfg.dim_modelo;
        } else if (arg == "--camadas" && i + 1 < argc) {
            cfg.num_camadas = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--cabecas" && i + 1 < argc) {
            cfg.num_cabecas = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--contexto" && i + 1 < argc) {
            cfg.contexto_max = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--batch" && i + 1 < argc) {
            tamanho_batch = static_cast<std::size_t>(std::stoul(argv[++i]));
        } else if (arg == "--lr" && i + 1 < argc) {
            lr_max = std::stof(argv[++i]);
        } else if (arg == "--salvar" && i + 1 < argc) {
            caminho_salvar = argv[++i];
        } else if (arg == "--carregar" && i + 1 < argc) {
            caminho_carregar = argv[++i];
        } else if (arg == "--continuar" && i + 1 < argc) {
            caminho_continuar = argv[++i];
        } else {
            caminho_dados = arg;
        }
    }

    std::cout << "==============================================================\n";
    std::cout << "  cpp-vibes-ai | Etapa 3: Mini-Transformer Conversacional (GPT)\n";
    std::cout << "==============================================================\n";

#ifdef USAR_OPENMP
    std::cout << "[Info] OpenMP ativo com " << omp_get_max_threads() << " threads.\n";
#else
    std::cout << "[Info] Executando em modo single-thread.\n";
#endif

    // 1. Ler corpus e preparar tokenizador (UTF-8 ou BPE)
    std::string texto;
    try {
        texto = ler_arquivo(caminho_dados);
    } catch (const std::exception& e) {
        std::cerr << "[Erro] " << e.what() << "\n";
        return 1;
    }

    TokenizadorUnificado tokenizador;
    const std::string ckpt_origem = !caminho_carregar.empty() ? caminho_carregar : caminho_continuar;

    if (vocab_bpe > 256) {
        tokenizador.usar_bpe = true;
        const std::string caminho_bpe = ckpt_origem.empty() ? "" : (ckpt_origem + ".bpe");
        std::ifstream teste_bpe(caminho_bpe);
        if (!caminho_bpe.empty() && teste_bpe.is_open()) {
            teste_bpe.close();
            tokenizador.bpe.carregar(caminho_bpe);
            std::cout << "[Tokenizador] BPE carregado de: " << caminho_bpe
                      << " (vocab=" << tokenizador.tamanho_vocabulario() << ")\n";
        } else {
            std::cout << "[Tokenizador] Treinando Byte-Level BPE (vocab alvo=" << vocab_bpe
                      << ")...\n";
            tokenizador.bpe.treinar(texto, vocab_bpe);
            std::cout << "[Tokenizador] BPE pronto (vocab=" << tokenizador.tamanho_vocabulario()
                      << ")\n";
        }
    } else {
        tokenizador.utf8.construir_vocabulario(texto);
    }

    std::vector<int> tokens;
    std::vector<float> mascara_tokens;
    codificar_corpus_com_sft(texto, tokenizador, modo_sft, tokens, mascara_tokens);

    if (tokens.size() < 16) {
        std::cerr << "[Erro] Corpus pequeno demais para treinar o Transformer.\n";
        return 1;
    }

    const std::size_t corte = (tokens.size() * 9) / 10;
    const std::vector<int> tokens_treino(tokens.begin(),
                                         tokens.begin() + static_cast<std::ptrdiff_t>(corte));
    const std::vector<int> tokens_val(tokens.begin() + static_cast<std::ptrdiff_t>(corte),
                                      tokens.end());

    std::vector<float> mascara_treino, mascara_val;
    if (modo_sft) {
        mascara_treino.assign(mascara_tokens.begin(),
                              mascara_tokens.begin() + static_cast<std::ptrdiff_t>(corte));
        mascara_val.assign(mascara_tokens.begin() + static_cast<std::ptrdiff_t>(corte),
                           mascara_tokens.end());
    }

    cfg.vocab = tokenizador.tamanho_vocabulario();
    vibes::GeradorAleatorio rng(42);
    vibes::ModeloTransformer modelo(cfg, rng);

    std::cout << "[Dados] Arquivo: " << caminho_dados << " | Tokens: " << tokens.size()
              << " (Treino: " << tokens_treino.size() << ", Val: " << tokens_val.size() << ")"
              << (modo_sft ? " [SFT: mascara apenas na IA]" : "") << "\n";
    std::cout << "[Modelo] Contexto=" << cfg.contexto_max << " | D=" << cfg.dim_modelo
              << " | Cabecas=" << cfg.num_cabecas << " | Camadas=" << cfg.num_camadas
              << " | WeightTying=" << (cfg.weight_tying ? "sim" : "nao")
              << " | Parametros: " << modelo.total_parametros() << "\n\n";

    std::cout << std::fixed << std::setprecision(4);

    // 2. Gradient Check opcional
    if (rodar_grad_check) {
        const auto mini_batch = amostrar_batch(tokens_treino, mascara_treino, 2, 12, rng);
        const float erro = vibes::verificar_gradiente_numerico_gpt(modelo, mini_batch, 3, 1e-5);
        std::cout << "[GradCheck] Maior erro relativo (analitico vs numerico): "
                  << std::scientific << erro << std::fixed << "\n";
        if (erro >= 1e-3f) {
            std::cerr << "[GradCheck] FALHA: erro relativo acima do limite esperado!\n";
            return 2;
        }
        std::cout << "[GradCheck] OK! Attention, LayerNorm, WeightTying e FFN verificados.\n\n";
    }

    const auto janelas_treino_eval =
        construir_janelas_avaliacao(tokens_treino, mascara_treino, cfg.contexto_max);
    const auto janelas_val_eval =
        construir_janelas_avaliacao(tokens_val, mascara_val, cfg.contexto_max);

    // 3. Carregar apenas para chat OU treinar (do zero ou continuando checkpoint)
    if (!caminho_carregar.empty()) {
        try {
            modelo.carregar(caminho_carregar, false);
            std::cout << "[Checkpoint] Pesos carregados de: " << caminho_carregar << "\n";
            std::cout << "[Avaliacao] Treino: " << modelo.avaliar(janelas_treino_eval)
                      << " | Validacao: " << modelo.avaliar(janelas_val_eval) << "\n\n";
        } catch (const std::exception& e) {
            std::cerr << "[Erro] " << e.what() << "\n";
            return 1;
        }
    } else {
        if (!caminho_continuar.empty()) {
            try {
                modelo.carregar(caminho_continuar, true);
                std::cout << "[Checkpoint] Retomando treino de: " << caminho_continuar
                          << " (passo AdamW anterior: " << modelo.passo_otimizador() << ")\n";
            } catch (const std::exception& e) {
                std::cerr << "[Erro] " << e.what() << "\n";
                return 1;
            }
        }

        const float lr_min = lr_max * 0.1f;
        const int warmup = caminho_continuar.empty() ? std::max(1, total_passos / 10) : 1;

        vibes::ConjuntoTensoresGPT grad(cfg, 0.0f);

        std::cout << "[Treino] Perda inicial -> Treino: " << modelo.avaliar(janelas_treino_eval)
                  << " | Validacao: " << modelo.avaliar(janelas_val_eval) << "\n";
        std::cout << "[Treino] Iniciando " << total_passos << " passos (batch=" << tamanho_batch
                  << ", lr_max=" << lr_max << ", dropout=" << cfg.dropout
                  << (salvar_melhor_val ? ", salvar_melhor_val=sim" : "") << ")...\n";

        float melhor_val = 1e30f;
        int sem_melhora = 0;
        bool salvou_melhor = false;

        for (int passo = 1; passo <= total_passos; ++passo) {
            float lr = lr_max;
            if (passo <= warmup) {
                lr = lr_max * (static_cast<float>(passo) / static_cast<float>(warmup));
            } else {
                const float prog = static_cast<float>(passo - warmup) /
                                   static_cast<float>(std::max(1, total_passos - warmup));
                constexpr float pi = 3.14159265358979323846f;
                lr = lr_min + 0.5f * (lr_max - lr_min) * (1.0f + std::cos(pi * prog));
            }

            const auto batch =
                amostrar_batch(tokens_treino, mascara_treino, tamanho_batch, cfg.contexto_max, rng);
            const float perda_batch = modelo.calcular_perda_e_gradiente(
                batch, grad, true, static_cast<std::uint32_t>(passo * 131 + 42));
            modelo.passo_adamw(grad, lr, 0.01f, 1.0f);

            if (passo == 1 || passo == 25 || passo % 50 == 0 || passo == total_passos) {
                const float perda_val = modelo.avaliar(janelas_val_eval);
                const bool melhorou = (perda_val < melhor_val);
                if (melhorou) {
                    melhor_val = perda_val;
                    sem_melhora = 0;
                    if (salvar_melhor_val && !caminho_salvar.empty()) {
                        modelo.salvar(caminho_salvar, true);
                        if (tokenizador.usar_bpe) {
                            tokenizador.bpe.salvar(caminho_salvar + ".bpe");
                        }
                        salvou_melhor = true;
                    }
                } else {
                    ++sem_melhora;
                }

                std::cout << "  Passo " << std::setw(4) << passo << "/" << total_passos
                          << " | lr=" << std::setprecision(5) << lr << std::setprecision(4)
                          << " | Batch: " << perda_batch << " | Validacao: " << perda_val
                          << (melhorou && salvar_melhor_val ? " [Melhor Val!]" : "") << "\n";

                if (paciencia > 0 && sem_melhora >= paciencia) {
                    std::cout << "[EarlyStopping] Parando cedo apos " << sem_melhora
                              << " avaliacoes sem melhora (melhor val=" << melhor_val << ").\n";
                    break;
                }
            }
        }

        if (!caminho_salvar.empty() && !salvou_melhor) {
            modelo.salvar(caminho_salvar, true);
            if (tokenizador.usar_bpe) {
                tokenizador.bpe.salvar(caminho_salvar + ".bpe");
            }
            std::cout << "\n[Checkpoint] Modelo salvo em: " << caminho_salvar << "\n";
        } else if (salvou_melhor) {
            // Recarrega o melhor checkpoint de validacao para demonstracao/chat
            modelo.carregar(caminho_salvar, false);
            std::cout << "\n[Checkpoint] Melhor modelo de validacao (val=" << melhor_val
                      << ") salvo em: " << caminho_salvar << "\n";
        }
        std::cout << "\n";
    }

    // 4. Demonstracao de geracao com Temperatura baixa + Top-K (reduz palavras inventadas)
    const std::string prompt_demo = "Usuario: como funciona um transformer?\nIA: ";
    const auto ids_demo = tokenizador.codificar(prompt_demo);
    const auto ids_resp =
        modelo.gerar_ids(ids_demo, 100, 0.25f, 4, rng, tokenizador.id_quebra_linha());
    std::cout << "[Amostra gerada (temperatura=0.25, top_k=4)]:\n";
    std::cout << "--------------------------------------------------------------\n";
    std::cout << prompt_demo << tokenizador.decodificar(ids_resp) << "\n";
    std::cout << "--------------------------------------------------------------\n\n";

    if (!modo_chat) {
        return 0;
    }

    // 5. Chat interativo com historico, indicador de confianca, /temas e fallback honesto
    float temperatura = 0.25f;
    std::size_t top_k = 4;
    float limiar_confianca = 0.72f;  // abaixo disso, tenta sem historico ou avisa fora do dominio
    std::string historico;
    const auto perguntas_conhecidas = extrair_perguntas_corpus(texto);

    auto mostrar_sugestoes = [&](const std::string& filtro, std::size_t limite) {
        if (perguntas_conhecidas.empty()) {
            std::cout << "[Temas] Nenhuma pergunta no formato 'Usuario: ...' encontrada.\n";
            return;
        }
        std::vector<std::size_t> candidatas;
        const std::string filtro_low = para_minusculas_ascii(filtro);
        for (std::size_t i = 0; i < perguntas_conhecidas.size(); ++i) {
            if (filtro_low.empty() ||
                para_minusculas_ascii(perguntas_conhecidas[i]).find(filtro_low) !=
                    std::string::npos) {
                candidatas.push_back(i);
            }
        }
        if (candidatas.empty()) {
            std::cout << "[Temas] Nenhuma pergunta encontrada com o filtro '" << filtro << "'.\n";
            return;
        }
        std::shuffle(candidatas.begin(), candidatas.end(), rng.motor());
        const std::size_t mostrar = std::min(limite, candidatas.size());
        std::cout << "[Temas] Mostrando " << mostrar << " de " << candidatas.size()
                  << (filtro.empty() ? " perguntas conhecidas"
                                     : (" perguntas sobre '" + filtro + "'"))
                  << ":\n";
        for (std::size_t i = 0; i < mostrar; ++i) {
            std::cout << "  - " << perguntas_conhecidas[candidatas[i]] << "\n";
        }
    };

    std::cout << "=== Modo Chat Conversacional (Etapa 3: Mini-Transformer) ===\n";
    std::cout << "Comandos: /temas [filtro] | /temp <val> | /topk <k> | /conf <0..1> | /limpar | "
                 "/salvar <arq> | /sair\n";
    mostrar_sugestoes("", 5);
    std::cout << "\n";

    std::string entrada;
    while (true) {
        std::cout << "Voce: " << std::flush;
        if (!std::getline(std::cin, entrada)) {
            break;
        }
        if (entrada == "/sair" || entrada == "sair") {
            std::cout << "Encerrando chat. Ate a proxima!\n";
            break;
        }
        if (entrada == "/temas" || entrada == "/ajuda") {
            mostrar_sugestoes("", 12);
            continue;
        }
        if (entrada.rfind("/temas ", 0) == 0) {
            mostrar_sugestoes(entrada.substr(7), 15);
            continue;
        }
        if (entrada == "/limpar") {
            historico.clear();
            std::cout << "[Chat] Historico limpo.\n";
            continue;
        }
        if (entrada.rfind("/temp ", 0) == 0) {
            try {
                const float nova = std::stof(entrada.substr(6));
                if (nova > 0.05f && nova <= 5.0f) {
                    temperatura = nova;
                    std::cout << "[Config] Temperatura=" << temperatura << "\n";
                } else {
                    std::cout << "[Config] Use um valor entre 0.05 e 5.0.\n";
                }
            } catch (...) {
                std::cout << "[Config] Valor invalido. Exemplo: /temp 0.25\n";
            }
            continue;
        }
        if (entrada.rfind("/topk ", 0) == 0) {
            try {
                top_k = static_cast<std::size_t>(std::stoul(entrada.substr(6)));
                std::cout << "[Config] Top-K=" << top_k << (top_k == 0 ? " (desativado)" : "")
                          << "\n";
            } catch (...) {
                std::cout << "[Config] Valor invalido. Exemplo: /topk 4\n";
            }
            continue;
        }
        if (entrada.rfind("/conf ", 0) == 0) {
            try {
                const float novo = std::stof(entrada.substr(6));
                if (novo >= 0.0f && novo <= 1.0f) {
                    limiar_confianca = novo;
                    std::cout << "[Config] Limiar de confianca="
                              << static_cast<int>(limiar_confianca * 100.0f) << "%"
                              << (limiar_confianca == 0.0f ? " (fallback desativado)" : "") << "\n";
                } else {
                    std::cout << "[Config] Use um valor entre 0.0 e 1.0.\n";
                }
            } catch (...) {
                std::cout << "[Config] Valor invalido. Exemplo: /conf 0.20\n";
            }
            continue;
        }
        if (entrada.rfind("/salvar ", 0) == 0) {
            const std::string destino = entrada.substr(8);
            try {
                modelo.salvar(destino, true);
                if (tokenizador.usar_bpe) {
                    tokenizador.bpe.salvar(destino + ".bpe");
                }
                std::cout << "[Checkpoint] Salvo com sucesso em: " << destino << "\n";
            } catch (const std::exception& e) {
                std::cout << "[Erro] " << e.what() << "\n";
            }
            continue;
        }

        std::string tentativa = historico + "Usuario: " + entrada + "\nIA: ";
        auto ids_ctx = tokenizador.codificar(tentativa);
        auto gen = modelo.gerar_com_confianca(ids_ctx, 120, temperatura, top_k, rng,
                                              tokenizador.id_quebra_linha());

        // Se havia historico acumulado e a confianca caiu (mudanca de assunto), tenta
        // responder apenas com a pergunta atual antes de acionar o fallback.
        if (!historico.empty() && limiar_confianca > 0.0f &&
            gen.confianca_media < limiar_confianca) {
            const std::string solo = "Usuario: " + entrada + "\nIA: ";
            const auto ids_solo = tokenizador.codificar(solo);
            const auto gen_solo = modelo.gerar_com_confianca(ids_solo, 120, temperatura, top_k, rng,
                                                             tokenizador.id_quebra_linha());
            if (gen_solo.confianca_media > gen.confianca_media) {
                tentativa = solo;
                gen = gen_solo;
            }
        }

        const int conf_pct = static_cast<int>(std::round(gen.confianca_media * 100.0f));
        const std::string resposta = tokenizador.decodificar(gen.ids);

        if (limiar_confianca > 0.0f && gen.confianca_media < limiar_confianca) {
            std::cout << "IA (GPT | conf " << conf_pct << "% - fora do dominio): "
                      << "Ainda nao aprendi bem sobre isso! Voce pode adicionar exemplos em dados/ "
                         "ou usar '/conf 0' para ver a resposta crua.\n";
        } else {
            std::cout << "IA (GPT | conf " << conf_pct << "%): " << resposta << "\n";
            historico = tentativa + resposta + "\n\n";
        }
    }

    return 0;
}
