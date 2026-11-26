#!/usr/bin/env python3
"""
Baixa e converte grandes conjuntos de dados abertos em portugues para o formato
do cpp-vibes-ai (`Usuario: ... / IA: ...` e texto puro UTF-8).

Zero dependencias externas: usa apenas a biblioteca padrao do Python 3.

Datasets suportados:
  1. oasst1      -> OpenAssistant OASST1 filtrado para portugues (dialogos multi-turno)
  2. alpaca_pt   -> Alpaca GPT-4 em portugues (~50 mil pares de instrucao/resposta)
  3. machado     -> Dom Casmurro (Machado de Assis, Project Gutenberg #55752, dominio publico)

Uso:
  python3 scripts/baixar_dados.py --todos
  python3 scripts/baixar_dados.py --dataset oasst1 --max-exemplos 2000
"""

import argparse
import gzip
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

PASTA_SAIDA = Path("dados/externos")

URL_OASST1 = (
    "https://huggingface.co/datasets/OpenAssistant/oasst1/resolve/main/"
    "2023-04-12_oasst_ready.trees.jsonl.gz"
)
URL_ALPACA_PT = (
    "https://huggingface.co/datasets/FreedomIntelligence/alpaca-gpt4-portuguese/resolve/main/"
    "alpaca-gpt4-portuguese.json"
)
URL_DOM_CASMURRO = "https://www.gutenberg.org/cache/epub/55752/pg55752.txt"


def baixar_bytes(url: str, descricao: str) -> bytes:
    print(f"[Download] Baixando {descricao}...")
    print(f"           URL: {url}")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "cpp-vibes-ai-downloader/1.0 (educativo)"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        dados = resp.read()
    print(f"[Download] Concluido: {len(dados) / (1024 * 1024):.2f} MB")
    return dados


def limpar_linha(texto: str) -> str:
    # Normaliza pontuacao tipografica, remove ruido e mantem cada turno em 1 linha limpa
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    substituicoes = {
        "\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'",
        "\u2013": "-", "\u2014": "-", "\u2026": "...", "\u00a0": " ",
    }
    for orig, novo in substituicoes.items():
        texto = texto.replace(orig, novo)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto


def extrair_caminhos_oasst(no: dict, caminho_atual: list, resultados: list):
    papel = no.get("role", "")
    texto = limpar_linha(no.get("text", ""))
    if not texto:
        return

    novo_caminho = caminho_atual + [(papel, texto)]
    filhos = no.get("replies") or []
    if not filhos:
        if len(novo_caminho) >= 2:
            resultados.append(novo_caminho)
    else:
        # Segue o filho mais bem avaliado primeiro para manter dialogos de alta qualidade
        filhos_ordenados = sorted(
            filhos,
            key=lambda c: c.get("rank") if c.get("rank") is not None else 999,
        )
        for filho in filhos_ordenados[:2]:
            extrair_caminhos_oasst(filho, novo_caminho, resultados)


def baixar_oasst1(max_conversas: int) -> Path:
    destino = PASTA_SAIDA / "oasst1_pt.txt"
    bruto = baixar_bytes(URL_OASST1, "OpenAssistant OASST1 (multilingue -> filtro PT)")

    conversas = []
    with gzip.GzipFile(fileobj=io.BytesIO(bruto)) as gz:
        for linha in gz:
            arvore = json.loads(linha.decode("utf-8"))
            prompt = arvore.get("prompt") or {}
            lang = (prompt.get("lang") or "").lower()
            if lang.startswith("pt"):
                extrair_caminhos_oasst(prompt, [], conversas)
                if max_conversas > 0 and len(conversas) >= max_conversas:
                    break

    with destino.open("w", encoding="utf-8") as arq:
        for conv in conversas[: max_conversas if max_conversas > 0 else len(conversas)]:
            for papel, texto in conv:
                prefixo = "Usuario" if papel == "prompter" else "IA"
                arq.write(f"{prefixo}: {texto}\n")
            arq.write("\n")

    tamanho_kb = destino.stat().st_size / 1024.0
    print(f"[OK] Salvo {destino} ({len(conversas)} conversas em portugues, {tamanho_kb:.1f} KB)\n")
    return destino


def baixar_alpaca_pt(max_exemplos: int) -> Path:
    destino = PASTA_SAIDA / "alpaca_pt.txt"
    bruto = baixar_bytes(URL_ALPACA_PT, "Alpaca GPT-4 Portugues (~50k instrucoes)")

    itens = json.loads(bruto.decode("utf-8"))
    if max_exemplos > 0:
        itens = itens[:max_exemplos]

    total = 0
    with destino.open("w", encoding="utf-8") as arq:
        for item in itens:
            if "conversations" in item:
                convs = item["conversations"]
                escreveu = False
                for msg in convs:
                    origem = (msg.get("from") or "").lower()
                    valor = limpar_linha(msg.get("value") or "")
                    if not valor:
                        continue
                    prefixo = "Usuario" if origem in ("human", "user") else "IA"
                    arq.write(f"{prefixo}: {valor}\n")
                    escreveu = True
                if escreveu:
                    arq.write("\n")
                    total += 1
            else:
                inst = limpar_linha(item.get("instruction") or "")
                inp = limpar_linha(item.get("input") or "")
                out = limpar_linha(item.get("output") or "")
                if not inst or not out:
                    continue
                pergunta = f"{inst} ({inp})" if inp else inst
                arq.write(f"Usuario: {pergunta}\nIA: {out}\n\n")
                total += 1

    tamanho_mb = destino.stat().st_size / (1024.0 * 1024.0)
    print(f"[OK] Salvo {destino} ({total} dialogos, {tamanho_mb:.2f} MB)\n")
    return destino


def baixar_machado() -> Path:
    destino = PASTA_SAIDA / "dom_casmurro.txt"
    bruto = baixar_bytes(URL_DOM_CASMURRO, "Dom Casmurro (Machado de Assis - Gutenberg #55752)")
    texto = bruto.decode("utf-8", errors="ignore").replace("\r\n", "\n")

    # Remove cabecalho e rodape padrao do Project Gutenberg
    inicio = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", texto)
    fim = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", texto)
    if inicio and fim:
        texto = texto[inicio.end() : fim.start()].strip()

    destino.write_text(texto + "\n", encoding="utf-8")
    tamanho_kb = destino.stat().st_size / 1024.0
    print(f"[OK] Salvo {destino} ({len(texto)} caracteres, {tamanho_kb:.1f} KB)\n")
    return destino


def main() -> int:
    parser = argparse.ArgumentParser(description="Baixa grandes datasets em portugues para treino.")
    parser.add_argument(
        "--dataset",
        choices=["oasst1", "alpaca_pt", "machado", "todos"],
        default="todos",
        help="Qual dataset baixar (padrao: todos)",
    )
    parser.add_argument(
        "--max-alpaca",
        type=int,
        default=5000,
        help="Limite de exemplos do Alpaca-PT (0 = todos os ~50k; padrao: 5000 para treino agil em CPU)",
    )
    parser.add_argument(
        "--max-oasst",
        type=int,
        default=0,
        help="Limite de conversas em portugues do OASST1 (0 = todas)",
    )
    args = parser.parse_args()

    PASTA_SAIDA.mkdir(parents=True, exist_ok=True)
    erros = 0

    tarefas = []
    if args.dataset in ("oasst1", "todos"):
        tarefas.append(("oasst1", lambda: baixar_oasst1(args.max_oasst)))
    if args.dataset in ("alpaca_pt", "todos"):
        tarefas.append(("alpaca_pt", lambda: baixar_alpaca_pt(args.max_alpaca)))
    if args.dataset in ("machado", "todos"):
        tarefas.append(("machado", baixar_machado))

    for nome, fn in tarefas:
        try:
            fn()
        except urllib.error.URLError as e:
            erros += 1
            print(f"[Aviso] Falha ao baixar {nome}: {e}\n", file=sys.stderr)

    # Cria tambem um corpus conversacional grande combinando OASST1 + Alpaca-PT + locais
    combinado = PASTA_SAIDA / "conversas_grande.txt"
    fontes_conv = [
        Path("dados/dialogos.txt"),
        Path("dados/conversa_cotidiana.txt"),
        Path("dados/programacao_cpp.txt"),
        Path("dados/machine_learning.txt"),
        PASTA_SAIDA / "oasst1_pt.txt",
        PASTA_SAIDA / "alpaca_pt.txt",
    ]
    existentes = [p for p in fontes_conv if p.exists()]
    if existentes:
        with combinado.open("w", encoding="utf-8") as out:
            for p in existentes:
                out.write(p.read_text(encoding="utf-8").strip() + "\n\n")
        mb = combinado.stat().st_size / (1024.0 * 1024.0)
        print(f"[Combinado] Criado {combinado} ({mb:.2f} MB de conversas em portugues!)")

    # Cria um corpus calibrado para CPU (~200-300 KB): combina os datasets locais +
    # os dialogos mais curtos e limpos de OASST1 e Alpaca-PT (sem blocos de codigo)
    corpus_cpu = PASTA_SAIDA / "conversas_cpu.txt"
    locais = [
        Path("dados/dialogos.txt"),
        Path("dados/conversa_cotidiana.txt"),
        Path("dados/programacao_cpp.txt"),
        Path("dados/machine_learning.txt"),
    ]
    externos_filtrados = []
    for fonte in (PASTA_SAIDA / "oasst1_pt.txt", PASTA_SAIDA / "alpaca_pt.txt"):
        if not fonte.exists():
            continue
        blocos = fonte.read_text(encoding="utf-8").strip().split("\n\n")
        for bloco in blocos:
            linhas = [l.strip() for l in bloco.splitlines() if l.strip()]
            if len(linhas) != 2:
                continue
            u, ia = linhas
            if not (u.startswith("Usuario: ") and ia.startswith("IA: ")):
                continue
            if "```" in bloco or "http://" in bloco or "https://" in bloco:
                continue
            if 12 <= len(u) <= 130 and 15 <= len(ia) <= 240:
                externos_filtrados.append(f"{u}\n{ia}")
            if len(externos_filtrados) >= 1200:
                break

    with corpus_cpu.open("w", encoding="utf-8") as out:
        for p in locais:
            if p.exists():
                out.write(p.read_text(encoding="utf-8").strip() + "\n\n")
        for bloco in externos_filtrados:
            out.write(bloco + "\n\n")
    kb = corpus_cpu.stat().st_size / 1024.0
    print(f"[CPU] Criado {corpus_cpu} ({len(externos_filtrados)} dialogos externos limpos + locais, {kb:.1f} KB)")

    return 0 if (existentes and erros < len(tarefas)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
