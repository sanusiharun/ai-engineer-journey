# AI Engineer Journey 🚀

Progres belajar AI Engineering — dari LLM basics sampai RAG, agent, dan evaluasi.
Setiap pelajaran ditulis sebagai kode nyata yang dijalankan & diverifikasi, bukan sekadar teori.

## Progress

| Fase | Topik | Status |
|---|---|---|
| 1 | LLM Basics (basic call) | ✅ |
| 1.5 | Function Calling | ✅ |
| 1.6 | Structured Output (JSON mode) | ✅ |
| 2 | RAG Basics | ✅ |
| 2.5 | RAG dengan ChromaDB (vector database) | ✅ |
| 3 | Mini-agent dengan tools (ReAct loop) | ✅ |
| 4 | Evaluasi prompt/output (heuristic + LLM-as-judge) | ✅ |
| 5 | Conversation Memory Management (sliding window + summarization) | ✅ |

## Struktur

```
fase1-llm-basics/    # Basic LLM call, function calling, structured output, RAG basics
fase2-rag/           # RAG dengan ChromaDB (vector database)
fase3-agent/         # Mini-agent dengan tools (ReAct loop: mikir -> panggil tool -> lihat hasil -> ulang)
fase4-eval/          # Evaluasi prompt/output: heuristic checks + LLM-as-judge (bandingin varian prompt pakai angka)
fase5-memory/        # Conversation memory: sliding window + rolling summarization biar token gak bengkak
```

## Setup

Setiap folder fase punya `venv` sendiri (di-ignore dari git) dan butuh `.env` (lihat `.env.example`):

```bash
cd fase1-llm-basics
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt   # kalau ada
cp .env.example .env              # isi dengan API key kamu sendiri
python 01_basic_call.py
```

## Stack

- **LLM Gateway:** [9router](https://github.com/diegosouzapw/OmniRoute) — self-hosted, OpenAI-compatible
- **Model:** `vps_combos` (free-tier models via 9router)
- **Vector DB:** ChromaDB (fase 2.5+)

---
*Belajar rutin dijadwalkan otomatis via [Hermes Agent](https://hermes-agent.nousresearch.com).*
