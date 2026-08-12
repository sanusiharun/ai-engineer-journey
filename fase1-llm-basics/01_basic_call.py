"""
Fase 1: LLM API Mastery — Starter Project
==========================================
Tujuan: paham cara kerja LLM API dari level paling dasar,
TANPA framework (LangChain dkk) dulu. Biar konsepnya nempel.

Kita pakai router kamu sendiri (9router di VPS ini, port 20128)
supaya gratis dan langsung pakai infra yang udah kamu bangun.

Cara jalanin:
    1. Isi .env (lihat .env.example)
    2. source venv/bin/activate
    3. python 01_basic_call.py
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# 9router kamu OpenAI-compatible, jadi bisa pakai SDK openai langsung
client = OpenAI(
    base_url=os.getenv("ROUTER_BASE_URL", "http://localhost:20128/v1"),
    api_key=os.getenv("ROUTER_API_KEY", "sk-placeholder"),
)

MODEL = os.getenv("ROUTER_MODEL", "vps_combos")


def basic_call():
    """Contoh paling dasar: satu pertanyaan, satu jawaban."""
    print("=== 1. Basic Call ===")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "user", "content": "Jelaskan apa itu token dalam LLM, dalam 2 kalimat."}
        ],
    )
    print(response.choices[0].message.content)
    print()

    # Ini bagian pentingnya buat belajar: lihat usage (token count)
    print(f"Token digunakan -> prompt: {response.usage.prompt_tokens}, "
          f"completion: {response.usage.completion_tokens}, "
          f"total: {response.usage.total_tokens}")
    print()


def system_prompt_demo():
    """System prompt = 'kepribadian'/aturan main buat model."""
    print("=== 2. System Prompt ===")
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Kamu adalah asisten yang HANYA menjawab dengan bahasa gaul Jaksel."},
            {"role": "user", "content": "Apa itu Python?"},
        ],
    )
    print(response.choices[0].message.content)
    print()


def temperature_demo():
    """Temperature = seberapa 'random'/kreatif jawabannya.
    0 = deterministik & konsisten, 1+ = kreatif & bervariasi."""
    print("=== 3. Temperature Comparison ===")
    prompt = "Beri satu ide nama startup AI."

    for temp in [0.0, 1.2]:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
        )
        print(f"[temperature={temp}] {response.choices[0].message.content}")
    print()


def streaming_demo():
    """Streaming = terima jawaban token-per-token (real-time),
    bukan nunggu semua selesai baru muncul. Ini yang bikin ChatGPT
    kelihatan 'ngetik'."""
    print("=== 4. Streaming ===")
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": "Hitung 1 sampai 5, satu angka per baris."}],
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
    print("\n")


if __name__ == "__main__":
    basic_call()
    system_prompt_demo()
    temperature_demo()
    streaming_demo()
