"""
Fase 1.6: Structured Output (JSON Mode)
========================================
Tujuan: paham cara maksa LLM balas dalam format JSON yang VALID & konsisten,
bukan teks bebas yang harus di-parse manual pakai regex (rawan gagal!).

Kenapa ini penting?
    Kalau kamu bikin aplikasi (bukan cuma chat), kamu butuh output LLM
    dalam bentuk yang bisa langsung dipakai kode: dict, list, angka, dst.
    Contoh: ekstrak data dari teks -> simpan ke database.

Ada 2 cara utama:
    1. response_format={"type": "json_object"} -> LLM dipaksa keluarin JSON
       yang valid secara sintaks, TAPI kamu masih perlu jelasin di prompt
       struktur field apa aja yang kamu mau (JSON mode gak tau skema kamu).
    2. JSON Schema strict (response_format type "json_schema") -> lebih
       ketat, field & tipe data dipaksa ikut skema persis. Tidak semua
       model/provider support ini (tergantung backend di 9router).

Di sini kita coba cara #1 dulu (paling universal & paling gampang jalan
di banyak model lewat 9router), lalu validasi hasilnya beneran JSON valid
pakai json.loads() + cek field yang wajib ada.

Contoh kasus: ekstrak data terstruktur dari review produk (teks bebas)
jadi: nama_produk, rating (1-5), sentimen, poin_positif[], poin_negatif[]

Cara jalanin:
    venv/bin/python 03_structured_output.py
"""

import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("ROUTER_BASE_URL", "http://localhost:20128/v1"),
    api_key=os.getenv("ROUTER_API_KEY", "***"),
)
MODEL = os.getenv("ROUTER_MODEL", "vps_combos")


SCHEMA_DESC = """
Kembalikan HANYA JSON object (tanpa teks lain, tanpa markdown code fence)
dengan struktur PERSIS seperti ini:
{
  "nama_produk": string,
  "rating": integer 1-5,
  "sentimen": "positif" | "negatif" | "netral",
  "poin_positif": [string, ...],
  "poin_negatif": [string, ...]
}
"""

REQUIRED_FIELDS = {"nama_produk", "rating", "sentimen", "poin_positif", "poin_negatif"}


def extract_review(review_text: str) -> dict:
    """Minta LLM ekstrak review teks bebas jadi JSON terstruktur."""
    messages = [
        {"role": "system", "content": f"Kamu adalah ekstraktor data. {SCHEMA_DESC}"},
        {"role": "user", "content": review_text},
    ]

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        response_format={"type": "json_object"},  # <- ini kuncinya: paksa JSON mode
        temperature=0,  # deterministik, kita mau data konsisten bukan kreatif
    )
    raw = response.choices[0].message.content

    # VALIDASI: jangan percaya buta, selalu cek beneran valid JSON + field lengkap
    data = json.loads(raw)  # kalau ini gagal -> LLM gak patuh format, harus retry/handle

    missing = REQUIRED_FIELDS - set(data.keys())
    if missing:
        raise ValueError(f"Field wajib hilang dari output LLM: {missing}")

    return data


def print_result(review_text: str, data: dict):
    print(f"\n=== REVIEW ASLI ===\n{review_text.strip()}")
    print(f"\n=== HASIL EKSTRAKSI (JSON tervalidasi) ===")
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    reviews = [
        """
        Baru beli kabel charger 9router Fast-Charge X1 di toko online.
        Chargingnya emang cepet banget, dari 10% ke 80% cuma 30 menit,
        kabelnya juga kuat gak gampang kupas. Tapi sayang warnanya
        cepet pudar dan colokannya agak longgar kalau dipasang ke HP
        tertentu. Overall lumayan sih, kasih 4 dari 5 bintang.
        """,
        """
        Powerbank VPS-Portable 20000mAh ini parah jelek banget, baru
        3 hari pakai udah gak bisa ngecas sama sekali, tombolnya juga
        keras susah dipencet. Buang-buang duit, nyesel beli. 1 bintang
        aja kebanyakan.
        """,
    ]

    all_ok = True
    for review in reviews:
        try:
            data = extract_review(review)
            print_result(review, data)
        except (json.JSONDecodeError, ValueError) as e:
            all_ok = False
            print(f"\n!!! GAGAL ekstrak review: {e}")

    print("\n" + "=" * 60)
    print("SEMUA REVIEW BERHASIL DIEKSTRAK & TERVALIDASI ✅" if all_ok
          else "ADA REVIEW YANG GAGAL DIVALIDASI ❌")
