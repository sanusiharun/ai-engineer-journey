"""
Fase 4: Evaluasi Prompt / Output (LLM-as-Judge + Heuristic Checks)
====================================================================
Sampai Fase 3 kita udah bisa bikin agent yang manggil tools & jawab.
Tapi gimana kita TAHU jawaban agent/LLM itu BAGUS? Selama ini kita cuma
`print()` dan baca manual pakai mata. Itu gak scalable — kalau prompt-nya
diubah, gimana tau itu perbaikan atau malah lebih jelek?

Fase ini ngajarin EVALUASI OTOMATIS, dua teknik yang paling umum dipakai
di industri buat evaluasi output LLM:

    1. HEURISTIC CHECKS (murah, cepat, deterministik)
       Cek properti objektif dari output: panjang teks, ada/tidaknya kata
       kunci wajib, format JSON valid, dll. Gak butuh panggil LLM lagi.

    2. LLM-AS-JUDGE (lebih mahal, tapi bisa menilai KUALITAS/nuansa)
       Pakai LLM KEDUA (bisa model yang sama) buat MENILAI jawaban LLM
       PERTAMA berdasarkan rubrik (skor 1-5 + alasan). Ini teknik yang
       dipakai buat benchmark model beneran (contoh: MT-Bench, AlpacaEval)
       dan juga dipakai buat regression testing prompt di produksi.

Task konkret di file ini: kita punya 2 VARIAN PROMPT buat task yang sama
("ringkas keluhan customer jadi 1 kalimat actionable buat tim support"),
lalu kita:
    a. Jalankan kedua prompt ke LLM beneran (via 9router) buat 3 keluhan
       customer.
    b. Heuristic check: ringkasan harus <= 25 kata & gak boleh kosong.
    c. LLM-as-judge: skor 1-5 (relevansi + actionable) + alasan singkat.
    d. Rekap skor rata-rata tiap varian prompt -> tentuin prompt mana yang
       menang, prompt engineering yang dipertanggungjawabkan pakai angka,
       bukan feeling.

Cara jalanin:
    venv/bin/python 07_prompt_eval.py
"""

import json
import os
import re
import statistics

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("ROUTER_BASE_URL", "http://localhost:20128/v1"),
    api_key=os.getenv("ROUTER_API_KEY", "***"),
)
MODEL = os.getenv("ROUTER_MODEL", "vps_combos")

# ---------------------------------------------------------------------
# 1. DATA: keluhan customer nyata-nyata-an buat ditest
# ---------------------------------------------------------------------
KELUHAN = [
    "Paket saya sudah 5 hari belum sampai padahal estimasi 2 hari, "
    "nomor resi juga gak update dari kemarin, saya butuh kepastian.",

    "Aplikasi selalu force close setiap saya buka menu pembayaran, "
    "sudah update ke versi terbaru tapi tetap sama, HP saya Android 13.",

    "Saya ditagih 2x untuk transaksi yang sama tanggal 10 kemarin, "
    "tolong salah satunya di-refund soalnya saldo saya jadi minus.",
]

# ---------------------------------------------------------------------
# 2. DUA VARIAN PROMPT buat task yang sama
# ---------------------------------------------------------------------
PROMPT_A_NAIVE = (
    "Ringkas keluhan customer berikut jadi 1 kalimat:\n\n{keluhan}"
)

PROMPT_B_STRUCTURED = (
    "Kamu asisten tim support. Ringkas keluhan customer berikut jadi TEPAT "
    "1 kalimat actionable (maksimal 25 kata) yang menyebutkan: (1) masalah "
    "inti, (2) apa yang customer minta/butuhkan. Jangan tambahkan opini, "
    "cuma ringkasan faktual buat tim support baca cepat.\n\n"
    "Keluhan customer:\n{keluhan}\n\nRingkasan:"
)

VARIANTS = {"A_naive": PROMPT_A_NAIVE, "B_structured": PROMPT_B_STRUCTURED}


def panggil_llm(prompt: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------
# 3. HEURISTIC CHECK (murah, deterministik, gak butuh LLM)
# ---------------------------------------------------------------------
def heuristic_check(ringkasan: str) -> dict:
    kata = re.findall(r"\w+", ringkasan)
    n_kalimat = len([s for s in re.split(r"[.!?]+", ringkasan) if s.strip()])
    return {
        "jumlah_kata": len(kata),
        "lolos_panjang": len(kata) <= 25,
        "tidak_kosong": len(ringkasan.strip()) > 0,
        "1_kalimat": n_kalimat <= 1,
    }


# ---------------------------------------------------------------------
# 4. LLM-AS-JUDGE: LLM kedua menilai output LLM pertama pakai rubrik
# ---------------------------------------------------------------------
JUDGE_SYSTEM = (
    "Kamu adalah QA evaluator yang ketat & objektif untuk tim support. "
    "Kamu akan diberi KELUHAN ASLI customer dan RINGKASAN yang dibuat AI. "
    "Nilai ringkasan itu dengan skor 1-5 berdasarkan 2 kriteria: "
    "(a) RELEVAN - menangkap masalah inti keluhan tanpa distorsi, "
    "(b) ACTIONABLE - tim support bisa langsung tahu harus ngapain dari "
    "ringkasan itu saja (tanpa baca keluhan asli). "
    "Balas HANYA dalam format JSON valid: "
    '{"skor": <angka 1-5>, "alasan": "<1 kalimat alasan singkat>"}'
)


def llm_judge(keluhan: str, ringkasan: str) -> dict:
    user_prompt = (
        f"KELUHAN ASLI:\n{keluhan}\n\nRINGKASAN AI:\n{ringkasan}\n\n"
        "Nilai sesuai rubrik. Balas JSON saja."
    )
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    raw = resp.choices[0].message.content.strip()
    # LLM kadang bungkus JSON pakai ```json ... ``` walau udah diminta polos
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
        return {"skor": int(parsed["skor"]), "alasan": parsed["alasan"]}
    except Exception as e:
        return {"skor": 0, "alasan": f"GAGAL PARSE JUDGE OUTPUT: {e} | raw={raw!r}"}


# ---------------------------------------------------------------------
# 5. JALANKAN EVALUASI PENUH
# ---------------------------------------------------------------------
def main():
    hasil_per_varian = {nama: [] for nama in VARIANTS}

    for i, keluhan in enumerate(KELUHAN, 1):
        print(f"\n{'='*70}\nKELUHAN #{i}: {keluhan[:60]}...\n{'='*70}")
        for nama_varian, template in VARIANTS.items():
            prompt = template.format(keluhan=keluhan)
            ringkasan = panggil_llm(prompt)
            h = heuristic_check(ringkasan)
            j = llm_judge(keluhan, ringkasan)

            print(f"\n-- Varian {nama_varian} --")
            print(f"Ringkasan : {ringkasan}")
            print(f"Heuristic : {h}")
            print(f"Judge     : skor={j['skor']}/5  alasan={j['alasan']}")

            hasil_per_varian[nama_varian].append({
                "keluhan_id": i,
                "ringkasan": ringkasan,
                "heuristic": h,
                "judge_skor": j["skor"],
                "judge_alasan": j["alasan"],
            })

    # ---------------------------------------------------------------
    # 6. REKAP: prompt mana yang menang, berdasarkan angka bukan feeling
    # ---------------------------------------------------------------
    print(f"\n{'='*70}\nREKAP EVALUASI\n{'='*70}")
    ringkasan_akhir = {}
    for nama_varian, hasil in hasil_per_varian.items():
        skor_list = [h["judge_skor"] for h in hasil]
        lolos_panjang = sum(1 for h in hasil if h["heuristic"]["lolos_panjang"])
        rata2 = statistics.mean(skor_list) if skor_list else 0
        ringkasan_akhir[nama_varian] = {
            "rata2_skor_judge": round(rata2, 2),
            "lolos_heuristic_panjang": f"{lolos_panjang}/{len(hasil)}",
        }
        print(f"{nama_varian:15s} -> rata-rata skor judge: {rata2:.2f}/5, "
              f"lolos batas panjang: {lolos_panjang}/{len(hasil)}")

    pemenang = max(ringkasan_akhir, key=lambda k: ringkasan_akhir[k]["rata2_skor_judge"])
    print(f"\n>> PEMENANG: {pemenang} (skor rata-rata tertinggi)")

    # simpan hasil lengkap ke JSON biar bisa dicek ulang / dibandingin run lain
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "detail_per_varian": hasil_per_varian,
            "rekap": ringkasan_akhir,
            "pemenang": pemenang,
        }, f, indent=2, ensure_ascii=False)
    print(f"\nHasil lengkap tersimpan di: {out_path}")


if __name__ == "__main__":
    main()
