"""
Fase 1.5: Function Calling / Tool Use
======================================
Tujuan: paham cara LLM "manggil fungsi" beneran. Ini konsep INTI yang
dipakai semua AI agent (termasuk Hermes yang kamu pakai sehari-hari!).

Cara kerjanya (4 langkah):
    1. Kita kasih tau LLM "ada fungsi bernama X, gunanya begini, parameternya begini"
       (lewat parameter `tools` di API call — bukan lewat kode Python beneran)
    2. LLM baca pertanyaan user, mikir "oh ini butuh fungsi X dengan argumen Y"
       lalu balas dengan tool_call (BUKAN teks jawaban langsung)
    3. Kita (kode Python) yang BENERAN eksekusi fungsi X dengan argumen Y itu
    4. Hasil eksekusi dikirim balik ke LLM, baru LLM susun jawaban akhir ke user

Jadi LLM sendiri gak pernah "run code" — dia cuma milih fungsi & argumen,
kita yang jalanin. Ini penting dipahami karena banyak orang kira LLM
langsung eksekusi kode, padahal enggak.

Contoh di sini: LLM punya akses ke fungsi get_weather() (fake/simulasi)
dan get_9router_status() (BENERAN cek 9router lokal kamu di port 20128!).

Cara jalanin:
    venv/bin/python 02_function_calling.py
"""

import os
import json
import urllib.request
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("ROUTER_BASE_URL", "http://localhost:20128/v1"),
    api_key=os.getenv("ROUTER_API_KEY", "***"),
)
MODEL = os.getenv("ROUTER_MODEL", "vps_combos")


# ---------------------------------------------------------------------
# 1. FUNGSI BENERAN yang bisa dipanggil LLM (Python biasa, gak ada magic)
# ---------------------------------------------------------------------

def get_weather(kota: str) -> str:
    """Fake weather lookup — cuma simulasi, biar simpel."""
    fake_data = {
        "jakarta": "32°C, cerah berawan",
        "bandung": "22°C, hujan ringan",
        "surabaya": "34°C, panas terik",
    }
    return fake_data.get(kota.lower(), f"Data cuaca untuk {kota} tidak tersedia")


def get_9router_status() -> str:
    """BENERAN cek 9router lokal di port 20128 — bukan simulasi!"""
    try:
        req = urllib.request.Request("http://localhost:20128/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m.get("id") for m in data.get("data", [])]
            return f"9router HIDUP, model tersedia: {models}"
    except Exception as e:
        return f"9router tidak bisa diakses: {e}"


# Mapping nama fungsi (string) -> fungsi Python beneran
AVAILABLE_FUNCTIONS = {
    "get_weather": get_weather,
    "get_9router_status": get_9router_status,
}

# ---------------------------------------------------------------------
# 2. Definisi "tools" — deskripsi fungsi dalam format JSON Schema
#    Ini yang dibaca LLM untuk mutusin mau manggil fungsi apa
# ---------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Ambil info cuaca terkini untuk sebuah kota di Indonesia",
            "parameters": {
                "type": "object",
                "properties": {
                    "kota": {
                        "type": "string",
                        "description": "Nama kota, misal: Jakarta, Bandung, Surabaya",
                    }
                },
                "required": ["kota"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_9router_status",
            "description": "Cek status/kesehatan 9router (LLM gateway lokal di VPS ini) dan model apa saja yang tersedia",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def chat_with_tools(user_message: str):
    print(f"\n=== USER: {user_message} ===")
    messages = [{"role": "user", "content": user_message}]

    # Langkah 1: kirim pesan + daftar tools ke LLM
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
    )
    msg = response.choices[0].message

    # Langkah 2: cek apakah LLM mau manggil fungsi
    if not msg.tool_calls:
        print(f"LLM jawab langsung (gak butuh tool): {msg.content}")
        return

    print(f"LLM minta panggil {len(msg.tool_calls)} fungsi:")
    messages.append(msg.model_dump(exclude_none=True))

    # Langkah 3: kita BENERAN eksekusi tiap tool_call
    for tool_call in msg.tool_calls:
        fname = tool_call.function.name
        fargs = json.loads(tool_call.function.arguments or "{}")
        print(f"  -> {fname}({fargs})")

        func = AVAILABLE_FUNCTIONS.get(fname)
        result = func(**fargs) if func else f"Fungsi {fname} tidak dikenal"
        print(f"     hasil eksekusi: {result}")

        # Langkah 4: kirim hasil balik ke LLM sebagai role "tool"
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": str(result),
        })

    # Minta LLM susun jawaban akhir berdasarkan hasil tool
    final = client.chat.completions.create(model=MODEL, messages=messages)
    print(f"\nJAWABAN AKHIR LLM: {final.choices[0].message.content}")


if __name__ == "__main__":
    chat_with_tools("Cuaca di Bandung gimana hari ini?")
    chat_with_tools("Cek dong status 9router aku masih hidup gak, model apa aja yang ada?")
    chat_with_tools("Siapa presiden Indonesia sekarang?")  # gak butuh tool
