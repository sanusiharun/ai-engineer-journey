"""
Fase 3: Mini-Agent dengan Tools (ReAct loop)
==============================================
Di Fase 1.5 (02_function_calling.py) kita udah belajar function calling:
LLM milih 1 fungsi, kita eksekusi, LLM jawab. SATU putaran, SELESAI.

Bedanya AGENT sama function-calling biasa:
    Function calling  = 1x tanya -> (mungkin) 1x panggil tool -> 1x jawab akhir
    AGENT (ReAct loop) = LOOP terus: mikir -> panggil tool -> lihat hasil ->
                          mikir lagi (butuh tool lagi? atau udah cukup?) ->
                          ... sampai LLM bilang "selesai, ini jawabannya"

Ini pola "ReAct" (Reasoning + Acting) — nama resminya dari paper ReAct
(Yao et al. 2022), tapi intinya simpel: kasih LLM tools, biarkan dia
manggil tool BERKALI-KALI dan MENGGABUNGKAN hasilnya sebelum jawab.
Ini pola inti di balik semua "agent" termasuk Hermes yang kamu pakai
sehari-hari — bedanya cuma jumlah & jenis tools + berapa lama loop-nya
dibiarkan jalan.

Tools yang disediakan agent ini (semua BENERAN, gak ada yang fake):
    1. baca_file(path)        -> baca isi file teks di VPS (dibatasi cwd,
                                  biar gak bisa baca /etc/passwd dkk)
    2. hitung(ekspresi)        -> kalkulator aman (pakai `ast`, BUKAN eval()
                                  mentah — eval() bisa dipakai buat RCE!)
    3. cek_9router_status()    -> beneran cek gateway lokal di port 20128
    4. tulis_catatan(isi)      -> beneran tulis ke file catatan_agent.txt

Task yang dikasih ke agent butuh MULTI-STEP tool chaining (bukan cuma 1
tool sekali panggil) — itu pembeda paling jelas dari Fase 1.5:
    "Baca file data_penjualan.txt, jumlahkan semua angkanya pakai
     kalkulator, terus tulis hasilnya ke catatan."
Agent harus: baca_file -> lihat isinya -> hitung(...) -> tulis_catatan(...)
    -> baru jawab ke user. 3 tool call berantai, bukan 1.

Cara jalanin:
    venv/bin/python 06_mini_agent.py
"""

import ast
import json
import operator
import os
import urllib.request

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("ROUTER_BASE_URL", "http://localhost:20128/v1"),
    api_key=os.getenv("ROUTER_API_KEY", "***"),
)
MODEL = os.getenv("ROUTER_MODEL", "vps_combos")

WORKDIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# 1. TOOLS BENERAN (Python biasa)
# ---------------------------------------------------------------------

def baca_file(path: str) -> str:
    """Baca file teks, dibatasi di dalam folder agent ini (anti path traversal)."""
    full = os.path.abspath(os.path.join(WORKDIR, path))
    if not full.startswith(WORKDIR):
        return "DITOLAK: path di luar folder yang diizinkan"
    if not os.path.isfile(full):
        return f"File tidak ditemukan: {path}"
    with open(full, "r") as f:
        return f.read()[:2000]  # batasi biar gak kebanyakan


# Kalkulator AMAN pakai ast — eval() mentah bisa dipakai orang jahat buat
# jalanin kode arbitrary (contoh: eval("__import__('os').system('rm -rf /')")).
# ast.literal_eval gak cukup buat "1+2*3", jadi kita bikin evaluator sendiri
# yang CUMA ngerti angka + operator matematika, gak ngerti apapun selain itu.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
}


def _eval_node(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return _OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        return _OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"Ekspresi tidak didukung: {ast.dump(node)}")


def hitung(ekspresi: str) -> str:
    """Kalkulator aman: hanya angka + - * / ** ( ), gak bisa eksekusi kode lain."""
    try:
        tree = ast.parse(ekspresi, mode="eval")
        return str(_eval_node(tree.body))
    except Exception as e:
        return f"Error hitung '{ekspresi}': {e}"


def cek_9router_status() -> str:
    """Beneran cek 9router lokal di port 20128."""
    try:
        req = urllib.request.Request("http://localhost:20128/v1/models")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            models = [m.get("id") for m in data.get("data", [])]
            return f"9router HIDUP, model: {models}"
    except Exception as e:
        return f"9router tidak bisa diakses: {e}"


def tulis_catatan(isi: str) -> str:
    """Beneran tulis file catatan_agent.txt di folder ini."""
    path = os.path.join(WORKDIR, "catatan_agent.txt")
    with open(path, "a") as f:
        f.write(isi.rstrip() + "\n")
    return f"Tersimpan ke catatan_agent.txt: {isi!r}"


AVAILABLE_FUNCTIONS = {
    "baca_file": baca_file,
    "hitung": hitung,
    "cek_9router_status": cek_9router_status,
    "tulis_catatan": tulis_catatan,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "baca_file",
        "description": "Baca isi file teks di folder kerja agent",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "nama file, misal data_penjualan.txt"}
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "hitung",
        "description": "Kalkulator matematika aman. Contoh ekspresi: '120+340+95' atau '(3+4)*2'",
        "parameters": {"type": "object", "properties": {
            "ekspresi": {"type": "string", "description": "ekspresi matematika, cth: '1+2+3'"}
        }, "required": ["ekspresi"]},
    }},
    {"type": "function", "function": {
        "name": "cek_9router_status",
        "description": "Cek status gateway LLM lokal (9router) di VPS ini",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "tulis_catatan",
        "description": "Simpan sebuah catatan/kesimpulan ke file catatan_agent.txt",
        "parameters": {"type": "object", "properties": {
            "isi": {"type": "string", "description": "isi catatan yang mau disimpan"}
        }, "required": ["isi"]},
    }},
]


# ---------------------------------------------------------------------
# 2. LOOP AGENT (bedanya sama function-calling biasa: ini LOOP)
# ---------------------------------------------------------------------

def run_agent(task: str, max_steps: int = 6):
    print(f"\n{'='*70}\nTASK: {task}\n{'='*70}")
    messages = [
        {"role": "system", "content": (
            "Kamu agent yang menyelesaikan task dengan memanggil tools "
            "berkali-kali jika perlu, satu per satu, sampai task selesai. "
            "Jangan menjawab final sebelum semua langkah yang dibutuhkan "
            "sudah benar-benar dikerjakan lewat tool. Jawab singkat & jelas."
        )},
        {"role": "user", "content": task},
    ]

    for step in range(1, max_steps + 1):
        response = client.chat.completions.create(
            model=MODEL, messages=messages, tools=TOOLS,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            print(f"\n[step {step}] LLM selesai, jawaban akhir:\n{msg.content}")
            return msg.content

        messages.append(msg.model_dump(exclude_none=True))
        for tc in msg.tool_calls:
            fname = tc.function.name
            fargs = json.loads(tc.function.arguments or "{}")
            func = AVAILABLE_FUNCTIONS.get(fname)
            result = func(**fargs) if func else f"Fungsi {fname} tidak dikenal"
            print(f"[step {step}] TOOL CALL -> {fname}({fargs})\n           hasil: {result}")
            messages.append({
                "role": "tool", "tool_call_id": tc.id, "content": str(result),
            })

    print("\n[STOP] max_steps tercapai tanpa jawaban final.")
    return None


if __name__ == "__main__":
    # siapkan data dummy buat ditest agent
    with open(os.path.join(WORKDIR, "data_penjualan.txt"), "w") as f:
        f.write("Penjualan Januari: 120\nPenjualan Februari: 340\nPenjualan Maret: 95\n")

    # bersihin catatan lama biar test ini bisa diverifikasi hasilnya
    catatan_path = os.path.join(WORKDIR, "catatan_agent.txt")
    if os.path.exists(catatan_path):
        os.remove(catatan_path)

    # Task 1: butuh chaining 3 tool (baca_file -> hitung -> tulis_catatan)
    run_agent(
        "Baca file data_penjualan.txt, jumlahkan semua angka penjualan di "
        "dalamnya pakai kalkulator, lalu simpan kesimpulannya (total "
        "penjualan Q1) ke catatan."
    )

    # Task 2: butuh 1 tool call beda (cek infra)
    run_agent("Agent kamu jalan di atas gateway apa? Cek statusnya beneran, jangan nebak.")

    # Task 3: gak butuh tool sama sekali (murni reasoning)
    run_agent("Kalau 3 tool dipanggil berurutan dan langkah ke-2 gagal, "
               "apa strategi yang masuk akal buat agent lakukan? Jelaskan singkat.")

    print(f"\n{'='*70}\nIsi catatan_agent.txt setelah run:\n{'='*70}")
    if os.path.exists(catatan_path):
        with open(catatan_path) as f:
            print(f.read())
    else:
        print("(belum ada catatan tersimpan)")
