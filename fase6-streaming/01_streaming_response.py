"""
Fase 6: Streaming Responses (Token-by-Token) + Streaming Function Calling
==========================================================================
Sampai Fase 5 kita udah punya: basic call, function calling, structured
output, RAG, mini-agent (ReAct), evaluasi, dan conversation memory.

Tapi semua call di fase-fase sebelumnya itu BLOCKING: kita kirim request,
lalu nunggu SAMPAI SELESAI baru dapet response utuh. Kalau jawaban LLM
panjang (misal 500 token), user nunggu diem beberapa detik sebelum lihat
apa-apa. Di produk chatbot beneran (ChatGPT, Claude, dst), teks muncul
KATA PER KATA secara real-time -- itu namanya STREAMING.

Kenapa streaming penting buat AI Engineer:
    1. UX -- user ngerasa lebih responsif walau total waktu sama.
       Time-to-first-token (TTFT) jadi metric penting, bukan cuma total
       latency.
    2. Bisa "cancel early" -- kalau user udah dapet cukup info atau nutup
       chat, kita bisa stop generate & hemat biaya (gak perlu tunggu
       token terakhir).
    3. Streaming + function calling itu TRICKY -- karena tool call
       (nama function & argument JSON) juga datang dalam PECAHAN
       (chunks), bukan sekaligus. Kita harus AKUMULASI potongan-potongan
       itu sebelum bisa parse & eksekusi function-nya.

Task konkret di file ini:
    a. Streaming basic: kirim request dengan stream=True, cetak token
       satu-satu begitu datang (simulasi "efek ngetik").
    b. Ukur Time-To-First-Token (TTFT) vs total generation time --
       supaya kelihatan kenapa streaming beda dari blocking call.
    c. Streaming + tool calls: minta LLM manggil function
       'get_weather', tapi lewat stream. Kita akumulasi delta dari tiap
       chunk (nama function ngumpul dari beberapa chunk, argument JSON
       juga ngumpul karakter demi karakter), baru setelah stream
       selesai kita parse & "eksekusi" function-nya.

Catatan: 9router adalah gateway OpenAI-compatible, jadi kita pakai SDK
`openai` resmi, cukup ganti base_url & api_key ke 9router (LANGKAH WAJIB:
API key TIDAK di-hardcode, selalu load dari .env).
"""

import json
import os
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

BASE_URL = os.environ["ROUTER_BASE_URL"]
API_KEY = os.environ["ROUTER_API_KEY"]
MODEL = os.environ["ROUTER_MODEL"]

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def demo_basic_streaming():
    """Streaming teks biasa -- cetak token begitu datang, ukur TTFT."""
    print("=" * 70)
    print("DEMO 1: Basic Streaming (token-by-token) + TTFT measurement")
    print("=" * 70)

    prompt = (
        "Jelaskan dalam 3 kalimat singkat: kenapa streaming response "
        "penting untuk aplikasi chatbot production. Bahasa Indonesia."
    )
    print(f"\n[User] {prompt}\n")
    print("[Assistant] ", end="", flush=True)

    start = time.perf_counter()
    first_token_time = None
    full_text = ""
    chunk_count = 0

    stream = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            print(delta.content, end="", flush=True)
            full_text += delta.content
            chunk_count += 1

    end = time.perf_counter()
    print()  # newline setelah stream selesai

    ttft = (first_token_time - start) if first_token_time else None
    total = end - start

    print("\n--- Metrics ---")
    print(f"Time-To-First-Token (TTFT) : {ttft:.3f}s" if ttft else "TTFT: N/A")
    print(f"Total generation time      : {total:.3f}s")
    print(f"Jumlah chunk diterima      : {chunk_count}")
    print(f"Panjang total output       : {len(full_text)} karakter")
    if ttft and total > 0:
        print(
            f"-> User mulai lihat teks di {ttft:.3f}s, jauh lebih cepat "
            f"dibanding nunggu semua {total:.3f}s selesai (blocking call)."
        )
    return full_text


def get_weather(city: str) -> str:
    """Fungsi 'palsu' (simulasi tool) -- di dunia nyata ini panggil API cuaca beneran."""
    fake_db = {
        "jakarta": "32°C, cerah berawan",
        "tangerang selatan": "31°C, cerah berawan",
        "bandung": "24°C, hujan ringan",
        "surabaya": "33°C, panas terik",
    }
    key = city.strip().lower()
    return fake_db.get(key, f"Data cuaca untuk '{city}' tidak tersedia (simulasi).")


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Dapatkan info cuaca terkini untuk sebuah kota.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Nama kota, misal 'Jakarta' atau 'Bandung'",
                    }
                },
                "required": ["city"],
            },
        },
    }
]


def demo_streaming_tool_calls():
    """
    Streaming + function calling. Bedanya sama non-stream: nama function
    & argument JSON datang dalam PECAHAN kecil per-chunk. Kita harus
    akumulasi berdasarkan `index` dari tiap tool_call delta.
    """
    print("\n" + "=" * 70)
    print("DEMO 2: Streaming + Function Calling (akumulasi tool call chunks)")
    print("=" * 70)

    user_msg = "Cuaca di Tangerang Selatan sekarang gimana?"
    print(f"\n[User] {user_msg}\n")

    messages = [{"role": "user", "content": user_msg}]

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        stream=True,
    )

    # Akumulator: tool_calls bisa lebih dari satu, di-index oleh posisi (index)
    # yang dikirim tiap delta.
    accumulated_tool_calls = {}
    accumulated_content = ""
    chunk_count = 0

    print("[Streaming raw chunks]")
    for chunk in stream:
        chunk_count += 1
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta is None:
            continue

        if delta.content:
            accumulated_content += delta.content
            print(delta.content, end="", flush=True)

        if delta.tool_calls:
            for tc_delta in delta.tool_calls:
                idx = tc_delta.index
                if idx not in accumulated_tool_calls:
                    accumulated_tool_calls[idx] = {
                        "id": tc_delta.id or "",
                        "name": "",
                        "arguments": "",
                    }
                if tc_delta.id:
                    accumulated_tool_calls[idx]["id"] = tc_delta.id
                if tc_delta.function:
                    if tc_delta.function.name:
                        accumulated_tool_calls[idx]["name"] += tc_delta.function.name
                    if tc_delta.function.arguments:
                        accumulated_tool_calls[idx]["arguments"] += (
                            tc_delta.function.arguments
                        )
                print(".", end="", flush=True)  # progres akumulasi argumen

    print(f"\n\n[Info] Total chunk diterima dari stream: {chunk_count}")

    if not accumulated_tool_calls:
        print("[Info] Model tidak memanggil tool, langsung jawab teks:")
        print(accumulated_content)
        return

    print(f"[Info] Tool call(s) terdeteksi & terakumulasi penuh:")
    for idx, tc in accumulated_tool_calls.items():
        print(f"  - index={idx} id={tc['id']} name={tc['name']!r} "
              f"raw_arguments={tc['arguments']!r}")

    # Eksekusi tiap tool call yang berhasil di-parse
    tool_results = []
    for idx, tc in accumulated_tool_calls.items():
        try:
            args = json.loads(tc["arguments"])
        except json.JSONDecodeError as e:
            print(f"[Error] Gagal parse argument JSON: {e}")
            continue

        if tc["name"] == "get_weather":
            result = get_weather(args.get("city", ""))
            print(f"\n[Tool Executed] get_weather({args}) -> {result}")
            tool_results.append(
                {
                    "tool_call_id": tc["id"],
                    "name": tc["name"],
                    "result": result,
                }
            )

    # Kirim hasil tool balik ke LLM (masih pakai streaming) buat jawaban final
    assistant_tool_call_msg = {
        "role": "assistant",
        "content": accumulated_content or None,
        "tool_calls": [
            {
                "id": tc["id"],
                "type": "function",
                "function": {"name": tc["name"], "arguments": tc["arguments"]},
            }
            for tc in accumulated_tool_calls.values()
        ],
    }
    messages.append(assistant_tool_call_msg)
    for tr in tool_results:
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tr["tool_call_id"],
                "content": tr["result"],
            }
        )

    print("\n[Final answer streaming]")
    print("[Assistant] ", end="", flush=True)
    final_stream = client.chat.completions.create(
        model=MODEL, messages=messages, stream=True
    )
    final_text = ""
    for chunk in final_stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if delta and delta.content:
            print(delta.content, end="", flush=True)
            final_text += delta.content
    print()
    return final_text


if __name__ == "__main__":
    demo_basic_streaming()
    demo_streaming_tool_calls()
    print("\n" + "=" * 70)
    print("SELESAI. Kunci pembelajaran Fase 6:")
    print("=" * 70)
    print(
        """
1. stream=True bikin response.choices[0].delta.content datang bertahap,
   bukan message.content sekaligus di akhir.
2. TTFT (time-to-first-token) adalah metric UX penting, terpisah dari
   total generation time -- streaming gak bikin total lebih cepat, tapi
   bikin user MULAI lihat hasil jauh lebih cepat.
3. Tool calls saat streaming datang sebagai delta.tool_calls list, tiap
   elemen punya `.index` buat tau tool call keberapa, dan `.function.name`
   / `.function.arguments` yang harus di-AKUMULASI (concat) sampai stream
   selesai sebelum bisa json.loads() argument-nya dengan aman.
4. Setelah tool dieksekusi, hasilnya dikirim balik sebagai role="tool"
   message, sama seperti pola non-streaming di Fase 1.5 -- bedanya cuma
   di cara terima delta-nya.
"""
    )
