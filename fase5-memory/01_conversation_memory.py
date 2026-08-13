"""
Fase 5: Conversation Memory Management (Sliding Window + Summarization)
========================================================================
Sampai Fase 4 kita punya: basic call, function calling, structured output,
RAG (basic + ChromaDB), mini-agent (ReAct), dan evaluasi otomatis.

Tapi semua contoh itu SINGLE-TURN atau agent yang cuma jalan sekali lalu
selesai. Di dunia nyata (chatbot, asisten), user chat BERKALI-KALI dalam
satu sesi. Masalahnya: context window LLM itu TERBATAS (token limit), dan
makin panjang history yang dikirim, makin MAHAL & LAMBAT tiap request.

Fase ini ngajarin CONVERSATION MEMORY MANAGEMENT, dua teknik yang paling
umum dipakai di produksi:

    1. SLIDING WINDOW (murah, simpel)
       Cuma kirim N pesan terakhir ke LLM. History lama di luar window
       tetap disimpan di memory penuh (buat log/audit), tapi TIDAK dikirim
       ke LLM lagi. Trade-off: LLM "lupa" konteks yang jauh di awal.

    2. SUMMARIZATION MEMORY (lebih mahal tapi tetap punya "ingatan")
       Begitu history melebihi batas token tertentu, pesan-pesan LAMA
       diringkas jadi satu blok "summary" pakai LLM, lalu summary itu yang
       dikirim sebagai konteks (bukan raw messages). Summary di-update
       terus (rolling summary) tiap kali window penuh lagi. Ini teknik
       yang dipakai LangChain's ConversationSummaryMemory & produk chatbot
       komersial (misal ChatGPT sendiri, di baliknya ada versi dari ini).

Task konkret di file ini: kita bikin kelas `ConversationMemory` yang:
    a. Nyimpen SEMUA history percakapan (full log).
    b. Ngitung token pakai tiktoken (approx, karena model bukan OpenAI asli
       tapi tiktoken cukup dekat buat estimasi budget token).
    c. Kalau history terkirim > MAX_TOKENS, otomatis SUMMARIZE pesan lama
       jadi 1 system note, sisain N pesan terakhir apa adanya (hybrid:
       summarization + sliding window).
    d. Dipakai buat simulasi percakapan panjang (banyak turn) & kita
       buktikan token yang dikirim ke LLM TETAP KECIL walau history-nya
       udah panjang banget.

Cara jalanin:
    venv/bin/python 01_conversation_memory.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

try:
    import tiktoken

    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback kalau tiktoken data gak ke-download
    _ENC = None

load_dotenv()

BASE_URL = os.environ["ROUTER_BASE_URL"]
API_KEY = os.environ["ROUTER_API_KEY"]
MODEL = os.environ["ROUTER_MODEL"]

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


def count_tokens(text: str) -> int:
    """Estimasi jumlah token. Pakai tiktoken kalau ada, fallback ke heuristik
    kasar (1 token ~= 4 karakter) kalau tiktoken gak bisa load encoding-nya
    (butuh koneksi internet buat download data BPE pertama kali)."""
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


def messages_tokens(messages: list[dict]) -> int:
    return sum(count_tokens(m["content"]) for m in messages)


class ConversationMemory:
    """Hybrid memory: sliding window + rolling summarization.

    - `full_log`   : SEMUA pesan dari awal sesi (buat audit/debug), gak
                      pernah dihapus.
    - `working_set`: apa yang beneran dikirim ke LLM tiap request:
                      [system prompt] + [summary note (kalau ada)] +
                      [N pesan terakhir mentah].
    """

    def __init__(self, system_prompt: str, max_tokens: int = 400, keep_last: int = 4):
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.keep_last = keep_last
        self.full_log: list[dict] = []
        self.raw_window: list[dict] = []  # pesan mentah yang belum di-summarize
        self.summary: str = ""  # rolling summary dari pesan-pesan lama

    def add(self, role: str, content: str):
        msg = {"role": role, "content": content}
        self.full_log.append(msg)
        self.raw_window.append(msg)
        self._maybe_summarize()

    def _build_context(self) -> list[dict]:
        ctx = [{"role": "system", "content": self.system_prompt}]
        if self.summary:
            ctx.append(
                {
                    "role": "system",
                    "content": f"[Ringkasan percakapan sebelumnya]\n{self.summary}",
                }
            )
        ctx.extend(self.raw_window)
        return ctx

    def _maybe_summarize(self):
        """Kalau raw_window sudah melebihi budget token, ringkas bagian
        LAMA (semua kecuali `keep_last` pesan terakhir) jadi summary, lalu
        buang bagian lama itu dari raw_window."""
        ctx = self._build_context()
        if messages_tokens(ctx) <= self.max_tokens:
            return
        if len(self.raw_window) <= self.keep_last:
            return  # gak ada lagi yang bisa diringkas tanpa buang pesan terbaru

        to_summarize = self.raw_window[: -self.keep_last]
        keep = self.raw_window[-self.keep_last :]

        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in to_summarize)
        prev_summary_note = f"Ringkasan sebelumnya: {self.summary}\n\n" if self.summary else ""
        prompt = (
            f"{prev_summary_note}Ringkas percakapan berikut jadi maksimal 3 kalimat padat, "
            f"fokus ke fakta & keputusan penting yang harus diingat untuk lanjutan chat:\n\n"
            f"{transcript}"
        )

        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        self.summary = resp.choices[0].message.content.strip()
        self.raw_window = keep

    def chat(self, user_message: str) -> str:
        self.add("user", user_message)
        context = self._build_context()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=context,
            temperature=0.5,
        )
        reply = resp.choices[0].message.content.strip()
        self.add("assistant", reply)
        return reply

    def stats(self) -> dict:
        sent_now = self._build_context()
        return {
            "total_turns_full_log": len(self.full_log),
            "tokens_if_full_log_sent": messages_tokens(self.full_log),
            "tokens_actually_sent": messages_tokens(sent_now),
            "has_summary": bool(self.summary),
            "raw_window_size": len(self.raw_window),
        }


def main():
    print("=" * 70)
    print("FASE 5: Conversation Memory Management (Sliding Window + Summary)")
    print("=" * 70)

    mem = ConversationMemory(
        system_prompt=(
            "Kamu asisten teknis yang jawab singkat & to the point dalam Bahasa Indonesia. "
            "Ingat detail proyek yang dibahas user."
        ),
        max_tokens=350,  # sengaja dibuat kecil biar summarization ke-trigger cepat di demo
        keep_last=3,
    )

    turns = [
        "Halo, aku lagi bikin service backend Go namanya 'refund-svc', pakai MongoDB.",
        "Service itu handle proses refund merchant, ada endpoint POST /refunds.",
        "Aku juga pakai maker-checker pattern, jadi tiap refund butuh approval dari checker.",
        "Sekarang aku mau tambahin retry mechanism kalau MongoDB timeout, pakai library apa yang bagus di Go?",
        "Oke, aku pakai exponential backoff. Btw nama service-ku tadi apa ya? Aku lupa nyebutin ulang.",
        "Terus pattern approval yang aku sebut di awal itu namanya apa?",
    ]

    for i, user_msg in enumerate(turns, start=1):
        print(f"\n--- Turn {i} ---")
        print(f"User: {user_msg}")
        reply = mem.chat(user_msg)
        print(f"Assistant: {reply}")
        s = mem.stats()
        print(
            f"[stats] full_log_turns={s['total_turns_full_log']} | "
            f"tokens_if_full_sent={s['tokens_if_full_log_sent']} | "
            f"tokens_actually_sent={s['tokens_actually_sent']} | "
            f"has_summary={s['has_summary']} | raw_window={s['raw_window_size']}"
        )

    print("\n" + "=" * 70)
    print("VERIFIKASI: LLM masih inget nama service & pattern dari turn awal")
    print("meskipun sudah di-summarize (bukan raw history lagi) -> lihat jawaban")
    print("turn 5 & 6 di atas.")
    print("=" * 70)

    final = mem.stats()
    print(f"\nFinal full_log tokens (kalau dikirim mentah semua): {final['tokens_if_full_log_sent']}")
    print(f"Final tokens yang ACTUALLY dikirim ke LLM tiap request: {final['tokens_actually_sent']}")
    if final["has_summary"]:
        print("\n--- Isi rolling summary saat ini ---")
        print(mem.summary)


if __name__ == "__main__":
    main()
