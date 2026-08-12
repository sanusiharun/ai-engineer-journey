"""
Fase 2: RAG Dasar (Retrieval-Augmented Generation)
====================================================
Tujuan: paham konsep INTI di balik "chatbot yang jawab dari dokumen sendiri"
(RAG), tanpa dulu pakai library vector-db yang berat (ChromaDB dsb — itu
di Fase 2.5). Di sini kita bikin versi PALING SEDERHANA dari retrieval:
TF-IDF + cosine similarity, semua pakai Python murni (stdlib doang, tanpa
numpy/sklearn) biar konsepnya kelihatan jelas, gak ketutup magic library.

Kenapa RAG penting?
    LLM base cuma tau apa yang dia pelajari waktu training. Dia GAK TAU
    dokumen internal kamu, catatan pribadi, atau data yang baru banget.
    RAG = kasih LLM "contekan" relevan dari dokumen kamu sendiri sebelum
    dia jawab. Jadi jawabannya grounded ke fakta yang kamu kasih, bukan
    ngarang (halusinasi).

Alur RAG (3 langkah):
    1. INDEXING  -> pecah dokumen jadi potongan (chunk), ubah tiap chunk
                    jadi representasi numerik (di sini: vektor TF-IDF).
    2. RETRIEVAL -> waktu ada pertanyaan, ubah pertanyaan itu jadi vektor
                    juga, cari chunk yang paling "mirip" (cosine similarity).
    3. GENERATION-> kasih chunk paling relevan itu ke LLM sebagai konteks,
                    LLM jawab pertanyaan BERDASARKAN konteks itu.

Kenapa TF-IDF dulu, bukan embedding model?
    - Embedding beneran (lewat API) butuh endpoint /v1/embeddings yang
      support model tsb. Sudah dicek: 9router (vps_combos) di VPS ini
      BELUM support endpoint embeddings ("Invalid model format" saat
      dicoba). Jadi kita pakai TF-IDF: teknik retrieval klasik dari
      information retrieval, tanpa perlu API embedding sama sekali,
      pure hitung statistik kata dari teks itu sendiri.
    - Ini juga bagus buat belajar: kamu lihat "vector search" itu
      sebenarnya cuma matematika (dot product / cosine) di atas angka,
      gak ada sihir. Nanti pas ganti ke embedding beneran (Fase 2.5),
      alurnya SAMA PERSIS, cuma cara bikin vektornya beda.

Dokumen contoh: catatan tentang infrastruktur AI Engineer Journey si user
sendiri (9router, VPS, Hermes) -- biar kerasa relevan & real.

Cara jalanin:
    venv/bin/python 04_rag_basics.py
"""

import math
import os
import re
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("ROUTER_BASE_URL"),
    api_key=os.getenv("ROUTER_API_KEY"),
)
MODEL = os.getenv("ROUTER_MODEL", "vps_combos")

# ---------------------------------------------------------------------------
# 1. "Knowledge base": beberapa dokumen pendek tentang infra si user sendiri.
#    Di dunia nyata ini bisa dari file .txt/.md/.pdf yang kamu baca beneran.
# ---------------------------------------------------------------------------
DOCUMENTS = [
    {
        "id": "doc1",
        "title": "9router",
        "text": (
            "9router adalah package npm global yang jalan sebagai systemd "
            "service di VPS milik user, terdaftar di "
            "/etc/systemd/system/9router.service. Dia berfungsi sebagai "
            "LLM router alias gateway: menerima request format OpenAI-compatible "
            "di port 20128, lalu meneruskan ke berbagai provider LLM di belakangnya "
            "dengan fitur credential pooling dan auto-fallback kalau satu provider gagal. "
            "9router juga dikenal dengan nama OmniRoute."
        ),
    },
    {
        "id": "doc2",
        "title": "vps_combos",
        "text": (
            "vps_combos adalah nama model gabungan (combo) yang didaftarkan di "
            "9router. Model ini bukan satu model tunggal, tapi kombinasi beberapa "
            "model backend (misalnya Claude, Gemini, dsb dari berbagai provider) "
            "yang dipilih otomatis oleh router. Hermes Agent milik user dikonfigurasi "
            "untuk memakai model vps_combos lewat base_url http://localhost:20128/v1, "
            "provider custom, alih-alih connect langsung ke provider aslinya."
        ),
    },
    {
        "id": "doc3",
        "title": "Hermes Agent",
        "text": (
            "Hermes Agent adalah asisten AI yang dipasang user di VPS-nya, bisa "
            "diakses lewat Telegram bot. User sedang eksplorasi setup multi-user, "
            "di mana orang kedua bernama Tama juga akan chat lewat bot Telegram "
            "terpisah, dengan token/API usage yang diisolasi per orang memakai "
            "fitur allowedConnections/allowedCombos milik 9router untuk membuat "
            "API key yang di-scope, tanpa perlu menjalankan instance 9router duplikat."
        ),
    },
    {
        "id": "doc4",
        "title": "AI Engineer Journey",
        "text": (
            "AI Engineer Journey adalah program belajar milik user, disimpan di folder "
            "~/ai-engineer-journey/. Progress belajar dicatat di file .progress.json. "
            "User sudah menyelesaikan Fase 1 (basic LLM call, system prompt, "
            "temperature, streaming), Fase 1.5 (function calling), dan Fase 1.6 "
            "(structured output / JSON mode). Fase 2 membahas RAG dasar."
        ),
    },
]


# ---------------------------------------------------------------------------
# 2. TF-IDF sederhana dari nol (pure Python, gak pakai library eksternal)
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Ubah teks jadi list kata lowercase, buang tanda baca."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_tfidf_index(documents: list[dict]) -> dict:
    """
    Bangun index TF-IDF dari semua dokumen.
    TF (Term Frequency)     = seberapa sering kata muncul di 1 dokumen.
    IDF (Inverse Doc Freq)  = kata yang muncul di SEDIKIT dokumen dianggap
                              lebih "khas"/penting daripada kata umum
                              (misal "yang", "di", "user") yang muncul di semua.
    """
    doc_tokens = {d["id"]: tokenize(d["text"]) for d in documents}
    n_docs = len(documents)

    # Document frequency: di berapa dokumen kata X muncul?
    df = Counter()
    for tokens in doc_tokens.values():
        for word in set(tokens):
            df[word] += 1

    idf = {word: math.log((n_docs + 1) / (freq + 1)) + 1 for word, freq in df.items()}

    # Vektor TF-IDF per dokumen: dict {kata: skor}
    doc_vectors = {}
    for doc_id, tokens in doc_tokens.items():
        tf = Counter(tokens)
        length = len(tokens) or 1
        doc_vectors[doc_id] = {
            word: (count / length) * idf.get(word, 0.0) for word, count in tf.items()
        }

    return {"idf": idf, "doc_vectors": doc_vectors}


def vectorize_query(query: str, idf: dict) -> dict:
    tokens = tokenize(query)
    tf = Counter(tokens)
    length = len(tokens) or 1
    return {word: (count / length) * idf.get(word, 0.0) for word, count in tf.items()}


def cosine_similarity(vec_a: dict, vec_b: dict) -> float:
    """cos(theta) = (A . B) / (|A| * |B|) -- ukuran kemiripan arah dua vektor."""
    common_words = set(vec_a) & set(vec_b)
    dot_product = sum(vec_a[w] * vec_b[w] for w in common_words)

    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def retrieve(query: str, index: dict, documents: list[dict], top_k: int = 2) -> list[dict]:
    """Cari top_k dokumen paling relevan terhadap query."""
    query_vec = vectorize_query(query, index["idf"])
    scores = []
    for doc in documents:
        sim = cosine_similarity(query_vec, index["doc_vectors"][doc["id"]])
        scores.append((sim, doc))
    scores.sort(key=lambda pair: pair[0], reverse=True)
    return scores[:top_k]


# ---------------------------------------------------------------------------
# 3. Generation: kasih hasil retrieval sebagai konteks ke LLM
# ---------------------------------------------------------------------------
def rag_answer(query: str, index: dict, documents: list[dict]) -> tuple[str, list[dict]]:
    top_results = retrieve(query, index, documents, top_k=2)

    context_text = "\n\n".join(
        f"[{doc['title']}]\n{doc['text']}" for _, doc in top_results
    )

    system_prompt = (
        "Kamu adalah asisten yang HANYA boleh jawab berdasarkan KONTEKS yang "
        "diberikan di bawah. Kalau informasinya tidak ada di konteks, bilang "
        "terus terang tidak tahu -- jangan mengarang. Jawab singkat dalam "
        "Bahasa Indonesia casual."
    )
    user_prompt = f"KONTEKS:\n{context_text}\n\nPERTANYAAN: {query}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )
    answer = response.choices[0].message.content
    return answer, top_results


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("RAG DASAR: TF-IDF retrieval + LLM generation (via 9router)")
    print("=" * 70)

    index = build_tfidf_index(DOCUMENTS)
    print(f"\n[Indexing] {len(DOCUMENTS)} dokumen berhasil di-index jadi vektor TF-IDF.")

    test_queries = [
        "Port berapa yang dipakai 9router dan apa fungsinya?",
        "Kenapa Tama butuh isolasi token terpisah dari user utama?",
        "Fase belajar apa saja yang sudah diselesaikan sejauh ini?",
    ]

    for i, query in enumerate(test_queries, start=1):
        print(f"\n{'-' * 70}")
        print(f"Query {i}: {query}")

        retrieved = retrieve(query, index, DOCUMENTS, top_k=2)
        print("Dokumen yang di-retrieve (top 2, dengan skor similarity):")
        for score, doc in retrieved:
            print(f"  - {doc['title']}  (skor: {score:.4f})")

        answer, _ = rag_answer(query, index, DOCUMENTS)
        print(f"\nJawaban LLM (grounded ke dokumen di atas):\n{answer}")

    print(f"\n{'=' * 70}")
    print("Selesai. Semua jawaban di atas didasarkan pada dokumen lokal,")
    print("bukan dari pengetahuan bawaan LLM -- itulah inti RAG.")
    print("=" * 70)
