"""
Fase 2.5: RAG dengan Vector Database (ChromaDB)
=================================================
Di Fase 2 (04_rag_basics.py) kita bikin RAG manual pakai TF-IDF + cosine
similarity, semua stdlib Python, biar kelihatan gak ada sihir di baliknya.

Sekarang kita naik level: pakai VECTOR DATABASE beneran (ChromaDB) dan
EMBEDDING MODEL beneran (bukan TF-IDF lagi).

Apa bedanya sama Fase 2?
    - TF-IDF cuma ngitung frekuensi kata -> gak paham makna/sinonim.
      Contoh: "server" dan "mesin" dianggap beda total walau makna mirip.
    - Embedding model (di sini: all-MiniLM-L6-v2, jalan LOKAL lewat
      ONNX runtime, TANPA butuh API call sama sekali) mengubah teks jadi
      vektor 384-dimensi yang menangkap MAKNA, bukan cuma kata persis.
      Dua kalimat yang mirip maknanya akan punya vektor yang "dekat"
      walau kata-katanya beda.
    - ChromaDB adalah database yang dioptimasi buat nyimpen jutaan vektor
      dan nyari yang paling mirip dengan CEPAT (pakai index khusus, HNSW),
      dibanding loop manual seperti di Fase 2.

Kenapa embedding-nya jalan LOKAL, bukan lewat 9router?
    Sudah dicek: 9router (vps_combos) di VPS ini BELUM support endpoint
    /v1/embeddings ("Invalid model format"). Untungnya ChromaDB punya
    default embedding function bawaan (all-MiniLM-L6-v2) yang download
    sekali lalu jalan 100% lokal di CPU, jadi RAG tetap bisa dites full
    tanpa API embedding. Bagian GENERATION (jawab pertanyaan pakai LLM)
    tetap pakai 9router / vps_combos seperti biasa -- gak ada model
    berbayar sama sekali di pipeline ini.

Alur RAG dengan ChromaDB (sama seperti Fase 2, cuma toolingnya beda):
    1. INDEXING  -> chunk dokumen, ChromaDB otomatis bikin embedding tiap
                    chunk pakai MiniLM lokal, simpan ke collection.
    2. RETRIEVAL -> query pertanyaan, ChromaDB otomatis embed & cari
                    top-k chunk paling mirip pakai cosine similarity di
                    index HNSW-nya.
    3. GENERATION-> kirim chunk relevan + pertanyaan ke LLM (vps_combos
                    lewat 9router) buat dijawab.

Cara jalanin:
    venv/bin/python 05_rag_chromadb.py
"""

import os

import chromadb
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# --- Dokumen contoh: catatan tentang infrastruktur user sendiri ---------
DOCS = [
    (
        "doc_9router",
        "9router (alias OmniRoute) adalah AI gateway/router yang jalan sebagai "
        "systemd service di VPS milik user, di port 20128. Fungsinya "
        "meneruskan permintaan LLM ke berbagai provider dengan fitur "
        "credential pooling dan auto-fallback kalau satu provider gagal.",
    ),
    (
        "doc_hermes",
        "Hermes adalah AI agent milik user yang berjalan di VPS yang sama. "
        "Model default Hermes bernama vps_combos, providernya diatur "
        "'custom' dengan base_url http://localhost:20128/v1, artinya semua "
        "request LLM dari Hermes lewat 9router dulu, bukan langsung ke "
        "provider seperti OpenAI atau Anthropic.",
    ),
    (
        "doc_multiuser",
        "User berencana bikin setup multi-user: orang kedua bernama Tama "
        "akan chat lewat bot Telegram terpisah. Supaya penggunaan token "
        "per-orang bisa dipisah, rencananya pakai fitur allowedConnections "
        "atau allowedCombos di 9router, yaitu API key yang di-scope, "
        "bukan bikin instance 9router baru.",
    ),
    (
        "doc_rag",
        "RAG (Retrieval-Augmented Generation) adalah teknik memberi LLM "
        "'contekan' dari dokumen sendiri sebelum menjawab, supaya jawaban "
        "grounded ke fakta yang diberikan alih-alih LLM mengarang jawaban "
        "dari ingatan trainingnya sendiri (halusinasi).",
    ),
    (
        "doc_embedding",
        "Embedding adalah representasi numerik dari teks berupa vektor, "
        "yang menangkap makna kalimat. Kalimat dengan makna mirip akan "
        "punya vektor yang berdekatan secara matematis, diukur biasanya "
        "pakai cosine similarity atau dot product.",
    ),
]

QUESTIONS = [
    "Apa itu 9router dan jalan di port berapa?",
    "Bagaimana rencana user memisahkan penggunaan token antar user?",
    "Kenapa RAG penting buat LLM?",
]


def build_collection():
    """Bikin (atau reset) ChromaDB collection lalu isi dengan dokumen."""
    client = chromadb.PersistentClient(path="./chroma_store")
    # Reset collection kalau sudah ada, biar demo ini idempotent.
    try:
        client.delete_collection("infra_notes")
    except Exception:
        pass
    collection = client.create_collection(name="infra_notes")

    ids = [doc_id for doc_id, _ in DOCS]
    texts = [text for _, text in DOCS]
    # add() otomatis: embed tiap teks pakai default embedding function
    # (all-MiniLM-L6-v2, lokal via ONNX) lalu simpan ke index HNSW.
    collection.add(ids=ids, documents=texts)
    return collection


def retrieve(collection, question, top_k=2):
    """Cari top_k chunk paling relevan buat 1 pertanyaan."""
    result = collection.query(query_texts=[question], n_results=top_k)
    docs = result["documents"][0]
    distances = result["distances"][0]
    ids = result["ids"][0]
    return list(zip(ids, docs, distances))


def answer_with_llm(client, model, question, context_chunks):
    context_text = "\n\n".join(f"[{cid}] {text}" for cid, text, _ in context_chunks)
    prompt = (
        "Jawab pertanyaan HANYA berdasarkan konteks di bawah ini. "
        "Kalau konteks gak cukup, bilang gak tau.\n\n"
        f"Konteks:\n{context_text}\n\nPertanyaan: {question}"
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Kamu asisten yang jawab singkat, padat, berdasarkan konteks yang diberikan."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def main():
    print("=" * 70)
    print("FASE 2.5: RAG dengan ChromaDB (vector database + embedding lokal)")
    print("=" * 70)

    print("\n[1/3] INDEXING: embed & simpan dokumen ke ChromaDB...")
    collection = build_collection()
    print(f"   -> {collection.count()} dokumen ter-index di collection 'infra_notes'")

    client = OpenAI(
        base_url=os.getenv("ROUTER_BASE_URL"),
        api_key=os.getenv("ROUTER_API_KEY"),
    )
    model = os.getenv("ROUTER_MODEL")

    for q in QUESTIONS:
        print(f"\n[2/3] RETRIEVAL untuk pertanyaan: {q!r}")
        hits = retrieve(collection, q, top_k=2)
        for cid, text, dist in hits:
            preview = text[:60].replace("\n", " ")
            print(f"   - {cid} (distance={dist:.4f}): {preview}...")

        print("[3/3] GENERATION (lewat 9router / vps_combos)...")
        answer = answer_with_llm(client, model, q, hits)
        print(f"   JAWABAN: {answer}")

    print("\n" + "=" * 70)
    print("Selesai. Semua embedding jalan LOKAL (MiniLM via ONNX),")
    print("semua generation lewat 9router (vps_combos) -- no paid model.")
    print("=" * 70)


if __name__ == "__main__":
    main()
