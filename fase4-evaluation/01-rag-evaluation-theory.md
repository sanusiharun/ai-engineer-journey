# Lesson 01 — RAG Evaluation: Recall@K

## Tujuan

Memahami cara mengukur apakah retriever menemukan dokumen yang benar sebelum LLM membuat jawaban.

## Kenapa evaluasi diperlukan?

Jawaban yang terlihat bagus belum tentu berasal dari context yang benar. Evaluasi memisahkan dua pertanyaan:

1. **Retrieval:** apakah dokumen yang dibutuhkan masuk ke Top-K?
2. **Generation:** apakah model menggunakan context itu dengan benar?

Lesson ini fokus pada retrieval.

## Recall@K

Recall@K mengukur proporsi dokumen relevan yang berhasil ditemukan di K hasil teratas.

```text
Recall@K = dokumen relevan yang ditemukan di Top-K
           --------------------------------------
           seluruh dokumen relevan yang diharapkan
```

Contoh:

- Expected relevant documents: `doc-a`, `doc-b`
- Top-5 retriever: `doc-x`, `doc-b`, `doc-y`, `doc-z`, `doc-q`
- Relevant ditemukan: `1 dari 2`
- Recall@5: `0.5`

## Apa arti K?

- `K=3`: context lebih kecil, lebih cepat, tetapi risiko dokumen penting terlewat lebih besar.
- `K=5`: kompromi umum untuk inspeksi awal.
- `K=20`: peluang menemukan dokumen lebih tinggi, tetapi context dan biaya LLM dapat meningkat.

Jangan menyebut Recall@20 tinggi sebagai bukti jawaban selalu benar. Recall hanya mengukur tahap retrieval.

## Golden dataset

Golden dataset adalah kumpulan query yang memiliki label dokumen relevan yang sudah ditentukan manusia.

Contoh:

```json
{
  "query": "Bagaimana service dijalankan setelah reboot?",
  "relevant_doc_ids": ["infra.md", "systemd.service"]
}
```

Tanpa expected document, kita hanya punya daftar hasil—belum punya evaluasi yang dapat dipercaya.

## Recall vs Precision

- **Recall:** berapa banyak dokumen relevan yang berhasil ditemukan?
- **Precision:** dari dokumen yang ditemukan, berapa banyak yang relevan?

Retrieval RAG biasanya perlu menjaga keduanya. Recall yang tinggi dengan banyak noise dapat membuat LLM bingung.

## Kesalahan umum

- Mengukur jawaban LLM lalu menganggap itu retrieval score.
- Mengubah golden label agar cocok dengan output retriever.
- Membandingkan score dari dataset yang berbeda.
- Mengambil angka dari satu run tanpa menyimpan query dan konfigurasi.
- Menganggap `Recall@20 = 0.852` berarti 85.2% jawaban faktual.

## Latihan pemahaman

1. Jika ada 4 dokumen relevan dan Top-5 menemukan 3, berapa Recall@5?
2. Apakah Recall@20 tinggi menjamin jawaban LLM faithful? Jelaskan.
3. Mengapa golden dataset dibutuhkan?

## Praktik berikutnya

Kita akan menjalankan query yang sama pada `K=3`, `K=5`, dan `K=20`, lalu menyimpan hasilnya sebelum membuat evaluator otomatis.

---

Status: teori lesson 01 siap dipelajari; coding evaluator belum dimulai.
Tanggal: 2026-08-28
