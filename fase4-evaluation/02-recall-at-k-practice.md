# Lesson 01 — Praktik Recall@K

## Dataset dan retriever

Eksperimen dijalankan pada project nyata `absen-kilat` menggunakan:

- 27 query evaluasi
- 81 file ground-truth yang relevan
- hybrid retriever lexical ∪ semantic
- nilai `K`: 3, 5, dan 20

Ground-truth bukan tebakan otomatis: setiap query memiliki daftar file relevan di `eval/eval_set.py`.

## Hasil eksperimen

| K | File relevan ditemukan | Total relevan | Recall@K |
|---:|---:|---:|---:|
| 3 | 27 | 81 | 0.333 |
| 5 | 41 | 81 | 0.506 |
| 20 | 69 | 81 | 0.852 |

Formula:

```text
Recall@K = dokumen relevan yang ditemukan di Top-K / total dokumen relevan
```

## Interpretasi

1. Top-3 hanya menemukan sekitar sepertiga file relevan. Untuk pertanyaan yang membutuhkan beberapa file, konteksnya terlalu sempit.
2. Top-5 membaik, tetapi masih hanya sekitar setengah file relevan.
3. Top-20 mencapai 0.852. Ini berarti 69 dari 81 file relevan masuk hasil Top-20.
4. Recall naik saat K naik, tetapi konteks yang dikirim ke LLM juga menjadi lebih besar. Jadi K bukan sekadar “semakin besar semakin baik”; perlu diuji bersama kualitas jawaban, latency, dan biaya.

## Contoh kegagalan yang terlihat

- Query geofencing: Top-3 berisi file test/admin yang mirip, tetapi melewatkan `attendance_service.go` dan `geofence.go`.
- Query notifikasi approval: hasil menemukan halaman/admin yang terkait, tetapi melewatkan repository dan service notifikasi.
- Query dashboard admin login: hasil menemukan shell admin, tetapi melewatkan halaman login dan auth service.

Ini menunjukkan masalah yang perlu dipelajari berikutnya: retriever bisa menemukan dokumen yang *mirip secara kata atau makna*, tetapi belum selalu menemukan seluruh dokumen yang dibutuhkan untuk menyelesaikan tugas.

## Artefak hasil

- `agentic-rag-hermes/eval/results-lesson01-k3.json`
- `agentic-rag-hermes/eval/results-lesson01-k5.json`
- `agentic-rag-hermes/eval/results-lesson01-k20.json`

## Kesimpulan lesson

Angka Recall@20 = `0.852` sekarang sudah diverifikasi ulang dengan harness dan dataset yang sama. Angka ini tidak boleh dibaca sebagai kualitas jawaban akhir; ini hanya mengukur coverage dokumen relevan pada tahap retrieval.

## Latihan modifikasi

1. Pilih satu query dengan recall rendah dan tambahkan kata domain yang lebih spesifik. Bandingkan hasilnya.
2. Ubah `top_k` menjadi 10 dan ukur lagi. Jangan menebak hasil sebelum menjalankan harness.
3. Untuk query `endpoint login email password mobile`, jelaskan kenapa menemukan halaman login tetapi melewatkan `user_repo.go` bisa tetap menghasilkan jawaban yang tidak lengkap.

Next: membuat script evaluation minimal yang reusable dan memisahkan perhitungan metrik dari retriever.

> Catatan: report JSON dihasilkan oleh eksekusi nyata; tidak ada credential/API key yang disimpan dalam artefak lesson.

---

## Teori singkat: Recall vs Precision

- **Recall**: dari semua dokumen yang seharusnya ditemukan, berapa yang berhasil ditemukan?
- **Precision**: dari semua dokumen yang ditemukan, berapa yang benar-benar relevan?

Top-20 dapat menaikkan recall, tetapi juga berpotensi menurunkan precision karena lebih banyak dokumen tidak relevan ikut masuk. Evaluasi production biasanya membutuhkan keduanya, lalu menghubungkannya dengan kualitas jawaban dan batas context window.

## Sumber kode

- `agentic-rag-hermes/eval/harness.py`
- `agentic-rag-hermes/eval/eval_set.py`
- `agentic-rag-hermes/retriever.py`
- dataset project: `/home/ubuntu/absen-kilat`

## Status

- Teori Recall@K: selesai
- Praktik Top-3/Top-5/Top-20: selesai dan terverifikasi
- Coding harness reusable: lesson berikutnya
- Regression gate: ditunda setelah metric helper selesai

---

*Dibuat dari hasil run lokal pada 28 Agustus 2026.*

## Detail formula per query

Untuk query dengan 3 file relevan, jika 2 muncul di Top-K maka recall query = `2/3 = 0.667`. Nilai keseluruhan di report dihitung micro-average: seluruh hit dibagi seluruh ground-truth files (`69/81 = 0.852` untuk K=20), bukan rata-rata pembulatan nilai per query.

Itu penting karena dua metode agregasi dapat menghasilkan angka berbeda ketika setiap query memiliki jumlah file relevan yang berbeda.

## Pertanyaan refleksi

- Apakah Top-3 cukup untuk pertanyaan yang hanya membutuhkan satu file?
- Kapan Top-20 menjadi terlalu mahal untuk dikirim ke LLM?
- Apakah file test seharusnya diberi bobot sama dengan file production code?
- Apakah ground-truth saat ini sudah lengkap, atau masih bias ke file yang kita ingat?

Pertanyaan-pertanyaan ini akan menjadi dasar lesson Precision, MRR, dan evaluasi kualitas jawaban.