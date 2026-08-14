"""
🎯 PELAJARAN 11: Few-Shot Prompting
===================================

OBJEKTIF:
- Memahami teknik few-shot learning untuk meningkatkan akurasi LLM
- Membandingkan zero-shot vs few-shot prompting
- Belajar cara menyusun contoh yang efektif

KONSEP KUNCI:
- Few-shot: memberikan beberapa contoh input-output sebelum task sebenarnya
- Format consistency: contoh harus punya struktur yang sama
- Example quality > quantity: 2-3 contoh yang bagus lebih baik dari 10 contoh biasa

KAPAN PAKAI:
- Task yang butuh format output spesifik
- Domain-specific tasks (medical, legal, technical)
- Ekstraksi terstruktur dari teks tidak terstruktur
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("ROUTER_BASE_URL"),
    api_key=os.getenv("ROUTER_API_KEY")
)

def zero_shot_classification(text: str) -> str:
    """Klasifikasi sentimen tanpa contoh"""
    response = client.chat.completions.create(
        model=os.getenv("ROUTER_MODEL"),
        messages=[
            {"role": "system", "content": "Klasifikasikan sentimen teks menjadi: POSITIF, NEGATIF, atau NETRAL."},
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def few_shot_classification(text: str) -> str:
    """Klasifikasi sentimen dengan contoh"""
    response = client.chat.completions.create(
        model=os.getenv("ROUTER_MODEL"),
        messages=[
            {"role": "system", "content": "Klasifikasikan sentimen teks menjadi: POSITIF, NEGATIF, atau NETRAL."},
            # Contoh 1
            {"role": "user", "content": "Produk ini luar biasa! Pengiriman cepat dan kualitas bagus."},
            {"role": "assistant", "content": "POSITIF"},
            # Contoh 2
            {"role": "user", "content": "Mengecewakan, barang rusak dan customer service tidak responsif."},
            {"role": "assistant", "content": "NEGATIF"},
            # Contoh 3
            {"role": "user", "content": "Barang sudah sampai, sesuai deskripsi."},
            {"role": "assistant", "content": "NETRAL"},
            # Task sebenarnya
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def few_shot_extraction(text: str) -> str:
    """Ekstraksi terstruktur dengan few-shot"""
    response = client.chat.completions.create(
        model=os.getenv("ROUTER_MODEL"),
        messages=[
            {"role": "system", "content": "Ekstrak informasi produk dalam format: Nama|Harga|Rating"},
            # Contoh 1
            {"role": "user", "content": "Beli Laptop Gaming ROG Strix harga 15 juta, rating 4.5 bintang"},
            {"role": "assistant", "content": "Laptop Gaming ROG Strix|15000000|4.5"},
            # Contoh 2
            {"role": "user", "content": "Mouse wireless Logitech cuma 250 ribu, dapat 4.8 stars"},
            {"role": "assistant", "content": "Mouse Wireless Logitech|250000|4.8"},
            # Task sebenarnya
            {"role": "user", "content": text}
        ],
        temperature=0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("=" * 60)
    print("🧪 EKSPERIMEN: Zero-Shot vs Few-Shot Prompting")
    print("=" * 60)
    
    # Test case 1: Sentimen ambigu
    test_text = "Harganya mahal tapi kualitasnya oke lah"
    
    print("\n📝 Test Text:", test_text)
    print("\n🔹 Zero-Shot Result:")
    zero_result = zero_shot_classification(test_text)
    print(f"   {zero_result}")
    
    print("\n🔹 Few-Shot Result:")
    few_result = few_shot_classification(test_text)
    print(f"   {few_result}")
    
    # Test case 2: Ekstraksi terstruktur
    print("\n" + "=" * 60)
    print("🧪 EKSPERIMEN: Ekstraksi Terstruktur")
    print("=" * 60)
    
    product_text = "Mechanical Keyboard Keychron K2 dijual 1.2 juta, review 4.7 dari 5"
    print("\n📝 Input:", product_text)
    print("\n🔹 Few-Shot Extraction:")
    extraction = few_shot_extraction(product_text)
    print(f"   {extraction}")
    
    # Parse hasil
    if "|" in extraction:
        parts = extraction.split("|")
        print("\n✅ Parsed:")
        print(f"   • Nama: {parts[0]}")
        if parts[1].isdigit():
            print(f"   • Harga: Rp {int(parts[1]):,}")
        else:
            print(f"   • Harga: {parts[1]}")
        print(f"   • Rating: {parts[2]}/5")
    
    print("\n" + "=" * 60)
    print("💡 KEY INSIGHTS:")
    print("=" * 60)
    print("1. Few-shot lebih konsisten untuk format output spesifik")
    print("2. Contoh mengajarkan model 'style' yang diharapkan")
    print("3. Gunakan temperature=0 untuk output deterministik")
    print("4. Few-shot cocok untuk domain-specific tasks")
    print("\n✅ Pelajaran 11 selesai!")
