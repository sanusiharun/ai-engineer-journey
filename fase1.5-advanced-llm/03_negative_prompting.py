"""
=== PELAJARAN 13: Negative Prompting & Constraints ===

OBJEKTIF:
- Mengontrol output LLM dengan constraint eksplisit
- Menggunakan negative prompting untuk menghindari output yang tidak diinginkan
- Membandingkan efektivitas berbagai teknik pembatasan

KONSEP KUNCI:
1. Negative Prompting: instruksi eksplisit tentang apa yang TIDAK boleh dilakukan
2. Format Constraints: membatasi format output (panjang, struktur, gaya)
3. Content Constraints: membatasi isi/topik yang boleh dibahas
4. Trade-offs: terlalu banyak constraint bisa membuat output kaku

KAPAN DIPAKAI:
- Production apps yang butuh output konsisten
- Content moderation/filtering
- Compliance requirements (GDPR, medical, legal)
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("ROUTER_API_KEY"),
    base_url=os.getenv("ROUTER_BASE_URL")
)

def test_without_constraints():
    """Baseline: tanpa constraint apapun"""
    print("=" * 60)
    print("TEST 1: Tanpa Constraint")
    print("=" * 60)
    
    response = client.chat.completions.create(
        model=os.getenv("ROUTER_MODEL", "vps_combos"),
        messages=[
            {"role": "user", "content": "Jelaskan apa itu Bitcoin dalam bahasa Indonesia"}
        ],
        temperature=0.7
    )
    
    output = response.choices[0].message.content
    print(f"Output:\n{output}\n")
    print(f"Panjang: {len(output)} karakter")
    return output

def test_with_length_constraint():
    """Constraint: batasi panjang output"""
    print("=" * 60)
    print("TEST 2: Dengan Length Constraint")
    print("=" * 60)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": """Jelaskan apa itu Bitcoin dalam bahasa Indonesia.

CONSTRAINT:
- Maksimal 3 kalimat
- Tidak boleh lebih dari 150 karakter
- Bahasa sederhana untuk pemula"""}
        ],
        temperature=0.7
    )
    
    output = response.choices[0].message.content
    print(f"Output:\n{output}\n")
    print(f"Panjang: {len(output)} karakter")
    print(f"Jumlah kalimat: {output.count('.')}")
    return output

def test_negative_prompting():
    """Negative prompting: instruksi tentang apa yang TIDAK boleh dilakukan"""
    print("=" * 60)
    print("TEST 3: Negative Prompting")
    print("=" * 60)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": """Jelaskan apa itu Bitcoin sebagai investasi dalam bahasa Indonesia.

ATURAN KETAT (WAJIB DIIKUTI):
❌ JANGAN gunakan istilah teknis seperti "blockchain", "mining", "cryptocurrency"
❌ JANGAN memberikan saran investasi eksplisit ("beli", "jual", "investasikan")
❌ JANGAN gunakan bullet points atau formatting markdown
❌ JANGAN lebih dari 100 kata

✅ HARUS gunakan analogi sehari-hari
✅ HARUS sebutkan risiko dengan jelas
✅ HARUS dalam 1 paragraf saja"""}
        ],
        temperature=0.7
    )
    
    output = response.choices[0].message.content
    print(f"Output:\n{output}\n")
    
    # Validasi constraint
    word_count = len(output.split())
    print(f"\n📊 Validasi Constraint:")
    print(f"  - Jumlah kata: {word_count} {'✅' if word_count <= 100 else '❌'}")
    print(f"  - Ada 'blockchain': {'❌ VIOLATION' if 'blockchain' in output.lower() else '✅'}")
    print(f"  - Ada 'mining': {'❌ VIOLATION' if 'mining' in output.lower() else '✅'}")
    print(f"  - Ada 'beli/jual': {'❌ VIOLATION' if any(x in output.lower() for x in ['beli', 'jual']) else '✅'}")
    print(f"  - Ada bullet points: {'❌ VIOLATION' if '•' in output or '-' in output[:10] else '✅'}")
    
    return output

def test_format_constraint_json():
    """Constraint: format output spesifik (JSON)"""
    print("=" * 60)
    print("TEST 4: Format Constraint (JSON)")
    print("=" * 60)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": """Analisis sentimen kalimat: "Bitcoin naik 10% hari ini!"

Output HARUS dalam format JSON seperti ini:
{
  "sentiment": "positive/negative/neutral",
  "confidence": 0.0-1.0,
  "keywords": ["kata1", "kata2"]
}

CONSTRAINT:
- HANYA return JSON, tidak ada teks tambahan
- Field harus persis seperti contoh
- Confidence harus angka desimal"""}
        ],
        temperature=0.3
    )
    
    output = response.choices[0].message.content
    print(f"Output:\n{output}\n")
    
    # Coba parse sebagai JSON
    import json
    try:
        parsed = json.loads(output)
        print("✅ Valid JSON!")
        print(f"  Sentiment: {parsed.get('sentiment')}")
        print(f"  Confidence: {parsed.get('confidence')}")
        print(f"  Keywords: {parsed.get('keywords')}")
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON: {e}")
    
    return output

def test_multi_constraint():
    """Kombinasi multiple constraints"""
    print("=" * 60)
    print("TEST 5: Multiple Constraints")
    print("=" * 60)
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": """Kamu adalah asisten keuangan untuk anak-anak.

PERSONA CONSTRAINT:
- Gunakan bahasa anak SD (8-10 tahun)
- Gunakan emoji yang sesuai
- Hindari jargon finansial

SAFETY CONSTRAINT:
- Jangan pernah menyarankan investasi spesifik
- Selalu sebutkan "tanya orang tua dulu"
- Fokus pada edukasi, bukan profit"""},
            {"role": "user", "content": """Bagaimana cara menghasilkan uang dari Bitcoin?

FORMAT:
- 3 poin singkat
- Setiap poin max 20 kata
- Akhiri dengan disclaimer keamanan"""}
        ],
        temperature=0.7
    )
    
    output = response.choices[0].message.content
    print(f"Output:\n{output}\n")
    
    return output

if __name__ == "__main__":
    print("\n🎯 PELAJARAN 13: Negative Prompting & Constraints\n")
    
    # Jalankan semua test
    test_without_constraints()
    print("\n" + "="*60 + "\n")
    
    test_with_length_constraint()
    print("\n" + "="*60 + "\n")
    
    test_negative_prompting()
    print("\n" + "="*60 + "\n")
    
    test_format_constraint_json()
    print("\n" + "="*60 + "\n")
    
    test_multi_constraint()
    
    print("\n" + "="*60)
    print("💡 KEY INSIGHTS:")
    print("="*60)
    print("""
1. Negative prompting efektif untuk menghindari output yang tidak diinginkan
2. Constraint eksplisit (emoji, format, panjang) membuat output lebih predictable
3. Validasi otomatis penting untuk memastikan constraint diikuti
4. Trade-off: lebih banyak constraint = lebih kaku, tapi lebih konsisten
5. Combine dengan system prompt untuk constraint yang persistent

REAL-WORLD USE CASES:
- Customer support chatbot (avoid refund promises, stay on-brand)
- Content generation for kids (safety constraints)
- Legal/medical AI (compliance requirements)
- API response formatting (JSON schema enforcement)
""")
