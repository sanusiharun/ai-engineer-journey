"""
🎯 PELAJARAN 12: Chain-of-Thought (CoT) Prompting
================================================

OBJEKTIF:
- Memahami teknik Chain-of-Thought untuk reasoning tasks
- Membandingkan direct answer vs step-by-step reasoning
- Belajar kapan CoT lebih efektif daripada direct prompting

KONSEP KUNCI:
- CoT: minta LLM berpikir langkah-demi-langkah sebelum jawaban akhir
- "Let's think step by step" adalah magic phrase untuk trigger CoT
- CoT meningkatkan akurasi untuk math, logic, dan reasoning tasks
- Zero-shot CoT: cukup tambah "think step by step", tanpa contoh

KAPAN PAKAI:
- Math problems (aritmatika, word problems)
- Logical reasoning (if-then, deduksi)
- Multi-step planning
- Complex decision making
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url=os.getenv("ROUTER_BASE_URL"),
    api_key=os.getenv("ROUTER_API_KEY")
)

def direct_answer(problem: str) -> str:
    """Jawab langsung tanpa reasoning"""
    response = client.chat.completions.create(
        model=os.getenv("ROUTER_MODEL"),
        messages=[
            {"role": "system", "content": "Jawab pertanyaan dengan singkat."},
            {"role": "user", "content": problem}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def chain_of_thought(problem: str) -> str:
    """Jawab dengan step-by-step reasoning"""
    response = client.chat.completions.create(
        model=os.getenv("ROUTER_MODEL"),
        messages=[
            {"role": "system", "content": "Jelaskan pemikiran kamu langkah demi langkah, lalu berikan jawaban akhir."},
            {"role": "user", "content": f"{problem}\n\nLet's think step by step:"}
        ],
        temperature=0
    )
    return response.choices[0].message.content

def few_shot_cot(problem: str) -> str:
    """Chain-of-Thought dengan contoh reasoning"""
    response = client.chat.completions.create(
        model=os.getenv("ROUTER_MODEL"),
        messages=[
            {"role": "system", "content": "Selesaikan soal dengan reasoning bertahap."},
            # Contoh 1 dengan reasoning
            {"role": "user", "content": "Jika 5 apel dijual 15 ribu, berapa harga 8 apel?"},
            {"role": "assistant", "content": """Mari kita hitung:
1. Harga per apel = 15,000 / 5 = 3,000
2. Harga 8 apel = 8 × 3,000 = 24,000

Jawaban: Rp 24,000"""},
            # Contoh 2 dengan reasoning
            {"role": "user", "content": "Toko buka jam 9 pagi, tutup jam 9 malam. Ada istirahat 1 jam siang. Berapa jam operasional?"},
            {"role": "assistant", "content": """Mari kita hitung:
1. Total jam dari 9 pagi - 9 malam = 12 jam
2. Kurangi istirahat = 12 - 1 = 11 jam

Jawaban: 11 jam operasional"""},
            # Task sebenarnya
            {"role": "user", "content": problem}
        ],
        temperature=0
    )
    return response.choices[0].message.content

if __name__ == "__main__":
    print("=" * 70)
    print("🧪 EKSPERIMEN: Direct Answer vs Chain-of-Thought")
    print("=" * 70)
    
    # Test Case 1: Math Word Problem
    problem1 = """Andi punya uang 100 ribu. Dia beli 3 buku @12 ribu dan 2 pulpen @5 ribu.
Berapa sisa uangnya?"""
    
    print(f"\n📝 Problem 1:\n{problem1}")
    
    print("\n🔹 Direct Answer:")
    direct1 = direct_answer(problem1)
    print(f"   {direct1}")
    
    print("\n🔹 Zero-Shot CoT:")
    cot1 = chain_of_thought(problem1)
    print(f"   {cot1}")
    
    # Test Case 2: Logical Reasoning
    print("\n" + "=" * 70)
    problem2 = """Jika semua kucing adalah hewan, dan Tom adalah kucing, apakah Tom adalah hewan?
Jika tidak semua hewan bisa terbang, apakah Tom bisa terbang?"""
    
    print(f"📝 Problem 2:\n{problem2}")
    
    print("\n🔹 Direct Answer:")
    direct2 = direct_answer(problem2)
    print(f"   {direct2}")
    
    print("\n🔹 Zero-Shot CoT:")
    cot2 = chain_of_thought(problem2)
    print(f"   {cot2}")
    
    # Test Case 3: Multi-step Planning
    print("\n" + "=" * 70)
    problem3 = """Restoran buka 7 hari seminggu. Senin-Jumat pengunjung 50 orang/hari,
weekend 100 orang/hari. Untung per pengunjung 20 ribu.
Berapa total untung seminggu?"""
    
    print(f"📝 Problem 3:\n{problem3}")
    
    print("\n🔹 Few-Shot CoT:")
    fewshot = few_shot_cot(problem3)
    print(f"   {fewshot}")
    
    print("\n" + "=" * 70)
    print("💡 KEY INSIGHTS:")
    print("=" * 70)
    print("1. CoT meningkatkan akurasi untuk reasoning tasks")
    print("2. 'Let's think step by step' adalah prompt ajaib")
    print("3. Few-shot CoT lebih konsisten dari zero-shot CoT")
    print("4. Tradeoff: CoT butuh lebih banyak tokens (lebih mahal)")
    print("5. Direct answer cukup untuk simple factual questions")
    print("\n✅ Pelajaran 12 selesai!")
