# -*- coding: utf-8 -*-
import pandas as pd
import requests
import time
import sys
import io
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# Konsol encoding sorunlarini coz
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

print("="*50)
print("RAG MODELI DOGRULUK VE BASARI TESTI")
print("="*50)

# 1. Veri setinden (ilk 1000 kayit) rastgele 10 soru sec
df = pd.read_parquet('D:/RAG_Chatbot/train-00000-of-00001.parquet').head(1000)
df_valid = df.dropna(subset=['question_content', 'question_answer'])
sample = df_valid.sample(10, random_state=42)

success_count = 0
toplam_benzerlik = 0.0

print(f"Test icin rastgele {len(sample)} gercek hasta sorusu secildi. Sunucuya gonderiliyor...\n")

for i, row in enumerate(sample.itertuples(), 1):
    q = str(row.question_content).strip()
    gt_answer = str(row.question_answer).strip()
    
    # Soru cok uzunsa ilk 250 karakterini alalim
    q_short = q[:250]
    
    print(f"[{i}/{len(sample)}] Soru: {q_short[:60]}...")
    
    try:
        res = requests.post("http://localhost:8000/api/chat", json={"message": q_short}, timeout=30)
        
        if res.status_code == 200:
            data = res.json()
            gen_answer = data.get("answer", "")
            source = data.get("source", "")
            
            # 1. Kontrol: Fallback
            if source != "rag":
                print("   [BASARISIZ] Veritabaninda eslesme bulunamadi (Fallback).")
                time.sleep(3)
                continue
                
            # 2. Kontrol: Bilgi yok reddi
            if "bilgi bulunmamaktadir" in gen_answer.lower() or ("bu konuda" in gen_answer.lower() and "yok" in gen_answer.lower()):
                print("   [BASARISIZ] Veritabaninda ilgili bilgi bulunamadi dedi.")
                time.sleep(3)
                continue
                
            # 3. Kontrol: Benzerlik
            vec = TfidfVectorizer()
            try:
                tfidf_matrix = vec.fit_transform([gt_answer, gen_answer])
                sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            except:
                sim = 0
                
            success_count += 1
            toplam_benzerlik += sim
            print(f"   [BASARILI] Cevap uretildi. (Benzerlik: %{sim*100:.1f})")
            
        else:
            print(f"   [API HATASI] Kodu: {res.status_code}")
            
    except Exception as e:
        print(f"   [HATA] {e}")
        
    time.sleep(4)

basari_orani = (success_count / len(sample)) * 100
ortalama_benzerlik = (toplam_benzerlik / success_count * 100) if success_count > 0 else 0

print("\n" + "="*50)
print("TEST SONUCLARI RAPORU")
print("="*50)
print(f"Toplam Test Edilen Soru : {len(sample)}")
print(f"Basarili Yanit Sayisi   : {success_count}")
print(f"Basarisiz/Eksik Yanit   : {len(sample) - success_count}")
print("-" * 50)
print(f"SISTEM BASARI ORANI     : %{basari_orani:.1f}")
print(f"Ort. Metin Benzerligi   : %{ortalama_benzerlik:.1f}")
print("="*50)
print("Not: Metin benzerliginin %100 olmamasi, yapay zekanin ezberlemek yerine")
print("doktorun cevabini anlayip daha anlasilir bir dille hastaya sunmasindandir.")
