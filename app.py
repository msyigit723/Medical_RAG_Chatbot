# -*- coding: utf-8 -*-
import pandas as pd
from dotenv import load_dotenv
import os
import sys
import time
import io

# Konsol encoding sorunlarini coz
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# .env dosyasindaki API anahtarlarini yukle
load_dotenv()

# ============================================================
# Adim 1: Hangi LLM kullanilacagini belirle
# ============================================================
def get_llm():
    """OpenAI veya Gemini API anahtarina gore LLM secer."""
    
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if openai_key and not openai_key.startswith("sk-senin"):
        from langchain_openai import ChatOpenAI
        print("[INFO] OpenAI GPT modeli kullaniliyor...")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    elif google_key and not google_key.startswith("senin"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("[INFO] Google Gemini modeli kullaniliyor...")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
    else:
        print("=" * 60)
        print("HATA: API anahtari bulunamadi!")
        print("=" * 60)
        print()
        print("Lutfen .env dosyasina asagidakilerden birini ekleyin:")
        print()
        print('  OpenAI icin:  OPENAI_API_KEY="sk-..."')
        print('  Gemini icin:  GOOGLE_API_KEY="AIza..."')
        print()
        print(".env dosyasi proje klasorunde (D:\\RAG_Chatbot\\.env) olmali.")
        print("=" * 60)
        sys.exit(1)

# ============================================================
# Adim 2: Veri Isleme (Data Engineering)
# ============================================================
print("=" * 60)
print("   TIBBI RAG CHATBOT - Saglik Bilgi Asistani")
print("=" * 60)
print()

# Parquet dosyasinin yolunu belirle
PROJE_DIZINI = os.path.dirname(os.path.abspath(__file__))
VERI_DOSYASI = os.path.join(PROJE_DIZINI, "train-00000-of-00001.parquet")
CHROMA_DB_DIZINI = os.path.join(PROJE_DIZINI, "chroma_db")

# Daha once ChromaDB olusturulmus mu kontrol et
chroma_var_mi = os.path.exists(CHROMA_DB_DIZINI) and len(os.listdir(CHROMA_DB_DIZINI)) > 0

# Turkce destekli ucretsiz metin gomme (embedding) modeli
print("[1/4] Embedding modeli yukleniyor...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

if chroma_var_mi:
    # Daha once olusturulmus vektor veritabanini yukle
    print("[2/4] Mevcut vektor veritabani yukleniyor...")
    vectorstore = Chroma(
        persist_directory=CHROMA_DB_DIZINI,
        embedding_function=embeddings
    )
    belge_sayisi = vectorstore._collection.count()
    print(f"       Veritabaninda {belge_sayisi} belge mevcut.")
    
    # Eger belge sayisi 0 ise yeniden olustur
    if belge_sayisi == 0:
        print("       [UYARI] Veritabani bos! Yeniden olusturuluyor...")
        chroma_var_mi = False
else:
    print("[2/4] Vektor veritabani ilk kez olusturulacak...")

if not chroma_var_mi:
    # Veri setini yukle ve isleyip ChromaDB'ye kaydet
    print("       Veri seti yukleniyor ve isleniyor...")
    
    if not os.path.exists(VERI_DOSYASI):
        print(f"\nHATA: Veri seti dosyasi bulunamadi: {VERI_DOSYASI}")
        print("Lutfen Zenodo'dan indirdiginiz parquet dosyasini proje klasorune koyun.")
        sys.exit(1)
    
    df = pd.read_parquet(VERI_DOSYASI)
    print(f"       Toplam {len(df)} kayit yuklendi.")
    print(f"       Sutunlar: {list(df.columns)}")
    
    # Workshop icin verinin ilk 1000 satirini aliyoruz (Hiz icin)
    KAYIT_SAYISI = 1000
    df_subset = df.head(KAYIT_SAYISI)
    
    # Eski chroma_db varsa sil
    if os.path.exists(CHROMA_DB_DIZINI):
        import shutil
        shutil.rmtree(CHROMA_DB_DIZINI)
        print("       Eski veritabani silindi.")
    
    # Veriyi LangChain Document formatina donusturme
    documents = []
    for index, row in df_subset.iterrows():
        # Doktor bilgisi
        doktor_bilgisi = ""
        doktor_unvan = str(row.get('doctor_title', ''))
        doktor_uzmanlik = str(row.get('doctor_speciality', ''))
        
        if doktor_unvan and doktor_unvan != 'nan' and doktor_unvan != 'None':
            doktor_bilgisi += f"Doktor: {doktor_unvan}"
        if doktor_uzmanlik and doktor_uzmanlik != 'nan' and doktor_uzmanlik != 'None':
            uzmanlik_temiz = doktor_uzmanlik.replace('-', ' ').title()
            doktor_bilgisi += f" - Uzmanlik: {uzmanlik_temiz}"
        
        soru = str(row.get('question_content', ''))
        cevap = str(row.get('question_answer', ''))
        
        if not soru or not cevap or soru == 'nan' or cevap == 'nan':
            continue
        
        # Daha yapilandirilmis icerik formati
        icerik = ""
        if doktor_bilgisi:
            icerik += f"{doktor_bilgisi}\n"
        icerik += f"Hasta Sorusu: {soru}\nDoktor Cevabi: {cevap}"
        
        metadata = {
            "id": index,
            "uzmanlik": doktor_uzmanlik if doktor_uzmanlik and doktor_uzmanlik != 'nan' else 'Bilinmiyor'
        }
        
        doc = Document(page_content=icerik, metadata=metadata)
        documents.append(doc)
    
    print(f"       {len(documents)} adet belge hazirlandi.")
    
    # Belgeleri ChromaDB'ye kaydetme
    print("[3/4] Vektor veritabani olusturuluyor (Bu islem birkac dakika surebilir)...")
    
    # Batch halinde ekleme (bellek tasarrufu icin)
    BATCH_SIZE = 100
    vectorstore = None
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        if vectorstore is None:
            vectorstore = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=CHROMA_DB_DIZINI
            )
        else:
            vectorstore.add_documents(batch)
        ilerleme = min(i + BATCH_SIZE, len(documents))
        print(f"       Islenen: {ilerleme}/{len(documents)} belge", end="\r")
    
    print(f"\n       Vektor veritabani olusturuldu! ({vectorstore._collection.count()} belge)")

# ============================================================
# Adim 3: Retriever Ayarlama
# ============================================================
# En alakali 5 belgeyi getirmesi icin ayarliyoruz (daha fazla context)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ============================================================
# Adim 4: LLM Baglantisi ve Prompt Tasarimi
# ============================================================
print("[3/4] Yapay Zeka modeli baglaniyor...")
llm = get_llm()

# RAG icin sistem promptu - daha esnek ve detayli
system_prompt = (
    "Sen Turkce konusan, deneyimli bir tibbi bilgi asistanisin. "
    "Asagida sana verilen 'Baglam' bolumunde doktorlarin hastalara verdikleri gercek yanitlar bulunmaktadir. "
    "Bu baglamdaki bilgileri kullanarak kullanicinin sorusunu yanitla. "
    "Baglamdaki bilgilerden faydalanarak kapsamli ve anlasilir bir cevap olustur. "
    "Eger baglamda soruyla hic ilgili bilgi yoksa, bunu belirt ve genel bir yonlendirme yap. "
    "Cevabini Turkce olarak ver. "
    "Her yanitinin sonuna su notu ekle: "
    "'Bu bilgiler genel bilgilendirme amaclidir, kesin tani ve tedavi icin mutlaka bir saglik kurulusuna basvurunuz.'\n\n"
    "Baglam:\n{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Getirilen belgelerle LLM'i birlestiren zincir (chain)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# ============================================================
# Adim 5: Sistemi Calistirma ve Test
# ============================================================
print("[4/4] Sistem hazir!")
print()
print("=" * 60)
print("   SISTEM HAZIR! Soru sorabilirsiniz.")
print("   Ornek: 'Mide agrisi neden olur?'")
print("   Cikmak icin 'q' veya 'cikis' yazin.")
print("=" * 60)
print()

while True:
    try:
        kullanici_sorusu = input("Saglik Sorunuz: ")
    except (EOFError, KeyboardInterrupt):
        print("\nProgram sonlandiriliyor...")
        break
    
    kullanici_sorusu = kullanici_sorusu.strip()
    
    if not kullanici_sorusu:
        continue
    
    if kullanici_sorusu.lower() in ('q', 'cikis', 'quit', 'exit'):
        print("\nGorusuruz! Sagliginiza dikkat edin.")
        break
    
    # Retry mekanizmasi (rate limit hatalari icin)
    MAX_DENEME = 3
    for deneme in range(1, MAX_DENEME + 1):
        try:
            # Sistemi calistir
            print("\n[Dusunuyor...]")
            response = rag_chain.invoke({"input": kullanici_sorusu})
            
            print("\n" + "-" * 50)
            print("AI YANITI:")
            print("-" * 50)
            print(response["answer"])
            print("-" * 50)
            
            # Kaynak belgeleri goster
            if response.get("context"):
                print(f"\n[{len(response['context'])} kaynak belgeden yararlanildi]")
                for i, doc in enumerate(response["context"], 1):
                    uzmanlik = doc.metadata.get("uzmanlik", "Bilinmiyor")
                    if uzmanlik != "Bilinmiyor":
                        uzmanlik = uzmanlik.replace("-", " ").title()
                    print(f"   {i}. {uzmanlik}")
            
            print()
            break  # Basarili, donguden cik
            
        except Exception as e:
            hata_mesaji = str(e)
            if "429" in hata_mesaji or "RESOURCE_EXHAUSTED" in hata_mesaji:
                if deneme < MAX_DENEME:
                    bekleme = 20 * deneme
                    print(f"\n[UYARI] Rate limit - {bekleme} saniye bekleniyor... (Deneme {deneme}/{MAX_DENEME})")
                    time.sleep(bekleme)
                else:
                    print(f"\n[HATA] API kota siniri asildi. Birkac dakika bekleyip tekrar deneyin.")
                    print("Ipucu: Google AI Studio'dan kotanizi kontrol edin: https://ai.dev/rate-limit\n")
            else:
                print(f"\n[HATA]: {hata_mesaji}")
                print("Lutfen tekrar deneyin veya API anahtarinizi kontrol edin.\n")
                break
