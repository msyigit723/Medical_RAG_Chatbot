# -*- coding: utf-8 -*-
"""
Tibbi RAG Chatbot - FastAPI Web Sunucusu
========================================
RAG (Retrieval-Augmented Generation) mimarisi ile
ChromaDB + Gemini API kullanarak tibbi sorulara yanit verir.
Frontend'i de serve eder - tek dosyada tam cozum.
"""

import os
import sys
import io
import time
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
load_dotenv()
PROJE_DIZINI = Path(__file__).resolve().parent
VERI_DOSYASI = PROJE_DIZINI / "train-00000-of-00001.parquet"
CHROMA_DB_DIZINI = PROJE_DIZINI / "chroma_db"
FRONTEND_DIR = PROJE_DIZINI / "frontend"

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
rag_chain = None
vectorstore_global = None
llm_global = None

# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------
def get_llm():
    """OpenAI veya Gemini API anahtarina gore LLM secer."""
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")

    if openai_key and not openai_key.startswith("sk-senin"):
        from langchain_openai import ChatOpenAI
        logger.info("OpenAI GPT modeli kullaniliyor...")
        return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    elif google_key and not google_key.startswith("senin"):
        from langchain_google_genai import ChatGoogleGenerativeAI
        logger.info("Google Gemini modeli kullaniliyor...")
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_retries=2)
    else:
        raise RuntimeError(
            "API anahtari bulunamadi! .env dosyasina "
            "GOOGLE_API_KEY veya OPENAI_API_KEY ekleyin."
        )

# ---------------------------------------------------------------------------
# ChromaDB + RAG Setup
# ---------------------------------------------------------------------------
def setup_vectorstore(embeddings):
    """ChromaDB vektor veritabanini yukle veya olustur."""
    chroma_var_mi = CHROMA_DB_DIZINI.exists() and any(CHROMA_DB_DIZINI.iterdir())

    if chroma_var_mi:
        logger.info("Mevcut vektor veritabani yukleniyor...")
        vs = Chroma(
            persist_directory=str(CHROMA_DB_DIZINI),
            embedding_function=embeddings
        )
        count = vs._collection.count()
        logger.info("Veritabaninda %d belge mevcut.", count)
        if count > 0:
            return vs
        logger.warning("Veritabani bos, yeniden olusturulacak...")

    # Veri setini yukle ve isle
    logger.info("Veri seti yukleniyor...")
    if not VERI_DOSYASI.exists():
        raise FileNotFoundError(f"Veri dosyasi bulunamadi: {VERI_DOSYASI}")

    df = pd.read_parquet(VERI_DOSYASI)
    logger.info("Toplam %d kayit yuklendi.", len(df))

    KAYIT_SAYISI = 1000
    df_subset = df.head(KAYIT_SAYISI)

    # Eski DB'yi temizle
    if CHROMA_DB_DIZINI.exists():
        import shutil
        shutil.rmtree(CHROMA_DB_DIZINI)

    # Belge olustur
    documents = []
    for index, row in df_subset.iterrows():
        doktor_bilgisi = ""
        doktor_unvan = str(row.get('doctor_title', ''))
        doktor_uzmanlik = str(row.get('doctor_speciality', ''))

        if doktor_unvan and doktor_unvan not in ('nan', 'None', ''):
            doktor_bilgisi += f"Doktor: {doktor_unvan}"
        if doktor_uzmanlik and doktor_uzmanlik not in ('nan', 'None', ''):
            uzmanlik_temiz = doktor_uzmanlik.replace('-', ' ').title()
            doktor_bilgisi += f" - Uzmanlik: {uzmanlik_temiz}"

        soru = str(row.get('question_content', ''))
        cevap = str(row.get('question_answer', ''))

        if not soru or not cevap or soru == 'nan' or cevap == 'nan':
            continue

        icerik = ""
        if doktor_bilgisi:
            icerik += f"{doktor_bilgisi}\n"
        icerik += f"Hasta Sorusu: {soru}\nDoktor Cevabi: {cevap}"

        metadata = {
            "id": index,
            "uzmanlik": doktor_uzmanlik if doktor_uzmanlik not in ('nan', 'None', '') else 'Bilinmiyor'
        }

        documents.append(Document(page_content=icerik, metadata=metadata))

    logger.info("%d adet belge hazirlandi.", len(documents))

    # Batch halinde ChromaDB'ye ekle
    BATCH_SIZE = 100
    vs = None
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i:i + BATCH_SIZE]
        if vs is None:
            vs = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=str(CHROMA_DB_DIZINI)
            )
        else:
            vs.add_documents(batch)
        logger.info("Islenen: %d/%d belge", min(i + BATCH_SIZE, len(documents)), len(documents))

    logger.info("Vektor veritabani olusturuldu! (%d belge)", vs._collection.count())
    return vs


def build_rag_chain(vectorstore, llm):
    """RAG chain olustur."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

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

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever, question_answer_chain)


# ---------------------------------------------------------------------------
# Fallback Responses (API kota asiminda kullanilir)
# ---------------------------------------------------------------------------
FALLBACK_RESPONSES = {
    "ates": "Ates Durumunda Yapilmasi Gerekenler:\n\n- 38 derece uzeri ates varsa ates dusurucu (parasetamol) kullanabilirsiniz\n- Bol sivi tuketmeye ozen gosterin\n- Hafif giysiler giyin ve odayi serin tutun\n- 39 derece uzerinde veya 3 gunden uzun surerse mutlaka doktora basvurun",
    "ilac": "Ilac Kullanim Rehberi:\n\n- Ilaclarinizi her gun ayni saatte almaya ozen gosterin\n- Antibiyotikleri asla yarida birakmayin\n- Ac/tok karnina kullanim talimatlarina uyun\n- Yan etkiler fark ederseniz doktorunuza haber verin",
    "tansiyon": "Tansiyon Takibi:\n\n- Normal tansiyon: 120/80 mmHg\n- Yuksek tansiyon: 140/90 mmHg uzeri\n- Tuz tuketimini azaltin, duzenli egzersiz yapin\n- 180/120 uzeri degerlerde acil servise basvurun!",
    "seker": "Kan Sekeri Yonetimi:\n\n- Aclik kan sekeri: 70-100 mg/dL normal\n- Basit seker ve karbonhidratlardan kacinin\n- Duzenli olcum yapin\n- 300 mg/dL uzeri veya 70 mg/dL alti acil durumdur!",
    "bas agri": "Bas Agrisi Hakkinda:\n\n- Bol su icin\n- Karanlik ve sessiz bir odada dinlenin\n- Siddetli, ani baslayan bas agrilarinda acile basvurun",
    "grip": "Grip Hakkinda:\n\n- Bol sivi tuketin\n- Dinlenmeye ozen gosterin\n- Belirtiler 7 gunden uzun surerse doktora basvurun",
    "mide": "Mide Rahatsizliklari:\n\n- Asitli, baharatli ve yagli yiyeceklerden kacinin\n- Kucuk porsiyonlar halinde yiyin\n- Yemekten hemen sonra yatmayin\n- Belirtiler surekli ise gastroenteroloji uzmani gorun",
}


def fallback_answer(question):
    """Anahtar kelime tabanli yedek cevap sistemi."""
    q = question.lower()
    for key, answer in FALLBACK_RESPONSES.items():
        if key in q:
            return answer, 0.5
    return (
        "Sorunuzu aldim ancak bu konuda yeterli bilgiye sahip degilim. "
        "Lutfen sorunuzu farkli bir sekilde sormayi deneyin veya "
        "tibbi acil durumlarda 112'yi arayin.",
        0.0,
    )


# ---------------------------------------------------------------------------
# Lifespan - Uygulama baslarken RAG pipeline'i kur
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_chain, vectorstore_global, llm_global

    try:
        logger.info("=" * 60)
        logger.info("   TIBBI RAG CHATBOT - Web Sunucusu Baslatiliyor")
        logger.info("=" * 60)

        # 1. Embedding modeli
        logger.info("[1/4] Embedding modeli yukleniyor...")
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # 2. ChromaDB
        logger.info("[2/4] Vektor veritabani hazirlaniyor...")
        vectorstore_global = setup_vectorstore(embeddings)

        # 3. LLM
        logger.info("[3/4] Yapay Zeka modeli baglaniyor...")
        llm_global = get_llm()

        # 4. RAG Chain
        logger.info("[4/4] RAG pipeline olusturuluyor...")
        rag_chain = build_rag_chain(vectorstore_global, llm_global)

        logger.info("=" * 60)
        logger.info("   SISTEM HAZIR! http://localhost:8000 adresinden erisebilirsiniz.")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("RAG pipeline kurulum hatasi: %s", e)
        logger.info("Backend fallback modda calisacak.")

    yield


# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Tibbi RAG Chatbot API",
    description="RAG mimarisi ile saglik soru-cevap chatbot servisi",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    answer: str
    confidence: float
    source: str  # "rag", "fallback"
    sources: list = []


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Kullanici mesajina RAG ile cevap uret."""
    question = req.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Mesaj bos olamaz.")

    # RAG chain varsa kullan
    if rag_chain is not None:
        MAX_DENEME = 2
        for deneme in range(1, MAX_DENEME + 1):
            try:
                response = rag_chain.invoke({"input": question})
                answer = response.get("answer", "")
                
                # Kaynak bilgilerini topla
                sources = []
                if response.get("context"):
                    for doc in response["context"]:
                        uzmanlik = doc.metadata.get("uzmanlik", "Bilinmiyor")
                        if uzmanlik != "Bilinmiyor":
                            uzmanlik = uzmanlik.replace("-", " ").title()
                        sources.append(uzmanlik)

                return ChatResponse(
                    answer=answer,
                    confidence=0.85,
                    source="rag",
                    sources=sources,
                )
            except Exception as e:
                hata = str(e)
                if ("429" in hata or "RESOURCE_EXHAUSTED" in hata) and deneme < MAX_DENEME:
                    logger.warning("Rate limit, bekleniyor... (Deneme %d)", deneme)
                    time.sleep(5)
                else:
                    logger.error("RAG hatasi: %s", e)
                    break

    # Fallback
    answer, conf = fallback_answer(question)
    return ChatResponse(answer=answer, confidence=conf, source="fallback", sources=[])


@app.get("/api/health")
async def health():
    """Saglik kontrolu."""
    rag_ready = rag_chain is not None
    belge_sayisi = 0
    if vectorstore_global:
        try:
            belge_sayisi = vectorstore_global._collection.count()
        except:
            pass
    return {
        "status": "ok",
        "rag_ready": rag_ready,
        "total_documents": belge_sayisi,
        "mode": "rag" if rag_ready else "fallback",
    }


# ---------------------------------------------------------------------------
# Frontend Serve
# ---------------------------------------------------------------------------
@app.get("/")
async def serve_index():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse(
        {"error": "Frontend bulunamadi. frontend/index.html dosyasini kontrol edin."},
        status_code=404
    )


@app.get("/{filename:path}")
async def serve_static(filename: str):
    file_path = FRONTEND_DIR / filename
    if file_path.exists() and file_path.is_file():
        return FileResponse(file_path)
    # SPA fallback
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return JSONResponse({"error": "Dosya bulunamadi."}, status_code=404)


# ---------------------------------------------------------------------------
# Dogrudan calistirma
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
