# 🩺 Tıbbi RAG Chatbot (Medical RAG Chatbot)

RAG (Retrieval-Augmented Generation) mimarisi kullanılarak geliştirilmiş, tıbbi sorulara doktorların gerçek yanıtlarından faydalanarak cevap veren akıllı bir sağlık asistanı web uygulamasıdır.

## 🚀 Özellikler
- **RAG Mimarisi:** FastAPI, Langchain ve ChromaDB entegrasyonu ile tasarlandı.
- **Çoklu LLM Desteği:** Kullanıcının yapılandırmasına göre OpenAI (GPT-3.5/GPT-4) veya Google Gemini (Gemini 2.5 Flash) API'lerini destekler.
- **Vektör Veritabanı:** İçerisinde barındırdığı medikal veri seti (Parquet formatında), `HuggingFace` yerleştirmeleri (embeddings) kullanılarak vektörize edilir ve ChromaDB üzerinde saklanır.
- **Acil Durum / Fallback Sistemi:** API kota aşımları durumunda veya offline senaryolarda yaygın sağlık sorunlarına önceden tanımlanmış güvenli ve yönlendirici yanıtlar verir.
- **Tek Sunucu:** FastAPI sayesinde hem backend hem de frontend dosyaları tek bir sunucu üzerinden asenkron olarak sunulur.

## 🛠️ Kullanılan Teknolojiler
- **Backend:** Python, FastAPI, Uvicorn
- **AI & RAG Mimarisi:** Langchain, Chroma, HuggingFace (`sentence-transformers`), Pandas
- **LLM Entegrasyonu:** `langchain-google-genai`, `langchain-openai`
- **Frontend:** HTML5, CSS3, Vanilla JavaScript

## 📦 Kurulum ve Çalıştırma

### 1. Depoyu Klonlayın
```bash
git clone https://github.com/KULLANICI_ADINIZ/RAG_Chatbot.git
cd RAG_Chatbot
```

### 2. Sanal Ortam Oluşturun ve Aktifleştirin
```bash
python -m venv venv
# Windows için:
venv\Scripts\activate
# macOS/Linux için:
source venv/bin/activate
```

### 3. Bağımlılıkları Yükleyin
*(Projeye ait bağımlılıkların yüklü olduğundan emin olun)*
```bash
pip install fastapi uvicorn langchain chromadb pandas python-dotenv langchain-openai langchain-google-genai sentence-transformers
```

### 4. API Anahtarlarını Ayarlayın
Proje dizininde bir `.env` dosyası oluşturun ve aşağıdaki değişkenlerden birini (veya ikisini) kendi API anahtarınız ile güncelleyin:
```env
GOOGLE_API_KEY=sizin_gemini_api_anahtariniz
OPENAI_API_KEY=sizin_openai_api_anahtariniz
```

### 5. Uygulamayı Başlatın
```bash
python server.py
```
Vektör veritabanı (ChromaDB) ilk açılışta `train-00000-of-00001.parquet` veri setini okuyarak oluşturulacaktır. İşlem tamamlandıktan sonra tarayıcınızdan http://localhost:8000 adresine giderek uygulamayı kullanabilirsiniz.

## ⚠️ Önemli Uyarı
Bu proje geliştirici portfolyosu kapsamında eğitim ve örnek amaçlı yapılmıştır. Bot tarafından sağlanan bilgiler **kesin tanı veya tedavi amacı taşımaz**. Tüm sağlık problemlerinizde öncelikle profesyonel bir hekime ve sağlık kuruluşuna başvurunuz.
