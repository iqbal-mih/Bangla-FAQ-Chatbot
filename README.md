# Bangla FAQ Chatbot

A **production-ready AI-powered Bangla FAQ Chatbot** designed to answer user questions accurately using **Retrieval-Augmented Generation (RAG)**. This project combines **FastAPI**, **vector search**, and **LLM-based reasoning** to deliver reliable responses from Bangla documents.

---
**Colab Link:** https://colab.research.google.com/drive/1e29SurjaYdu3uJDvljKOiHSuA0qUq5ut?usp=sharing
---
## 🚀 Key Features

* 🧠 **RAG-based Question Answering** (semantic search + LLM)
* 🇧🇩 **Bangla Language Support**
* ⚡ **FastAPI Backend** for high-performance APIs
* 📄 Supports **PDF / CSV / Text-based FAQs**
* 🔍 Vector search using **FAISS / embeddings**
* 🔐 Secure API key handling using **environment variables**
* 🧩 Modular & scalable project structure

---

## 🏗️ System Architecture

```
User Query
   ↓
FastAPI Backend
   ↓
Embedding Generator
   ↓
Vector Database (FAISS)
   ↓
Relevant Context Retrieval
   ↓
LLM (Answer Generation)
   ↓
Final Bangla Response
```

---

## 🧰 Tech Stack

| Layer          | Technology            |
| -------------- | --------------------- |
| Backend API    | FastAPI               |
| Language Model | OpenAI / LLM API      |
| Embeddings     | Sentence Transformers |
| Vector DB      | FAISS                 |
| Language       | Python                |
| Environment    | Conda / venv          |

---

## 📂 Project Structure

```
Bangla-FAQ-Chatbot/
│
├── Backend/
│   ├── main.py                  # FastAPI entry point
│   ├── backend_voice_chatbot.py # Core chatbot logic
│   ├── requirements.txt
│
├── data/
│   ├── faqs.pdf
│   ├── faqs.csv
│
├── .gitignore
├── README.md
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/iqbal-mih/Bangla-FAQ-Chatbot.git
cd Bangla-FAQ-Chatbot
```

### 2️⃣ Create Virtual Environment

```bash
conda create -n bangla-faq python=3.10 -y
conda activate bangla-faq
```

### 3️⃣ Install Dependencies

```bash
pip install -r Backend/requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

⚠️ **Never commit `.env` files to GitHub**

---

## ▶️ Running the Application

```bash
cd Backend
uvicorn main:app --reload
```

API will be available at:

```
http://127.0.0.1:8000
```

Swagger Docs:

```
http://127.0.0.1:8000/docs
```

---

## 🧪 Example API Usage

**POST** `/chat`

```json
{
  "question": "ভর্তি সংক্রান্ত তথ্য কী?"
}
```

**Response:**

```json
{
  "answer": "ভর্তি সংক্রান্ত সকল তথ্য আমাদের অফিসিয়াল ওয়েবসাইটে পাওয়া যাবে।"
}
```

---

## 🛡️ Security Best Practices

* ✔ API keys stored using environment variables
* ✔ `.gitignore` configured properly
* ✔ No secrets in commit history

---

## 📌 Future Improvements

* 🔊 Voice-based input & output (frontend/backend separation)
* 🌐 Web frontend (React / Next.js)
* 📈 Conversation history & analytics
* 🧪 Automated testing
* ☁️ Cloud deployment (Docker + AWS/GCP)

---

## 👤 Author

**Iqbal**
AI Engineering Enthusiast | Machine Learning | RAG Systems

* GitHub: [https://github.com/iqbal-mih](https://github.com/iqbal-mih)

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ If you find this project helpful, please give it a **star** and feel free to contribute!
