# backend_voice_chatbot.py
"""
Backend for Bangla FAQ Chatbot with Voice (STT/TTS) Support
- REST API using FastAPI
- Speech-to-Text (STT) using openai-whisper or SpeechRecognition
- Text-to-Speech (TTS) using gTTS or pyttsx3
"""
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import tempfile

# --- Vector DB & LLM Imports ---
import faiss
import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from openai import OpenAI

# --- TTS & STT Imports ---
import speech_recognition as sr
from gtts import gTTS

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Embedding Model ---
embedding_model = HuggingFaceEmbeddings(model_name="l3cube-pune/bengali-sentence-similarity-sbert")

# --- Data Preparation  ---
education_chunks = [
    ("শিক্ষা মানুষের জ্ঞান ও দক্ষতা বৃদ্ধি করে।", {"category": "education"}),
    ("প্রাথমিক শিক্ষা একটি শিশুর ভিত্তি গড়ে তোলে।", {"category": "education"}),
    ("উচ্চশিক্ষা ক্যারিয়ার গঠনে গুরুত্বপূর্ণ ভূমিকা রাখে।", {"category": "education"}),
    ("অনলাইন শিক্ষা বর্তমানে জনপ্রিয় হয়ে উঠছে।", {"category": "education"}),
    ("নিয়মিত পড়াশোনা ভালো ফলাফল নিশ্চিত করে।", {"category": "education"}),
    ("শিক্ষকের ভূমিকা শিক্ষার্থীর জীবনে অত্যন্ত গুরুত্বপূর্ণ।", {"category": "education"}),
    ("কারিগরি শিক্ষা কর্মসংস্থানে সহায়ক।", {"category": "education"}),
    ("শিক্ষায় প্রযুক্তির ব্যবহার শেখাকে সহজ করে।", {"category": "education"}),
    ("পরীক্ষার প্রস্তুতির জন্য সময় ব্যবস্থাপনা জরুরি।", {"category": "education"}),
    ("আজীবন শিক্ষা মানুষকে প্রতিনিয়ত উন্নত করে।", {"category": "education"})
]

health_chunks = [
    ("সুস্থ থাকতে নিয়মিত ব্যায়াম করা জরুরি।", {"category": "health"}),
    ("পর্যাপ্ত ঘুম শরীরের জন্য অত্যন্ত প্রয়োজনীয়।", {"category": "health"}),
    ("সুষম খাদ্য গ্রহণ স্বাস্থ্য ভালো রাখে।", {"category": "health"}),
    ("পরিষ্কার পানি পান রোগ প্রতিরোধে সাহায্য করে।", {"category": "health"}),
    ("ধূমপান স্বাস্থ্যের জন্য ক্ষতিকর।", {"category": "health"}),
    ("মানসিক স্বাস্থ্য শারীরিক স্বাস্থ্যের মতোই গুরুত্বপূর্ণ।", {"category": "health"}),
    ("নিয়মিত স্বাস্থ্য পরীক্ষা রোগ নির্ণয়ে সহায়ক।", {"category": "health"}),
    ("অতিরিক্ত চিনি ডায়াবেটিসের ঝুঁকি বাড়ায়।", {"category": "health"}),
    ("হাত ধোয়ার অভ্যাস সংক্রমণ কমায়।", {"category": "health"}),
    ("স্বাস্থ্য সচেতনতা জীবনমান উন্নত করে।", {"category": "health"})
]

travel_chunks = [
    ("ভ্রমণ মানুষের মানসিক চাপ কমায়।", {"category": "travel"}),
    ("ভ্রমণের আগে পরিকল্পনা করা জরুরি।", {"category": "travel"}),
    ("বাংলাদেশে অনেক সুন্দর পর্যটন স্থান রয়েছে।", {"category": "travel"}),
    ("ভ্রমণের সময় নিরাপত্তা নিশ্চিত করা উচিত।", {"category": "travel"}),
    ("অফ-সিজনে ভ্রমণ খরচ কম হয়।", {"category": "travel"}),
    ("ভ্রমণে স্থানীয় সংস্কৃতি জানা গুরুত্বপূর্ণ।", {"category": "travel"}),
    ("ভ্রমণের সময় প্রয়োজনীয় কাগজপত্র সাথে রাখা দরকার।", {"category": "travel"}),
    ("পর্যটন শিল্প অর্থনীতিতে গুরুত্বপূর্ণ ভূমিকা রাখে।", {"category": "travel"}),
    ("ভ্রমণ অভিজ্ঞতা মানুষকে নতুনভাবে ভাবতে শেখায়।", {"category": "travel"}),
    ("ভ্রমণের সময় পরিবেশ রক্ষা করা উচিত।", {"category": "travel"})
]

technology_chunks = [
    ("প্রযুক্তি মানুষের জীবনকে সহজ করেছে।", {"category": "technology"}),
    ("কৃত্রিম বুদ্ধিমত্তা আধুনিক প্রযুক্তির একটি অংশ।", {"category": "technology"}),
    ("ইন্টারনেট তথ্য আদান-প্রদানে গুরুত্বপূর্ণ ভূমিকা রাখে।", {"category": "technology"}),
    ("মোবাইল ফোন যোগাযোগ ব্যবস্থাকে বদলে দিয়েছে।", {"category": "technology"}),
    ("মেশিন লার্নিং ডেটা থেকে শেখে।", {"category": "technology"}),
    ("প্রযুক্তি শিক্ষা ব্যবস্থাকে আরও উন্নত করছে।", {"category": "technology"}),
    ("সাইবার নিরাপত্তা এখন একটি বড় চ্যালেঞ্জ।", {"category": "technology"}),
    ("ক্লাউড কম্পিউটিং ডেটা সংরক্ষণ সহজ করেছে।", {"category": "technology"}),
    ("প্রযুক্তির অপব্যবহার ক্ষতিকর হতে পারে।", {"category": "technology"}),
    ("নতুন প্রযুক্তি কর্মসংস্থানের সুযোগ সৃষ্টি করে।", {"category": "technology"})
]

sports_chunks = [
    ("খেলাধুলা শরীর সুস্থ রাখে।", {"category": "sports"}),
    ("নিয়মিত খেলাধুলা মানসিক চাপ কমায়।", {"category": "sports"}),
    ("ফুটবল বিশ্বব্যাপী জনপ্রিয় একটি খেলা।", {"category": "sports"}),
    ("ক্রিকেট বাংলাদেশের সবচেয়ে জনপ্রিয় খেলা।", {"category": "sports"}),
    ("খেলাধুলা দলগত কাজ শেখায়।", {"category": "sports"}),
    ("শিশুদের বিকাশে খেলাধুলা গুরুত্বপূর্ণ।", {"category": "sports"}),
    ("অলিম্পিক বিশ্বমানের ক্রীড়া আসর।", {"category": "sports"}),
    ("খেলাধুলায় নিয়মানুবর্তিতা প্রয়োজন।", {"category": "sports"}),
    ("শারীরিক সক্ষমতা বাড়াতে খেলাধুলা দরকার।", {"category": "sports"}),
    ("খেলাধুলা বিনোদনের একটি ভালো মাধ্যম।", {"category": "sports"})
]
all_chunks = education_chunks + health_chunks + travel_chunks + technology_chunks + sports_chunks

documents = [Document(page_content=text, metadata=meta) for text, meta in all_chunks]

# --- LLM Setup ---
import sys
if 'GITHUB_TOKEN' not in os.environ:
    print("Error: GITHUB_TOKEN environment variable not set.", file=sys.stderr)
    sys.exit(1)
token = os.environ['GITHUB_TOKEN']
endpoint = "https://models.github.ai/inference"
model = "openai/gpt-4.1-nano"
client = OpenAI(base_url=endpoint, api_key=token)

# --- Category Router ---
def detect_category_llm(question):
    system_msg = "তুমি একটি শ্রেণিবিন্যাসকারী এজেন্ট। নিচের প্রশ্নটি পড়ে বলো এটি কোন ক্যাটাগরির মধ্যে পড়ে: education,health,travel,technology,sports। শুধুমাত্র ক্যাটাগরির নাম এক শব্দে ইংরেজিতে উত্তর দাও। in case the question doesnt belong to any class predict sports"
    response = client.chat.completions.create(
        messages=[{"role": "system", "content": system_msg}, {"role": "user", "content": question}],
        model=model, temperature=0, top_p=1.0)
    category = response.choices[0].message.content.strip().lower()
    return category

# --- Metadata Filter ---
def filter_by_metadata(query, category):
    filtered_docs = [doc for doc in documents if doc.metadata['category'] == category]
    if not filtered_docs:
        return []
    temp_vector_store = FAISS.from_documents(filtered_docs, embedding_model)
    similar_docs = temp_vector_store.similarity_search(query, k=3)
    return similar_docs

# --- RAG Chain ---
def ask_faq_bot(user_question: str, category: str):
    docs = filter_by_metadata(user_question, category)
    context = "\n".join([doc.page_content for doc in docs])
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "তুমি একজন সহায়ক বাংলা সহকারী। শুধুমাত্র নিচের প্রাসঙ্গিক তথ্য থেকে উত্তর দাও। যদি প্রশ্নের উত্তর এতে না থাকে, বলো 'দুঃখিত, এই বিষয়ে আমার জানা নেই। Strictly dont answer the question if the context is empty say i dont know , context : " + context},
            {"role": "user", "content": user_question},
        ],
        temperature=0.7, top_p=0.7, model=model)
    return response.choices[0].message.content

# --- API Endpoints ---
@app.post("/ask_text")
async def ask_text(question: str):
    category = detect_category_llm(question)
    answer = ask_faq_bot(question, category)
    return {"category": category, "answer": answer}

@app.post("/ask_voice")
async def ask_voice(file: UploadFile = File(...)):
    # Save uploaded file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    # Speech-to-Text
    recognizer = sr.Recognizer()
    with sr.AudioFile(tmp_path) as source:
        audio = recognizer.record(source)
    try:
        question = recognizer.recognize_google(audio, language="bn-BD")
    except Exception as e:
        os.remove(tmp_path)
        return JSONResponse(status_code=400, content={"error": str(e)})
    os.remove(tmp_path)
    # Get answer
    category = detect_category_llm(question)
    answer = ask_faq_bot(question, category)
    return {"question": question, "category": category, "answer": answer}

@app.post("/tts")
async def tts(text: str):
    tts = gTTS(text=text, lang="bn")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tts.save(tmp.name)
        tmp.seek(0)
        audio_bytes = tmp.read()
    os.remove(tmp.name)
    return JSONResponse(content={"audio": audio_bytes.hex()})

# --- Main ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
