# Bangla FAQ Chatbot Backend API Documentation

Base URL: `http://localhost:8000`

## Endpoints

---

### 1. Ask Text Question
**POST** `/ask_text`

- **Description:** Get an answer to a Bangla text question.
- **Request Body (JSON):**
  - `question` (string): The user's question in Bangla.
- **Response (JSON):**
  - `category` (string): Detected category (education, health, travel, technology, sports)
  - `answer` (string): Answer in Bangla

**Example Request:**
```bash
curl -X POST "http://localhost:8000/ask_text" -H "Content-Type: application/json" -d '{"question": "শিক্ষা বলতে কী বুঝায়?"}'
```

---

### 2. Ask Voice Question
**POST** `/ask_voice`

- **Description:** Upload a Bangla voice (.wav) file and get the recognized question and answer.
- **Request (multipart/form-data):**
  - `file`: Audio file (WAV format, Bangla speech)
- **Response (JSON):**
  - `question` (string): Recognized Bangla question
  - `category` (string): Detected category
  - `answer` (string): Answer in Bangla

**Example Request:**
```bash
curl -X POST "http://localhost:8000/ask_voice" -F "file=@your_question.wav"
```

---

### 3. Text-to-Speech (TTS)
**POST** `/tts`

- **Description:** Convert Bangla text to speech (returns audio as hex string).
- **Request Body (JSON):**
  - `text` (string): Bangla text to convert
- **Response (JSON):**
  - `audio` (string): Audio data as hex string (MP3 format)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/tts" -H "Content-Type: application/json" -d '{"text": "আপনার প্রশ্নের উত্তর এখানে।"}'
```

- **Frontend Note:** To play the audio, convert the hex string to binary and play as MP3.

---

## CORS
- All origins are allowed (CORS enabled for all domains).

## Authentication
- No authentication required.

## Contact
- For any issues, contact the backend developer.
