SanketSetu

Real-time Indian Sign Language communication and AI-guided sign practice using a standard webcam.

SanketSetu is an accessibility-focused web application designed to reduce communication barriers between Indian Sign Language (ISL) users and non-signers.

It combines webcam-based sign recognition, confidence-aware translation, text-to-speech, speech/text conversation support, and AI-guided sign practice in one accessible web experience.

The current MVP focuses on 20 commonly used ISL signs, with a clearly defined vocabulary boundary and safe handling of uncertain predictions.

✨ Core Features

🎥 Real-Time ISL Recognition

Webcam-based sign recognition.

Hand-landmark-based feature extraction.

Designed to work with a standard laptop or USB webcam.

No specialized gloves or depth cameras required.

Supports a defined MVP vocabulary of 20 signs.

🧠 Confidence-Aware Recognition

SanketSetu avoids forcing uncertain predictions into the conversation.

Camera
   ↓
Hand Landmark Extraction
   ↓
ISL Classifier
   ↓
Confidence Gate
   ↓
Reliable Prediction → Output
Low Confidence      → "Sign unclear. Please try again."

💬 Translate

The Translate page is the primary ISL communication interface.

Flow:

Webcam
   ↓
Hand Tracking
   ↓
Sign Classification
   ↓
Confidence Check
   ↓
Phrase Builder
   ↓
Text + Speech

Features:

Start and end camera sessions.

Real-time sign recognition.

Confidence-aware predictions.

Current-session phrase building.

Text output.

Text-to-speech playback.

Clear/restart translation session.

🗣️ Conversation

The Conversation page provides a simple communication bridge using speech and text.

Features:

Text-based messaging.

Speech input where browser support is available.

Converts recognized speech into editable text before sending.

Text-to-speech for messages.

Optional Auto-Speak.

Current-session conversation.

Clear conversation option.

Graceful microphone and speech error handling.

Conversation flow:

Speech
  ↓
Speech Recognition
  ↓
Editable Text
  ↓
Send
  ↓
Conversation

Text
  ↓
Text-to-Speech
  ↓
Spoken Output

The Conversation page is a communication interface, not an AI chatbot. It does not generate fake or automated replies.

🎓 Learn & Practice

SanketSetu includes an AI-guided sign practice module.

The current library contains 20 supported signs covering areas such as:

Basic communication

Daily needs

Emergencies

Feelings

Practice flow:

Select Sign
    ↓
View Reference
    ↓
Practice Using Webcam
    ↓
Sign Recognition
    ↓
Compare With Target
    ↓
Instant Feedback

Feedback can include:

✅ Correct

🔄 Try Again

⚠️ Sign Unclear — Please Repeat

The practice experience is intentionally lightweight. It does not require persistent learning progress, streaks, practice history, or a learning database.

🔎 Learn Sign Library

20-sign learning library.

Search functionality.

Category filtering.

Sign reference cards.

Sign detail/practice interface.

Webcam-based practice.

Immediate recognition feedback.

🆘 Help Center

The Help page explains:

What SanketSetu does.

How the recognition workflow works.

How to use Translate.

Supported vocabulary.

Confidence-aware recognition.

Learn & Practice.

Conversation mode.

Privacy.

Troubleshooting.

Current limitations.

Future roadmap.

🏗️ System Architecture

                  ┌───────────────┐
                  │    Webcam     │
                  └───────┬───────┘
                          ↓
                  ┌───────────────┐
                  │ Hand Tracking │
                  │  MediaPipe    │
                  └───────┬───────┘
                          ↓
                  ┌───────────────┐
                  │ Landmark      │
                  │ Normalization │
                  └───────┬───────┘
                          ↓
                  ┌───────────────┐
                  │ ISL Classifier│
                  └───────┬───────┘
                          ↓
                  ┌───────────────┐
                  │ Confidence    │
                  │     Gate      │
                  └───────┬───────┘
                          ↓
                 ┌────────┴─────────┐
                 ↓                  ↓
          ┌──────────────┐   ┌───────────────┐
          │  Translate   │   │ Learn &       │
          │              │   │ Practice      │
          └──────┬───────┘   └───────┬───────┘
                 ↓                   ↓
          Phrase Builder       Target Comparison
                 ↓                   ↓
          Text + Speech         Instant Feedback

🛠️ Technology Stack

Frontend

Next.js 16 App Router

React 19

TypeScript

Tailwind CSS 4

Lucide React

Computer Vision & Machine Learning

Python

FastAPI

MediaPipe

Trained ISL classifier

Landmark-based feature processing

Authentication & Data

Better Auth

Neon PostgreSQL

Drizzle ORM

pg

Authentication/database components are included in the current project structure. The core Translate and Learn & Practice flows do not require persistent learning progress.

Voice

Browser Speech Synthesis

Browser Speech Recognition where supported

Optional server-side speech integrations

📁 Project Structure

SanketSetu/
│
├── app/
│   ├── page.tsx
│   │   └── Translator dashboard and phrase builder
│   ├── conversation/
│   │   └── page.tsx
│   │       └── Speech/text conversation
│   ├── learn/
│   │   └── page.tsx
│   │       └── 20-sign Learn & Practice library
│   ├── help/
│   │   └── page.tsx
│   │       └── Help Center
│   ├── sign-in/
│   │   └── page.tsx
│   │       └── Authentication UI
│   └── api/
│       ├── auth/[...all]/route.ts
│       └── detect/route.ts
│
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── models/
│       └── isl_classifier.tflite
│
├── lib/
│   ├── auth.ts
│   ├── auth-client.ts
│   └── db/
│       └── schema.ts
│
├── public/
├── package.json
├── pnpm-lock.yaml
└── README.md

💻 Requirements

Node.js 20 or newer

pnpm 12 or newer

Python 3.10 or newer

Modern browser

Webcam

Microphone

Neon PostgreSQL database if authentication is enabled

🚀 Local Setup

1. Clone the Repository

git clone <YOUR_GITHUB_REPOSITORY_URL>
cd SanketSetu

Or open a terminal directly in the existing project folder.

2. Install Frontend Dependencies

pnpm install

3. Configure Environment Variables

Create .env.local in the project root:

DATABASE_URL=your_neon_pooled_connection_string
BETTER_AUTH_SECRET=your_random_secret_at_least_32_characters

Never commit .env.local or expose secrets in client-side code.

If authentication/database is removed from the final MVP, these variables may not be required by that version.

4. Configure the Database

If authentication is enabled, make sure the required Better Auth tables exist in the configured PostgreSQL database. The project uses the Drizzle schema in lib/db/schema.ts.

▶️ Run the Frontend

From the project root:

pnpm dev

Open:

http://localhost:3000

Main Routes

Route

Purpose

/

Translator dashboard

/conversation

Speech/text conversation

/learn

20-sign Learn & Practice

/help

Help Center

/sign-in

Authentication

/api/auth/[...all]

Authentication API

/api/detect

Detection proxy

Example:

http://localhost:3000/conversation

🐍 Run the FastAPI Backend

For model-backed recognition, open a second terminal in the project root.

Create the virtual environment:

python -m venv .venv

Windows PowerShell

.\.venv\Scripts\Activate.ps1

macOS / Linux

source .venv/bin/activate

Install dependencies:

pip install -r backend/requirements.txt

Start FastAPI:

uvicorn backend.main:app --reload --port 8000

Backend:

http://localhost:8000

Example response format:

{
  "label": "hello",
  "confidence": 0.94
}
🔐 Privacy

SanketSetu follows a privacy-first approach for the core interaction.

The application is designed to avoid unnecessary storage of raw camera recordings.

Learn & Practice does not require:

Persistent learning history

Streaks

Progress percentages

Practice analytics

A learning database

Conversation is designed around the current session rather than long-term conversation storage.
⚠️ Current Limitations

Recognition performance may vary with:

Poor lighting

Hand occlusion

Camera angle

Distance from the camera

Background conditions

Different signing styles

Signer variation

The current MVP has a limited vocabulary of 20 signs and primarily focuses on supported/common gestures. Dynamic and continuous sign recognition requires additional temporal modelling and is outside the current MVP scope.

Speech recognition and synthesis also depend on browser/device support.

🚧 Future Roadmap

Phase 1 — MVP
20 common ISL signs
Real-time recognition
Confidence-aware output
Phrase building
Text-to-speech
Conversation support
AI-guided practice

Phase 2 — Recognition Expansion
Larger ISL vocabulary
More signer diversity
Improved robustness
Better handling of two-hand signs

Phase 3 — Dynamic Sign Recognition
Temporal sequence modelling
LSTM/TCN/Transformer-based approaches
Continuous sign recognition

Phase 4 — Accessibility Expansion
Offline inference
Mobile deployment
Kiosk deployment
Regional-language speech support