# 🌱 MindMate, AI-Powered Mental Health Assistant for Students

MindMate is a privacy-respecting wellbeing companion built for university
students. It combines a supportive conversational assistant with machine
learning that **personalises** the support each student receives, adapting
its tone, recommendations and coping strategies to the individual.

> ⚠️ **Important:** MindMate is a self-help and educational tool. It is **not**
> a medical device, does not provide a diagnosis, and is not a substitute for
> professional care or an emergency service. Crisis helplines are surfaced
> throughout the app.

---

## ✨ Features

| Area | What it does |
|------|--------------|
| 💬 **Conversational assistant** | Empathetic, judgement-free chat. Uses the OpenAI API when a key is configured, and a built-in **offline engine** otherwise, so it always works. |
| 🛟 **Crisis safety layer** | Every message is screened for self-harm / crisis signals before a reply is generated; if detected, the app responds safely and surfaces 24x7 helplines. |
| 📈 **Mood tracker** | Daily mood, energy and sleep check-ins with sentiment analysis of free-text notes and trend charts. |
| 📝 **Self-checks** | Standard **PHQ-9** (mood) and **GAD-7** (anxiety) screening questionnaires with severity bands. |
| 🧠 **ML personalisation** | A **K-Means** model assigns each student a support *segment*; a **Logistic-Regression** model predicts their *stress level*; a transparent recommender tailors coping strategies. |
| 📚 **Resource library** | Helplines (India-focused + global), psychoeducation and a full coping-strategy library. |

---

## 🏗️ Architecture

```
                         ┌──────────────────────────┐
                         │      Streamlit UI         │  app.py + src/ui/*
                         │  Home · Chat · Mood ·      │
                         │  Self-Check · Insights ·   │
                         │  Resources                 │
                         └─────────────┬─────────────┘
                                       │
                         ┌─────────────▼─────────────┐
                         │     Service layer          │  src/service.py
                         │  (model loading, profiles, │
                         │   recommendations)         │
                         └──────┬───────────┬─────────┘
                ┌───────────────┘           └───────────────┐
     ┌──────────▼──────────┐              ┌─────────────────▼─────────────┐
     │  Assistant package   │              │        ML package             │
     │  chat_engine ·       │              │  sentiment · personalization  │
     │  safety · prompts    │              │  stress · recommender         │
     └──────────┬───────────┘              └───────────────┬───────────────┘
                └───────────────┬───────────────────────────┘
                      ┌─────────▼─────────┐
                      │  SQLite database   │  src/database.py
                      │  users · moods ·   │
                      │  chats · screens   │
                      └────────────────────┘
```

---

## 🚀 Quick start

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/gannyprodigy/mindmate-mental-health-assistant.git mindmate
cd mindmate
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate data & train the ML models

```bash
python -m scripts.generate_data      # writes synthetic datasets to data/
python -m scripts.train_models       # trains & saves the segmenter + classifier
python -m scripts.analyze_evaluation # (optional) reproduces the study stats
```

### 3. (Optional) enable live AI responses

The app runs fully **without** an API key using its offline engine. To use
live OpenAI responses instead:

```bash
cp .env.example .env
# then edit .env and set OPENAI_API_KEY=sk-...
```

### 4. Run the app

```bash
streamlit run app.py
```

Open the URL Streamlit prints (default <http://localhost:8501>).

---

## 🧪 Tests

```bash
pytest
```

The suite covers the safety layer, sentiment/emotion analysis, screening
scoring, the database layer, the ML models and the chat engine.

---

## 📁 Project layout

```
Mental_Assistant_AI/
├── app.py                     # Streamlit entry point
├── requirements.txt
├── .env.example
├── src/
│   ├── config.py              # env-driven settings
│   ├── database.py            # SQLite schema + helpers
│   ├── features.py            # feature engineering for the ML models
│   ├── screening.py           # PHQ-9 / GAD-7 instruments
│   ├── content.py             # coping library, helplines, psychoeducation
│   ├── service.py             # app service layer
│   ├── assistant/             # chat_engine · safety · prompts
│   └── ml/                    # sentiment · personalization · stress · recommender
│       └── ...
│   └── ui/                    # one module per Streamlit page
├── scripts/
│   ├── generate_data.py       # synthetic dataset generators
│   ├── train_models.py        # train + evaluate + persist models
│   └── analyze_evaluation.py  # pilot-study statistics
├── tests/                     # pytest suite
└── docs/                      # report, diagrams, presentation
```

---

## 🔐 Privacy & ethics

- All data is stored **locally** in a SQLite file (`data/mindmate.db`); nothing
  is uploaded anywhere except, optionally, the text you send to the OpenAI API.
- The evaluation datasets are **synthetic** and clearly labelled as such, no
  real human data is used.
- The assistant never diagnoses, and it always defers clinical concerns to
  professionals and helplines.

---

## 📄 Licence

Released under the MIT Licence, see [`LICENSE`](LICENSE).
