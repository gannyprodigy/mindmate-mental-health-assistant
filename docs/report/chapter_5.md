# Chapter 5: System Implementation

## 5.1 Introduction

This chapter documents how the design presented in Chapter 4 was translated into working software. Where the previous chapter dealt with architecture, data flow and the entity-relationship model in the abstract, the focus here is concrete: the development environment, the layout of the codebase, the database, and the individual machine-learning and conversational modules that together make up MindMate. For each module I describe the key algorithm, quote a short, representative excerpt of the real source code, and then explain the logic in prose so that the reader can follow the reasoning behind each design decision rather than simply reading the listing.

Two themes recur throughout the implementation and are worth stating at the outset. The first is **graceful degradation**: the application is designed to work fully even when no external service is available. The conversational core can call the OpenAI Chat Completions API when a key is supplied, but it falls back to an entirely local, rule-based engine otherwise, so the system never breaks and never incurs a cost simply to demonstrate it. The second is **privacy by design**: all user data is held in a local SQLite database, no secrets are written into source code, and the large-language-model integration is strictly optional. These principles shaped many of the smaller choices described below.

A further commitment that runs through the whole project is **reproducibility**. Every stochastic step in model training is seeded with the fixed value `42`, the dependency versions are pinned, and the synthetic datasets are regenerated deterministically by scripts under version control. Anyone who clones the repository and runs the training script should obtain exactly the metrics reported in Chapter 6.

## 5.2 Development Environment and Tools

MindMate was developed in **Python 3.12**, chosen for its mature data-science ecosystem and first-class support across all the libraries the project depends on. Development followed standard Python practice: an isolated virtual environment created with the built-in `venv` module, with all third-party packages installed through `pip` from a pinned `requirements.txt`. Pinning exact versions (for example `scikit-learn==1.5.2`, `streamlit==1.39.0`, `vaderSentiment==3.3.2`) is deliberate, it guarantees that the numerical results and the user interface behave identically on any machine, which is essential for an academic submission that must be reproducible.

The principal tools and their roles are as follows:

- **Streamlit (1.39.0)**, the web-application framework that renders the multipage user interface, using the modern `st.navigation`/`st.Page` API.
- **scikit-learn (1.5.2)**, the machine-learning library providing `KMeans`, `LogisticRegression`, `StandardScaler`, `Pipeline` and `PCA` (Pedregosa et al., 2011).
- **vaderSentiment (3.3.2)**, the VADER rule-based sentiment analyser used to score the affect of free text (Hutto & Gilbert, 2014).
- **pandas, numpy and scipy**, for data manipulation, numerical arrays and the statistical tests used in the evaluation.
- **plotly** for interactive in-app charts and **matplotlib** for the static figures included in this report.
- **SQLite**, accessed through Python's standard-library `sqlite3` module, for persistence.
- **joblib (1.4.2)** for serialising the trained model pipelines to disk.
- **python-dotenv (1.0.1)** for loading configuration from a local `.env` file into environment variables.
- **pytest (8.3.3)** for the automated test suite (34 tests, all passing).
- **graphviz** for generating the structural diagrams referenced in earlier chapters.

The choice of Python and scikit-learn for the analytical core reflects the conventional, well-documented workflow recommended in standard texts on applied machine learning (Géron, 2019). Keeping the model-building stack identical to the one used everywhere in the data-science community also lowers the barrier for future maintainers.

## 5.3 Project Structure

The codebase is organised into clearly separated packages, each corresponding to one architectural layer described in Chapter 4. The top level contains the Streamlit entry point `app.py`, the pinned `requirements.txt`, a `.env.example` template, configuration files, and directories for data, models, scripts, tests and the report. The application logic lives under `src/`:

- `src/config.py`, central configuration resolved from environment variables.
- `src/database.py`, the SQLite persistence layer.
- `src/content.py`, static content (coping strategies, helpline resources).
- `src/features.py`, `src/screening.py`, `src/service.py`, feature assembly, PHQ-9/GAD-7 scoring, and the service layer that mediates between the UI and the lower layers.
- `src/ml/`, the machine-learning package: `sentiment.py`, `personalization.py`, `stress.py`, `recommender.py`.
- `src/assistant/`, the conversational package: `chat_engine.py`, `safety.py`, `prompts.py`.
- `src/ui/`, one module per Streamlit page (`home`, `chat`, `mood`, `screening_page`, `insights`, `resources`).

Offline tooling lives in `scripts/` (`generate_data.py`, `train_models.py`, `make_figures.py`, `make_diagrams.py`, `analyze_evaluation.py`), trained artefacts in `models/artifacts/`, and the test suite in `tests/`. This layered separation means the presentation code never imports machine-learning internals directly; everything passes through the service layer, which keeps the modules independently testable (Sommerville, 2016).

## 5.4 Database Implementation

Persistence uses SQLite through Python's `sqlite3` module. SQLite was selected because it is a serverless, file-based database requiring no installation or running daemon, which suits a self-contained student wellbeing application and keeps all data on the user's own machine, a meaningful privacy property given the sensitivity of mental-health information. The schema is normalised into five tables (`users`, `mood_logs`, `chat_messages`, `screening_results`, `recommendation_logs`) with a one-to-many relationship from `users` to each of the others.

The schema is declared as a single SQL string and applied idempotently at start-up. A representative fragment is shown below:

```python
CREATE TABLE IF NOT EXISTS chat_messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id         INTEGER NOT NULL,
    role            TEXT    NOT NULL,       -- 'user' | 'assistant'
    content         TEXT    NOT NULL,
    sentiment       REAL,
    risk_level      TEXT,                   -- 'none' | 'elevated' | 'crisis'
    created_at      TEXT    NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE
);
```

Each child table carries a `FOREIGN KEY ... ON DELETE CASCADE` constraint, so removing a user automatically removes all of their logs, messages, screenings and recommendations, important for honouring a deletion request without orphaning data. Note that `chat_messages` stores both the computed `sentiment` and the assessed `risk_level` alongside the text, so the safety and analytics behaviour is auditable after the fact.

All connections are obtained through a single context manager that enforces referential integrity and guarantees clean-up:

```python
@contextmanager
def get_connection(db_path: Optional[Path] = None):
    """Yield a SQLite connection with foreign keys enabled."""
    path = Path(db_path) if db_path else SETTINGS.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
```

The `PRAGMA foreign_keys = ON` statement is required because SQLite does not enforce foreign keys by default. Setting `row_factory` to `sqlite3.Row` lets rows be addressed by column name and converted to dictionaries, so the rest of the application works with plain Python objects rather than tuples. Every public function, `create_user`, `add_chat_message`, `add_mood_log`, and the rest, wraps its query inside this context manager, meaning no raw SQL ever leaks into the higher layers. The database file path itself is configurable through the `MINDMATE_DB_PATH` environment variable, and the live database is excluded from version control via `.gitignore`.

## 5.5 Sentiment and Emotion Analysis Module

The sentiment module in `src/ml/sentiment.py` provides the affect signal used across the application, on mood-journal notes, on chat messages, and as one of the six features driving the machine-learning models. The primary scorer is **VADER** (Valence Aware Dictionary and sEntiment Reasoner), a lexicon- and rule-based model that is particularly well suited to the short, informal, emoji-laden text that students actually write (Hutto & Gilbert, 2014). VADER requires no training, is deterministic, and runs locally, all of which fit the project's reproducibility and privacy goals.

On top of the raw VADER output, a lightweight emotion layer maps text onto a small set of student-relevant states. A coarse keyword lexicon defines each emotion:

```python
EMOTION_LEXICON: Dict[str, list[str]] = {
    "anxiety": [
        "anxious", "anxiety", "panic", "nervous", "worried", "worry", "scared",
        "afraid", "overwhelm", "overwhelmed", "tense", "restless", "dread",
    ],
    "sadness": [
        "sad", "down", "depressed", "hopeless", "lonely", "empty", "cry",
        "crying", "miserable", "worthless", "numb", "unmotivated",
    ],
    "stress": [
        "stress", "stressed", "pressure", "deadline", "exam", "exams",
        "assignment", "burnout", "burnt out", "workload", "overworked",
    ],
    ...
}
```

The dominant emotion is chosen by counting keyword hits per group and returning the highest-scoring group, defaulting to `"neutral"` when nothing matches. The core scoring routine then combines both signals:

```python
def analyze(text: str) -> SentimentResult:
    """Run sentiment + emotion analysis on a single piece of text."""
    if not text or not text.strip():
        return SentimentResult(0.0, 0.0, 1.0, 0.0, "neutral", "neutral")

    scores = _ANALYZER.polarity_scores(text)
    compound = float(scores["compound"])
    return SentimentResult(
        compound=compound,
        positive=float(scores["pos"]),
        neutral=float(scores["neu"]),
        negative=float(scores["neg"]),
        label=_label_from_compound(compound),
        emotion=detect_emotion(text),
    )
```

The function returns a structured `SentimentResult` dataclass carrying the compound score (in the range -1 to +1), the positive/neutral/negative proportions, a three-way label, and the detected emotion. Empty input short-circuits to a neutral result, avoiding meaningless scores on blank notes. The label is derived from the compound score using VADER's conventional thresholds (`>= 0.05` positive, `<= -0.05` negative, otherwise neutral). This dual output is deliberate: the continuous `compound` value feeds the numerical feature vector, while the categorical `emotion` is consumed directly by the recommender and the offline reply engine, which need a discrete label to select an appropriate strategy or opening line (Liu, 2012).

## 5.6 Student Segmentation Module

The segmentation module in `src/ml/personalization.py` groups students into behavioural **segments** using unsupervised **K-Means** clustering (MacQueen, 1967; Lloyd, 1982). Clustering operates on the six canonical features, average mood, average sleep hours, PHQ-9 score, GAD-7 score, average sentiment and engagement, whose order is fixed in a `FEATURE_COLUMNS` constant shared with the stress classifier to guarantee consistency between training and inference.

Training fits a two-step pipeline of standardisation followed by clustering:

```python
def train_segmenter(matrix: np.ndarray, n_clusters: int = 4,
                    random_state: int = 42) -> StudentSegmenter:
    """Fit the standardisation + K-Means pipeline on a feature matrix."""
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=n_clusters, n_init=10,
                              random_state=random_state)),
        ]
    )
    pipeline.fit(matrix)
    mapping = label_clusters(pipeline, n_clusters)
    return StudentSegmenter(pipeline=pipeline, cluster_to_label=mapping)
```

Wrapping the `StandardScaler` and `KMeans` in a scikit-learn `Pipeline` ensures the identical scaling is applied at inference time, which matters because K-Means is distance-based and therefore sensitive to feature magnitude, without standardisation, the PHQ-9 range of 0-27 would dominate the 1-5 mood range. The `random_state=42` and `n_init=10` arguments make the result reproducible while running ten initialisations and keeping the best, mitigating K-Means' sensitivity to starting centroids.

A subtlety of K-Means is that cluster identifiers are arbitrary: cluster `0` carries no meaning. To produce labels that are genuinely interpretable, *Thriving*, *Coping*, *At-Risk*, *High-Need*, the centroids are ranked on a composite wellbeing index:

```python
def _wellbeing_index(centroid: np.ndarray) -> float:
    """Composite score: higher = better wellbeing."""
    avg_mood, avg_sleep, phq9, gad7, sentiment, _engagement = centroid
    return (
        (avg_mood / 5.0)
        + (min(avg_sleep, 9.0) / 9.0)
        + ((sentiment + 1.0) / 2.0)
        - (phq9 / 27.0)
        - (gad7 / 21.0)
    )
```

Before computing this index, the centroids are passed back through `scaler.inverse_transform` so they are expressed in their original units. The index rewards higher mood, sleep and sentiment and penalises higher PHQ-9 and GAD-7 scores, each term normalised to a comparable scale. Clusters are then sorted by this index and the ordered labels assigned in descending order of wellbeing, so the names remain meaningful across re-trainings even though the raw cluster ids change. On the synthetic dataset this segmenter achieves a silhouette score of 0.209 with segment sizes of At-Risk 198, Coping 176, Thriving 119 and High-Need 107, figures discussed in Chapter 6 and validated using the silhouette criterion of Rousseeuw (1987). At runtime the fitted pipeline and its cluster-to-label mapping are serialised with joblib and reloaded by `StudentSegmenter.load`.

## 5.7 Stress Classification Module

The stress module in `src/ml/stress.py` adds a supervised layer: a **multinomial Logistic Regression** that predicts a student's stress level (`Low`, `Moderate` or `High`) from the same six features. Logistic Regression was chosen over more opaque models because its outputs are well-calibrated probabilities and its behaviour is interpretable, a property that matters in a wellbeing setting, where a student deserves a transparent rather than a black-box judgement (Shatte et al., 2019).

```python
def train_classifier(matrix: np.ndarray, labels: np.ndarray,
                     random_state: int = 42) -> StressClassifier:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]
    )
    pipeline.fit(matrix, labels)
    return StressClassifier(pipeline=pipeline, classes=STRESS_LEVELS)
```

As with the segmenter, standardisation is bundled into the pipeline so that the gradient-based optimiser converges reliably and the same transformation is reapplied at prediction time; `max_iter=1000` gives the solver ample iterations to converge and the fixed `random_state` keeps results reproducible. The classifier exposes two prediction methods. `predict` returns the single most likely stress label, while `predict_proba` returns the full probability distribution:

```python
def predict_proba(self, features: Dict[str, float]) -> Dict[str, float]:
    vector = np.array([[features.get(c, 0.0) for c in FEATURE_COLUMNS]])
    probs = self.pipeline.predict_proba(vector)[0]
    ordered = list(self.pipeline.classes_)
    return {str(cls): float(p) for cls, p in zip(ordered, probs)}
```

The probability output is what the Insights page surfaces to the user, turning an opaque label into a transparent "how confident is the model" view. Both methods read features through `features.get(c, 0.0)`, so a missing field degrades gracefully to zero rather than raising. On a stratified 25% hold-out of 150 samples the classifier attains an accuracy of 0.900 and a macro-F1 of 0.889, with full per-class metrics reported in Chapter 6 (Pedregosa et al., 2011). The trained pipeline is persisted with joblib and reloaded at runtime, exactly mirroring the segmenter so that both models share a single, predictable lifecycle.

## 5.8 Recommendation Engine

The recommender in `src/ml/recommender.py` decides which coping strategies a student sees. It is a **hybrid, content-based and rule-weighted** engine that combines three signals, the detected emotion, the student's segment, and the predicted stress level, and crucially produces an explanation for every score. A black-box recommender would be inappropriate in a wellbeing context, so transparency was treated as a first-class requirement (Ricci et al., 2015).

The scoring of a single strategy makes the weighting scheme explicit:

```python
def _score_strategy(strategy: Dict, ctx: RecommendationContext) -> tuple[float, str]:
    """Return (score, human-readable rationale) for one strategy."""
    score = 0.0
    reasons: List[str] = []

    # 1. Emotion match, the strongest single signal.
    if ctx.emotion in strategy["emotions"]:
        score += 2.0
        reasons.append(f"matches your current feeling ({ctx.emotion})")

    # 2. Segment preference.
    preferred = _SEGMENT_PREFERENCES.get(ctx.segment, set())
    if strategy["category"] in preferred:
        score += 1.0
        reasons.append(f"suits the '{ctx.segment}' support plan")

    # 3. Stress-driven de-escalation boost.
    stress_w = _STRESS_WEIGHT.get(ctx.stress_level, 0.5)
    if strategy["category"] in _DEESCALATION_CATEGORIES:
        score += 1.5 * stress_w
        if stress_w > 0:
            reasons.append("helps settle high stress quickly")

    rationale = "; ".join(reasons) if reasons else "a generally helpful technique"
    return score, rationale
```

Three weighted rules combine additively. A match between the detected emotion and a strategy's target emotions contributes the strongest signal (weight 2.0). A match between the strategy's category and the student's segment preferences adds 1.0. Finally, when a student is stressed, calming and grounding categories receive a de-escalation boost of up to 1.5, scaled by a stress weight that is 0.0 for *Low*, 0.5 for *Moderate* and 1.0 for *High*, so a highly stressed student is steered firmly toward settling exercises, while a relaxed one is not. As each rule fires it appends a plain-language reason, and these are joined into a `rationale` string returned alongside the score.

The public `recommend` function scores every strategy in the catalogue, sorts them by descending score and returns the top *k*. Because the score is a simple sum of named contributions, the ranking is fully explainable: the Insights page can show not just *what* was recommended but *why*. The same engine is reused by the chat layer through the convenience wrapper `recommend_from_signals`, ensuring that the advice given in conversation is consistent with the advice shown on the dashboard.

## 5.9 Conversational Assistant and OpenAI Integration

The conversational core lives in `src/assistant/chat_engine.py` and exposes a single public function, `generate_reply`. Its design embodies the graceful-degradation principle: it can produce a reply using the **OpenAI Chat Completions API** when a key is configured, or fall back to a fully **offline, rule-based engine** when one is not. Because the fallback is always available, the application runs end-to-end with no external dependency and no API cost, vital for a reproducible submission and a dependable live demo.

When a key is present, the engine assembles a message list and calls the API:

```python
def _openai_reply(message: str, history: List[Dict], note: str) -> Optional[str]:
    """Call the OpenAI API; return None on any failure so we can fall back."""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=SETTINGS.openai_api_key)
        messages = [{"role": "system", "content": prompts.SYSTEM_PROMPT},
                    {"role": "system", "content": note}]
        # Include a trimmed window of prior turns for continuity.
        for turn in history[-8:]:
            role = "assistant" if turn["role"] == "assistant" else "user"
            messages.append({"role": role, "content": turn["content"]})
        messages.append({"role": "user", "content": message})

        response = client.chat.completions.create(
            model=SETTINGS.openai_model,
            messages=messages,
            temperature=0.7,
            max_tokens=400,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        # Any error (missing package, network, quota) -> use offline engine.
        return None
```

Two design choices deserve emphasis. First, the persona-defining `SYSTEM_PROMPT` and a personalisation `note` (built from the current emotion, segment and stress level) are injected as system messages, so the model adopts MindMate's empathetic, non-clinical voice and tailors its tone without ever naming the internal labels to the student. Only the last eight turns of history are sent, bounding token use and cost. Second, and most importantly, the entire call is wrapped in a broad `try/except` that returns `None` on *any* failure: a missing package, no network, an exhausted quota, or an invalid key. Returning `None` rather than raising is what lets the fallback take over without the student ever noticing an interruption.

When the API path is unavailable, the offline engine constructs a supportive reply from a small set of emotion-specific openers plus a single recommendation drawn from the same recommender used elsewhere:

```python
def _offline_reply(message: str, emotion: str, segment: str, stress_level: str) -> str:
    """Construct a supportive, personalised reply with no external calls."""
    opener = _EMOTION_OPENERS.get(emotion, _EMOTION_OPENERS["neutral"])
    recs = recommend_from_signals(emotion=emotion, segment=segment,
                                  stress_level=stress_level, top_k=1)
    parts = [opener]
    ...
```

The public `generate_reply` function ties the layers together. It first assesses risk and analyses sentiment on every message, regardless of mode. If a crisis is detected it returns the safe crisis response immediately, never reaching the language model. Otherwise it tries the OpenAI path when `SETTINGS.llm_enabled` is true, falls back to the offline reply when that returns `None`, and finally appends a gentle resource nudge for elevated distress. The result is wrapped in an `AssistantReply` dataclass that records the text, risk level, sentiment, emotion and `source` (`'openai'`, `'offline'` or `'safety'`), so the system always knows, and can store, which path produced each reply.

## 5.10 Safety and Crisis-Detection Layer

Safety is implemented in `src/assistant/safety.py` and is the most consequential module in the system. Following the precedent of earlier therapeutic conversational agents (Fitzpatrick et al., 2017; Inkster et al., 2018), risk screening runs *before* any reply is generated, on every single message. The layer is deliberately **rule-based and conservative**: in a wellbeing product a false positive (showing helplines unnecessarily) is far cheaper than a false negative (missing a genuine crisis), and a transparent set of patterns is easier to audit and reason about than a statistical classifier whose mistakes are hard to predict.

Risk is detected with two ordered lists of regular-expression patterns matched on word boundaries:

```python
CRISIS_PATTERNS: List[str] = [
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\btake my (own )?life\b",
    r"\bsuicid(e|al)\b",
    r"\bwant to die\b",
    r"\bwish i (was|were) dead\b",
    r"\bdon'?t want to (be alive|live)\b",
    r"\bno reason to live\b",
    r"\bharm(ing)? myself\b",
    r"\bself[- ]harm\b",
    r"\boverdose\b",
    r"\bcan'?t go on\b",
    r"\bbetter off without me\b",
]
```

The `\b` word-boundary anchors are important: they let the patterns catch genuine expressions of self-harm while avoiding false hits on innocuous phrasing such as "killing it in exams." A second, separate list of `ELEVATED_PATTERNS` (for example `\bhopeless\b`, `\bcan'?t cope\b`, `\bgiving up\b`, `\bnumb\b`) captures distress that is serious but not acute. The two lists are compiled into case-insensitive regular expressions, and `assess_risk` evaluates them in strict priority order:

```python
def assess_risk(text: str) -> RiskAssessment:
    """Classify the risk level of a user message."""
    if not text:
        return RiskAssessment("none", [])

    crisis_hits = _CRISIS_RE.findall(text)
    if crisis_hits:
        flat = [h if isinstance(h, str) else next(filter(None, h), "") for h in crisis_hits]
        return RiskAssessment("crisis", [m for m in flat if m])

    elevated_hits = _ELEVATED_RE.findall(text)
    if elevated_hits:
        flat = [h if isinstance(h, str) else next(filter(None, h), "") for h in elevated_hits]
        return RiskAssessment("elevated", [m for m in flat if m])

    return RiskAssessment("none", [])
```

Crisis patterns are checked first, so any acute signal takes precedence over a merely elevated one. When a crisis is matched the chat engine bypasses both the language model and the offline generator entirely and returns `crisis_response()`, which provides a brief validating message and prominently lists crisis and suicide-prevention helplines (iCall, Vandrevala Foundation, Tele-MANAS, AASRA, and the IASP directory) drawn from the resources catalogue. The assistant explicitly does *not* attempt to counsel a student in crisis on its own. For elevated risk, a normal reply is generated and a gentle nudge toward professional support is appended. The assessed level is stored on every chat message so the behaviour is fully auditable. It must be stated plainly that this keyword-based approach has limits, it cannot understand sarcasm, indirect expression or unanticipated phrasing, and MindMate is therefore explicitly a self-help tool, not a clinical or emergency service.

## 5.11 User Interface Implementation

The user interface is built with **Streamlit** and organised as a multipage application. The entry point `app.py` configures the page, initialises the database schema on first launch, and assembles the navigation using Streamlit's modern `st.Page`/`st.navigation` API:

```python
def main() -> None:
    _ensure_user()
    _sidebar()

    pages = [
        st.Page(home.render, title="Home", icon="🏠", default=True),
        st.Page(chat.render, title="Talk to MindMate", icon="💬"),
        st.Page(mood.render, title="Mood Tracker", icon="📈"),
        st.Page(screening_page.render, title="Self-Check", icon="📝"),
        st.Page(insights.render, title="Insights", icon="🧠"),
        st.Page(resources.render, title="Resources", icon="📚"),
    ]
    st.navigation(pages).run()
```

Each page is a separate module under `src/ui/` exposing a `render` function, which keeps the presentation code modular and independently testable. Before any page renders, `_ensure_user` performs a lightweight onboarding flow: if no user is active in session state it shows a consent form (including an explicit disclaimer that MindMate is not a medical service) and, once submitted, creates a user record and stores the new `user_id` in `st.session_state`. The sidebar transparently reports whether the assistant is running in live OpenAI mode or the offline engine, so the user always knows which path is active.

Streamlit's **session state** is used throughout to bridge the framework's re-run-on-every-interaction execution model. The chat page is the clearest example. On first load it lazily pulls stored history from the database into `st.session_state.chat_messages`, then drives the conversation with `st.chat_message` for rendering and `st.chat_input` for the prompt box:

```python
def render() -> None:
    user_id = st.session_state.user_id
    ...
    messages = _load_history(user_id)
    ...
    prompt = st.chat_input("Type how you're feeling...")
    if not prompt:
        return

    with st.chat_message("user"):
        st.markdown(prompt)
    messages.append({"role": "user", "content": prompt})

    profile = service.profile_for_user(user_id)
    with st.chat_message("assistant"):
        with st.spinner("MindMate is thinking..."):
            reply = chat_engine.generate_reply(
                prompt,
                history=messages[:-1],
                segment=profile["segment"],
                stress_level=profile["stress_level"],
            )
        st.markdown(reply.text)
```

When the user submits a message, the page displays it, retrieves the user's personalisation profile (segment and stress level) through the service layer, and calls `generate_reply` with the conversation history. A spinner gives feedback while the reply is produced. If the assessed risk level is `crisis`, an additional prominent error banner directs the student to the Resources page. Finally both turns are persisted to the database together with their sentiment and risk metadata, so the conversation survives a page refresh or a return visit. The other pages follow the same pattern, the Mood Tracker captures daily mood, energy, sleep and a free-text note (scored with VADER), the Self-Check page administers the PHQ-9 and GAD-7 instruments (Kroenke et al., 2001; Spitzer et al., 2006), and the Insights page renders the segment, the stress probabilities, the six features and the recommendation rationale to keep the machine learning transparent to the user.

## 5.12 Version Control

The project was developed under **Git** for version control, following standard practice in modern software engineering (Sommerville, 2016). Git provided an incremental history of the implementation, made it safe to experiment on feature changes, and supports the reproducibility goals of the project by recording exactly which version of the code produced any given result.

A carefully curated `.gitignore` enforces the project's security and reproducibility principles directly at the version-control boundary. Several categories are deliberately excluded:

- **Secrets and environment files**, `.env` is ignored so that no API key is ever committed. Only the `.env.example` template, which contains empty placeholders, is tracked.
- **The live database**, `data/mindmate.db` is ignored so that no real user data is checked in, while the schema and data-generation scripts remain under version control.
- **Trained model artefacts**, the `*.joblib` files in `models/artifacts/` are excluded to avoid committing large binaries; they are regenerated deterministically after cloning by running `scripts/train_models.py`.
- **Generated and environment clutter**, virtual-environment directories, `__pycache__`, the pytest cache and OS/IDE files.

This arrangement reflects a clear separation between *source* (tracked) and *generated or sensitive artefacts* (ignored). Because configuration is read entirely from environment variables through `src/config.py`, no secret ever appears in tracked source, and because the models are reproducible from seeded scripts, omitting the binaries costs nothing. Together with the pinned dependency versions and the fixed random seed of 42, this version-control discipline ensures that a reviewer can clone the repository, recreate the virtual environment, regenerate the data and models, and obtain precisely the results reported in the next chapter.
