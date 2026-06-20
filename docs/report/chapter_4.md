# Chapter 4: System Design

## 4.1 Introduction

This chapter translates the requirements established in the preceding chapters into a concrete technical blueprint for MindMate, the AI-powered mental-health assistant for students. Where the earlier analysis described *what* the system must do, the present chapter describes *how* the system is organised internally so that those obligations can be met reliably, safely, and in a way that remains comprehensible to both maintainers and examiners. Design, in the sense used here, is the bridge between abstract requirement statements and the source code that ultimately realises them (Sommerville, 2016).

The design is presented through a series of complementary views. The structural view explains how the software is decomposed into layers and packages and how the principal classes relate to one another. The data view documents the persistent storage as a normalised relational schema together with its entity-relationship model. The behavioural view captures the dynamic interactions between components when a student sends a chat message, including the crisis-safety branch that is central to the application's responsible operation. Finally, the chapter addresses the user-interface design and the design of the machine-learning algorithms that personalise the assistant's behaviour. Taken together these views provide a single coherent account of the system that can be implemented, tested, and evaluated.

Throughout the chapter the description is kept faithful to the implemented system rather than to an idealised one. MindMate is a Streamlit web application written in Python 3.12, persisting data in SQLite and drawing on scikit-learn for its machine-learning components. The crisis-safety layer, the offline fallback chat engine, and the transparent recommender are emphasised because they embody the design priorities that distinguish a responsible student wellbeing tool from a generic chatbot.

## 4.2 Design Objectives and Principles

The design of MindMate is governed by a small set of objectives that follow directly from the project's three aims: to build an assistant for students, to personalise its support with machine learning, and to evaluate its effectiveness. From these aims the following design objectives were derived.

- **Safety first.** No component may generate or display a reply to a potentially distressed student before that message has been screened for crisis risk. The architecture must make it structurally difficult to bypass this check.
- **Separation of concerns.** Presentation, application logic, domain intelligence, and persistence must be cleanly separated so that each can be developed, reasoned about, and tested in isolation.
- **Testability without the user interface.** The core behaviour of the system must be exercisable by automated tests that do not depend on the Streamlit runtime, which is why application logic is concentrated in a thin service layer rather than scattered across page scripts.
- **Transparency.** Because the system makes inferences about a student's psychological state, every machine-derived output, including segment assignment, predicted stress level, and recommendations, must be explainable rather than opaque.
- **Graceful degradation.** The application must remain fully usable when optional external dependencies, in particular the language-model API, are unavailable; an offline rule-based engine guarantees this.
- **Privacy by design.** Data is stored locally in a single SQLite database, the schema is minimal, and deletion of a user cascades to all of their records.

These objectives are realised through established design principles. A layered architecture enforces separation of concerns and constrains the direction of dependencies, a recognised structural style for information systems of this kind (Sommerville, 2016; Pressman & Maxim, 2014). High cohesion within modules and low coupling between them are pursued so that, for example, the sentiment component can be modified without disturbing the database layer. Information hiding is applied so that no part of the application issues raw SQL; all persistence flows through small, well-named helper functions. Finally, a conservative bias is deliberately encoded into the safety design: a false positive, in which a benign message is treated as concerning, is considered far less costly than a false negative, in which genuine distress is missed.

## 4.3 System Architecture

MindMate adopts a layered architecture in which each layer offers services to the layer above and depends only on the layer below. This style localises change, clarifies responsibilities, and supports incremental testing, and it is widely recommended for applications that combine a user interface with persistent data and analytical logic (Sommerville, 2016). Figure 4.1 shows the four principal layers and the optional external dependency.

[[FIG: diagram_architecture.png | Figure 4.1: MindMate layered system architecture]]

The **presentation layer** comprises the Streamlit user interface. It is organised as a multipage application assembled through `st.navigation`, with six pages: Home, Talk to MindMate, Mood Tracker, Self-Check, Insights, and Resources. The pages are deliberately thin; they are responsible for rendering widgets, collecting input, and displaying results, but they delegate all decision-making to the layer beneath. Keeping the pages thin is what allows the application's behaviour to be tested independently of the browser-based runtime.

The **service layer** is implemented in `service.py` and acts as the single point of contact between the user interface and the underlying intelligence and storage. It exposes a compact set of operations such as computing a user's live machine-learning profile, calculating the 0-100 wellness index for the dashboard headline, and producing personalised recommendations. By centralising orchestration here, the design ensures that the various lower-level modules are composed in one place and that the user interface never needs to know how a recommendation or a stress prediction is actually produced.

Beneath the service layer sit two sibling packages that together constitute the **domain-intelligence layer**. The *assistant* package contains the conversational components: the chat engine, the crisis-safety module, and the prompt templates. The *machine-learning* package contains the sentiment analyser, the K-Means student segmenter (`personalization.py`), the Logistic-Regression stress classifier (`stress.py`), and the hybrid recommender (`recommender.py`). These packages encapsulate the analytical heart of the system. Crucially, the safety module within the assistant package is invoked before the chat engine, so that risk screening cannot be skipped by the normal control flow.

The **data layer** is implemented in `database.py` and wraps a single SQLite database accessed through Python's `sqlite3` module. It defines the schema, manages connections with foreign-key enforcement enabled, and provides typed helper functions for every read and write. No higher layer issues SQL directly.

Finally, the **OpenAI Chat Completions API** is shown as an optional external dependency rather than a core layer. When an API key is configured, the chat engine uses the language model (by default `gpt-4o-mini`) to generate empathetic replies; when no key is present, an offline rule-based engine produces a reply instead. This arrangement satisfies the graceful-degradation objective: the loss of an external service downgrades the quality of conversation but never disables the application.

## 4.4 Class Design

The class design refines the layered architecture into the concrete types that carry the system's state and behaviour. The principal classes are small, focused, and, where they represent the result of a computation, implemented as immutable data carriers so that results can be passed safely between layers. Figure 4.2 depicts the core classes and their associations.

[[FIG: diagram_class.png | Figure 4.2: Core class diagram]]

**Settings** is the central configuration object. It holds runtime parameters such as the database path, the model file locations, and the optional API key, and it is loaded once from the environment using python-dotenv. A single shared instance (`SETTINGS`) is consumed by the database and machine-learning modules, which keeps configuration in one place and avoids hard-coded paths.

**StudentSegmenter** wraps a fitted scikit-learn pipeline together with a mapping from raw cluster identifiers to human-readable labels. Its `predict` method accepts a dictionary of the six canonical features, assembles them into the fixed feature order, runs the pipeline, and returns the segment label associated with the resulting cluster. The class also offers `predict_many` for batch prediction, `strategy_for` to retrieve the support strategy attached to a label, and `save`/`load` methods that serialise the pipeline and its label mapping with joblib. By bundling the pipeline and the mapping into one serialisable object, the design guarantees that a loaded model always carries its interpretation with it.

**StressClassifier** plays the analogous role for supervised prediction. It encapsulates a multinomial Logistic-Regression model and exposes `predict`, which returns a categorical stress level (Low, Moderate, or High), and `predict_proba`, which returns the class probability distribution used by the Insights page to display the model's confidence. Like the segmenter, it supports joblib persistence so that training and inference are cleanly separated.

**SentimentResult** is an immutable value object returned by the sentiment analyser. It captures the VADER compound score together with the dominant emotion inferred from a custom emotion-keyword lexicon (Hutto & Gilbert, 2014). Representing the analysis as a single typed result, rather than as a loose tuple, makes the downstream code that consumes it self-documenting.

**RiskAssessment** is the output of the crisis-safety module. It records the detected risk level, one of `none`, `elevated`, or `crisis`, and any safe message and helpline information that must accompany the reply. Because risk screening governs whether a normal reply is generated at all, this small object has outsized importance in the control flow.

**AssistantReply** is the value object returned by the chat engine to the service and presentation layers. It bundles the textual reply with the metadata needed for storage and display, including the risk level and sentiment, so that a single object fully describes the assistant's turn.

**Recommender** is realised through the recommender module, which scores the nine coping strategies against the current emotional and segment signals and returns ranked recommendations, each accompanied by a human-readable rationale. The accompanying recommendation objects carry a title, a category, and the explanatory rationale string, which together support the transparency objective.

The associations among these classes follow the dependency direction of the architecture. The service layer holds references to a segmenter and a classifier, obtains a `SentimentResult` from the sentiment analyser and a `RiskAssessment` from the safety module, and assembles an `AssistantReply`. None of the value objects depend on the service or presentation layers, which keeps the lower layers reusable and independently testable.

## 4.5 Database Design

### 4.5.1 Entity-Relationship Model

Persistent data is modelled relationally and stored in a single SQLite database. The schema is deliberately normalised so that it can be documented cleanly as an entity-relationship model and so that each fact is recorded in exactly one place. There are five entities: a central **User** entity and four dependent entities that record the different kinds of activity a student generates over time, namely **Mood Log**, **Chat Message**, **Screening Result**, and **Recommendation Log**.

The model is organised around a single one-to-many relationship pattern. Each user may own many mood logs, many chat messages, many screening results, and many recommendation logs, while each of those records belongs to exactly one user. The User entity is therefore the parent in four separate one-to-many associations, and the four activity entities are the children. Figure 4.3 presents this structure.

[[FIG: diagram_er.png | Figure 4.3: Entity-relationship diagram]]

Referential integrity is enforced at the database level. Every child table declares a foreign key referencing `users(id)` with an `ON DELETE CASCADE` rule, and foreign-key checking is switched on for every connection through the `PRAGMA foreign_keys = ON` statement. The practical consequence is that deleting a user automatically and atomically removes all of that user's mood logs, chat messages, screening results, and recommendation logs. This supports the privacy-by-design objective directly: a single deletion erases an individual's entire footprint, leaving no orphaned records behind.

### 4.5.2 Schema and Table Structures

The five tables are created by an idempotent schema script that uses `CREATE TABLE IF NOT EXISTS`, so initialising an existing database is harmless. Every table uses an auto-incrementing integer primary key, and every record carries an ISO-8601 timestamp recorded at insertion time. Timestamps are stored as text, which keeps the schema portable and chronologically sortable. The following tables document each relation as implemented in `database.py`.

**users**

| Column | Type | Constraint | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier for the student |
| name | TEXT | NOT NULL | Display name of the student |
| age | INTEGER |, | Age in years (optional) |
| course | TEXT |, | Programme of study (optional) |
| year_of_study | INTEGER |, | Current year of study (optional) |
| segment | TEXT |, | Most recently assigned ML segment label |
| created_at | TEXT | NOT NULL | ISO-8601 timestamp of account creation |

**mood_logs**

| Column | Type | Constraint | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier for the mood entry |
| user_id | INTEGER | NOT NULL, FK → users(id) ON DELETE CASCADE | Owning student |
| mood_score | INTEGER | NOT NULL | Self-reported mood, 1 (very low) to 5 (very good) |
| energy_score | INTEGER |, | Self-reported energy, 1 to 5 |
| sleep_hours | REAL |, | Reported hours of sleep |
| note | TEXT |, | Free-text note for the day |
| sentiment | REAL |, | VADER compound sentiment of the note |
| logged_at | TEXT | NOT NULL | ISO-8601 timestamp of the log |

**chat_messages**

| Column | Type | Constraint | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier for the message |
| user_id | INTEGER | NOT NULL, FK → users(id) ON DELETE CASCADE | Owning student |
| role | TEXT | NOT NULL | Author of the message, `user` or `assistant` |
| content | TEXT | NOT NULL | Message text |
| sentiment | REAL |, | Sentiment score of the message |
| risk_level | TEXT |, | Screening outcome: `none`, `elevated`, or `crisis` |
| created_at | TEXT | NOT NULL | ISO-8601 timestamp of the message |

**screening_results**

| Column | Type | Constraint | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier for the screening |
| user_id | INTEGER | NOT NULL, FK → users(id) ON DELETE CASCADE | Owning student |
| instrument | TEXT | NOT NULL | Questionnaire used, `PHQ-9` or `GAD-7` |
| total_score | INTEGER | NOT NULL | Summed item score |
| severity | TEXT | NOT NULL | Severity band derived from published cut-offs |
| taken_at | TEXT | NOT NULL | ISO-8601 timestamp of completion |

**recommendation_logs**

| Column | Type | Constraint | Description |
|--------|------|------------|-------------|
| id | INTEGER | PRIMARY KEY AUTOINCREMENT | Unique identifier for the log entry |
| user_id | INTEGER | NOT NULL, FK → users(id) ON DELETE CASCADE | Owning student |
| strategy | TEXT | NOT NULL | Title of the recommended coping strategy |
| category | TEXT |, | Category of the strategy |
| context | TEXT |, | Signals behind the recommendation (emotion, segment) |
| created_at | TEXT | NOT NULL | ISO-8601 timestamp of the recommendation |

The screening table reflects the standard instruments used by the Self-Check page, the nine-item PHQ-9 for depressive symptoms (Kroenke et al., 2001) and the seven-item GAD-7 for anxiety (Spitzer et al., 2006), with each result reduced to a total score and a severity band. The `context` column of the recommendation log preserves the signals that produced a recommendation, which is what later allows the Insights page to reconstruct and display the rationale behind past suggestions. Access to every table is mediated by dedicated helper functions, for instance `add_mood_log`, `add_chat_message`, and `get_screening_results`, so that the rest of the application interacts with the database through a small, audited surface rather than ad-hoc queries.

## 4.6 Behavioural Design (Sequence and Activity)

The structural and data designs describe the static shape of the system; the behavioural design describes how the components collaborate over time. The most important interaction in MindMate is the handling of a chat message, because it is here that the safety obligation, the sentiment analysis, the personalisation, and the persistence all come together in a single ordered flow.

[[FIG: diagram_sequence.png | Figure 4.4: Sequence diagram for sending a chat message]]

Figure 4.4 shows the sequence of method calls. The interaction begins when the student submits text on the Talk to MindMate page. The page passes the message to the service layer, which first invokes the crisis-safety module to obtain a `RiskAssessment`. This screening happens before any reply is generated, which is the structural guarantee that distress cannot be answered with a casual response. The service then requests a `SentimentResult` from the sentiment analyser, both to inform the reply and to be stored with the message. With the risk and sentiment known, the service delegates reply generation to the chat engine. When an API key is configured, the engine calls the OpenAI Chat Completions API; otherwise it falls back to the offline rule-based engine. The engine returns an `AssistantReply`, which the service persists by writing both the student's message and the assistant's response to the `chat_messages` table, each annotated with its sentiment and risk level. The completed reply is finally returned to the page for display.

The same behaviour is captured from a control-flow perspective in the activity diagram of Figure 4.5, which makes the crisis branch explicit.

[[FIG: diagram_activity.png | Figure 4.5: Activity diagram for message handling with the safety branch]]

After the message is received, the safety module classifies its risk level, and the flow branches on the result. If the level is **crisis**, the system does not attempt to counsel. Instead it returns a brief, validating message accompanied by crisis helplines, including iCall, the Vandrevala Foundation, Tele-MANAS, and AASRA, and the interaction terminates without invoking the conversational engine. If the level is **elevated**, a normal empathetic reply is generated and then augmented with a gentle nudge toward supporting resources. If the level is **none**, the engine produces an ordinary supportive reply. In every branch the messages and their metadata are written to the database before the reply is shown. This conservative branching, in which any sign of acute risk short-circuits the ordinary conversational path, embodies the principle that a missed crisis is a far worse outcome than an over-cautious one. The rule-based nature of this safeguard is acknowledged as a limitation: it is intentionally simple and explainable, not a clinical risk-assessment instrument.

## 4.7 User-Interface Design

The user interface is built with Streamlit and organised as a multipage application assembled through `st.navigation`. Six pages divide the application's functionality into clear, task-oriented areas, each accessible from a persistent sidebar.

- **Home** presents the dashboard headline: a 0-100 wellness index, the student's current support segment, the predicted stress level, a concise support plan, and the top personalised recommendations. It is designed to give an at-a-glance, encouraging summary rather than to overwhelm the visitor with detail.
- **Talk to MindMate** is the conversational page where the safety-screened chat takes place. It is the most carefully designed screen because every message passes through the crisis-safety layer before a reply appears.
- **Mood Tracker** allows daily logging of mood and energy on a one-to-five scale, sleep hours, and a free-text note, and it renders trend charts so that the student can observe patterns over time.
- **Self-Check** administers the PHQ-9 and GAD-7 questionnaires using the standard 0-3 response scale and published severity bands, with the ninth PHQ-9 item triggering a self-harm safeguard.
- **Insights** exposes the machine-learning internals transparently, showing the assigned segment, the stress probabilities, the six underlying features, and the rationale behind recommendations.
- **Resources** gathers Indian and global helplines, psychoeducation, and the full coping-strategy library in one reliably reachable place.

The visual design adopts a calming tone appropriate to a wellbeing context: a restrained, soft colour palette, generous spacing, and plain, supportive language rather than clinical jargon. Interactive charts are rendered with Plotly so that students can explore their own trends. Accessibility was a design consideration throughout. The interface relies on simple, linear layouts that read well on smaller screens, uses descriptive labels on every input widget, avoids conveying meaning through colour alone by pairing colour with text labels for severity and risk, and keeps reading level low so that the content is approachable during periods of distress. Crisis information, in particular, is presented in clear, high-contrast text and is never hidden behind additional interaction.

## 4.8 Algorithm and Model Design

The personalisation that distinguishes MindMate rests on three designed algorithms: an unsupervised segmenter, a supervised stress classifier, and a transparent hybrid recommender. All three operate on the same six canonical features, computed for each student in a fixed order: average mood, average sleep hours, PHQ-9 score, GAD-7 score, average sentiment, and engagement. Fixing this feature order is a deliberate design choice, because the order must remain stable for the serialised models to be applied correctly at inference time.

**Student segmentation with K-Means.** Segmentation is performed by a scikit-learn `Pipeline` (Pedregosa et al., 2011) that chains a `StandardScaler` to a `KMeans` estimator. Standardisation is essential here because the six features occupy very different numerical ranges, from a one-to-five mood scale to a 0-27 PHQ-9 score; without scaling, the distance computation at the heart of K-Means would be dominated by the widest-ranged feature (Géron, 2019). The clustering uses k = 4, with ten random initialisations and a fixed random seed of 42 to guarantee reproducibility (MacQueen, 1967; Lloyd, 1982). A central difficulty with K-Means is that the cluster identifiers it produces are arbitrary, so a raw cluster number carries no meaning. The design solves this with a labelling step that, after fitting, inverts the scaling to bring each cluster centroid back into original feature units and then ranks the centroids on a composite wellbeing index in which higher mood, sleep, and sentiment are favourable while higher PHQ-9 and GAD-7 scores are unfavourable. The clusters are then assigned ordered, meaningful labels, from **Thriving** at the top of the wellbeing ranking through **Coping** and **At-Risk** to **High-Need** at the bottom. On the synthetic training population of 600 students the four segments contain 198 At-Risk, 176 Coping, 119 Thriving, and 107 High-Need students, with a silhouette score of 0.209, a modest but positive indication of structure (Rousseeuw, 1987). Each label carries a tailored support strategy that the recommender later consults.

**Stress classification with Logistic Regression.** The supervised component is a multinomial Logistic-Regression classifier that maps the same six features to one of three stress levels: Low, Moderate, or High. Logistic Regression was chosen for its interpretability and its native ability to output calibrated class probabilities, which the Insights page surfaces directly so that students see not only a predicted level but the confidence behind it. The model is evaluated on a stratified 25% hold-out of 150 samples, on which it attains an accuracy of 0.900 and a macro-averaged F1 score of 0.889, with the strongest per-class performance on the High class (F1 = 0.945) and the weakest on the Moderate class (F1 = 0.819), the latter being the hardest because it lies between the two extremes. The trained model is persisted with joblib so that inference incurs no retraining cost.

**Hybrid transparent recommender.** Recommendations are produced by a hybrid scoring algorithm rather than an opaque model, in keeping with the transparency objective and the broader literature on recommender design (Ricci et al., 2015). Each of the nine coping strategies, which range from 4-7-8 Breathing and 5-4-3-2-1 Grounding to Thought Reframing drawn from cognitive behavioural therapy (Beck, 1979), is scored on three combined signals. A match between the strategy and the student's detected emotion contributes the largest weight (2.0); alignment with the student's segment preference contributes a moderate weight (1.0); and a stress-driven de-escalation boost of up to 1.5 raises calming strategies for students under high stress. The strategies are then ranked by total score and the top suggestions returned. Critically, every recommendation is accompanied by a rationale string explaining why it was chosen, and the context behind each suggestion is recorded in the recommendation log, so that no recommendation is ever presented as an unexplained verdict. This design makes the system's personalisation legible to the student, which is both an ethical requirement in a mental-health setting and a practical aid to evaluation.

Together these three algorithms convert a student's recent activity into a segment, a stress estimate, and an ordered set of explainable suggestions, and they do so deterministically and reproducibly. Their design reflects the project's guiding commitments throughout: personalisation that is genuinely tailored, intelligence that remains transparent, and an honest acknowledgement that the models are trained on synthetic data and are intended to demonstrate a methodology rather than to serve as a clinical instrument.
