# Chapter 6: Testing

## 6.1 Introduction

Testing is the discipline through which a software product is examined for the presence of defects and through which confidence is built that the system behaves as its specification requires. For a wellbeing assistant aimed at students, this activity carries a weight that goes beyond ordinary correctness: an error in the conversational interface is an inconvenience, but an error in the crisis-safety layer could allow a genuine cry for help to pass unnoticed. The testing effort for MindMate was therefore designed not only to confirm that features work, but to demonstrate that the parts of the system on which a vulnerable user might depend are robust, predictable, and conservative in the face of ambiguity.

This chapter documents the verification and validation activities carried out on MindMate. It explains the overall testing strategy, the structure of the automated test suite, and the way responsibilities were divided across unit, integration, and system levels. It then presents a consolidated table of representative test cases, describes the special treatment given to safety-critical behaviour, and reports the methodology used to validate the machine-learning models. The chapter closes with an honest account of the results obtained and of the defects and limitations that remain. Throughout, the distinction between verification, building the product correctly, and validation, building the correct product, is observed, following the conventional framing of software engineering practice (Sommerville, 2016; Pressman & Maxim, 2014).

In total, the project ships with a suite of **34 automated tests**, all of which pass. These are written using the **pytest** framework and are executed as part of the ordinary development workflow, so that any regression introduced by a later change is caught immediately rather than discovered in use.

## 6.2 Testing Strategy

The testing strategy for MindMate followed a layered, risk-driven model. Rather than allocating effort uniformly across the codebase, I directed the most rigorous testing toward the components whose failure carries the greatest consequence. The crisis-detection layer, the screening scorers, and the persistence layer therefore received the densest coverage, while purely presentational Streamlit code, which is difficult to test automatically and low in risk, was verified manually.

Several principles guided the strategy:

- **Automation first.** Every behaviour that can be expressed as a deterministic input-output relationship is captured as an automated pytest case, so that the full suite can be re-run cheaply after any change. This supports the regression-testing discipline recommended for evolving systems (Sommerville, 2016).
- **Isolation of side effects.** Database tests run against a temporary, throwaway SQLite file supplied by a pytest fixture, so they never touch production data and leave no residue. The chat-engine tests run with no API key configured, which forces the deterministic offline path and removes any dependency on a remote service.
- **Conservatism in safety tests.** Where the safety layer is concerned, the tests deliberately probe both the cases that must be flagged and the cases that must *not* be flagged, so that sensitivity is verified without ignoring the cost of over-flagging.
- **Reproducibility of model validation.** Machine-learning evaluation is anchored to fixed random seeds and a stratified hold-out split, so that the reported metrics can be regenerated exactly.

The test suite is organised by component into six files, each targeting one area of the system: `test_safety.py` (crisis detection), `test_sentiment.py` (sentiment and emotion analysis), `test_screening.py` (PHQ-9 and GAD-7 scoring), `test_database.py` (the SQLite persistence layer), `test_ml.py` (segmentation, the stress classifier, and the recommender), and `test_chat_engine.py` (the assistant's offline reply path and safety routing). This mapping of tests onto architectural layers mirrors the structure described in earlier chapters and makes it straightforward to locate the cause of any failure.

## 6.3 Unit Testing

Unit testing exercises the smallest independently meaningful units of behaviour in isolation. In MindMate these units are the pure functions and small classes that make up the assistant and ML packages, chosen because they are deterministic and free of external state.

The **safety** unit tests confirm that `safety.assess_risk` returns the correct risk level for a representative range of inputs. Explicit crisis phrases such as "I want to kill myself", "Sometimes I think about ending my life", and "there is no reason to live anymore" are required to yield the `crisis` level. Phrases expressing distress without imminent danger, "I feel so hopeless", "I just can't cope with all this", "I feel worthless lately", are required to yield the `elevated` level. These cases are parameterised so that each phrase is reported as an individual result.

The **sentiment** unit tests verify the VADER-based analyser and the custom emotion lexicon (Hutto & Gilbert, 2014). Clearly positive text is required to produce a positive compound score and a `positive` label; clearly negative text the reverse. Separate cases confirm that the emotion classifier recognises anxiety ("I am so anxious and worried about exams") and stress ("the workload and deadlines are too much"), and that empty input is treated as neutral in both label and emotion.

The **screening** unit tests check the scoring logic for the two standardised instruments. PHQ-9 with all-zero responses must total zero and be classified as "Minimal"; with all responses at the maximum it must total 27 and be classified as "Severe" (Kroenke et al., 2001). GAD-7 is checked at both extremes of its severity range (Spitzer et al., 2006). A guard test confirms that passing a response vector of the wrong length raises a `ValueError`, protecting the scorers from malformed input.

The **machine-learning** unit tests confirm that each model behaves sensibly on small, controlled fixtures. The segmenter, trained on a synthetic two-group matrix, must place a clearly thriving profile in a healthy segment and a clearly struggling profile in an at-risk segment. The stress classifier, trained on a generated signal, must label an unambiguously high-stress feature vector as "High". The recommender must return strategies in non-increasing order of score and, for an anxious high-stress context, must surface calming or grounding work such as 4-7-8 Breathing or 5-4-3-2-1 Grounding.

## 6.4 Integration Testing

Integration testing checks that units cooperate correctly once assembled, including the contracts between layers. Two areas of MindMate are inherently integrative and are tested at this level: the persistence layer and the chat engine.

The **database** tests treat the SQLite layer as an integrated whole, writing through the public API and reading back through it to confirm that data survives a complete round-trip. A user is created and re-fetched, with name and course verified. A mood log is written with mood, energy, sleep hours, a note, and a sentiment value, then read back to confirm the stored fields. Chat messages are written in sequence and retrieved to confirm that ordering is preserved. A user's support segment is updated and re-read, and a screening result is persisted and recovered. Because these tests exercise the schema, the SQL, and the row-to-dictionary mapping together, they catch faults that no single unit test would expose. Each test runs against an isolated temporary database supplied by a fixture, so the tests are independent of one another and of any real data.

The **chat-engine** tests are integrative because a single call to `chat_engine.generate_reply` orchestrates the safety layer, the personalisation context, and the reply generator. With no API key present, an ordinary message such as "I'm really stressed about my exams" must be answered through the offline engine, with the reply marked `offline`, a `none` risk level, and a substantive body of text. This confirms that the offline fallback engine and the safety screening are wired together correctly and that the system degrades gracefully when no external language model is available.

## 6.5 System Testing

System testing evaluates the complete application against its intended use, treating it as the user would encounter it. For MindMate this combined automated end-to-end checks of the most critical user journey with manual walkthroughs of the Streamlit interface.

The most important system-level behaviour is **crisis routing**: when a user types a message expressing suicidal intent anywhere in the chat, the system must abandon ordinary reply generation and respond from the safety layer. The chat-engine test "crisis message routes to safety" verifies exactly this path: the input "I want to kill myself" produces a reply whose source is `safety`, whose risk level is `crisis`, and whose text contains a helpline reference. The complementary "elevated reply appends resource nudge" test confirms that a distressed but non-crisis message ("I feel hopeless about everything") is classified as `elevated` and that the reply gently points the user toward resources. Together these establish that the safety policy holds at the level of the assembled system, not merely in the isolated detector.

Beyond the automated checks, each Streamlit page was exercised manually across a typical session: registering a profile, logging several days of mood data and observing the trend charts, completing PHQ-9 and GAD-7 self-checks and confirming that the reported severity bands matched the scored totals, reviewing the Insights page to confirm that the six features, the segment, and the stress probabilities were displayed consistently, and confirming that the Resources page presented the correct Indian and global helplines. This manual layer covers presentation and navigation behaviour that is impractical to automate but essential to validate before use.

## 6.6 Test Cases

The following table presents a representative selection of the executed test cases drawn from across the suite. Each derives from an automated pytest case described above, and every case passed when the suite was last executed. Inputs are shown in an abbreviated form for readability.

| Test Case ID | Description | Input | Expected Output | Actual Output | Status |
|---|---|---|---|---|---|
| TC-01 | Crisis detection on explicit intent | "I want to kill myself" | risk level = crisis | crisis | Pass |
| TC-02 | Non-crisis idiom not over-flagged | "I'm killing it in my exams!" | risk level ≠ crisis | none | Pass |
| TC-03 | Elevated distress detected | "I feel so hopeless" | risk level = elevated | elevated | Pass |
| TC-04 | Positive sentiment classification | "I feel great and really hopeful today" | compound > 0, label = positive | positive | Pass |
| TC-05 | Negative sentiment classification | "I am miserable and everything feels hopeless" | compound < 0, label = negative | negative | Pass |
| TC-06 | Emotion detection (anxiety) | "I am so anxious and worried about exams" | emotion = anxiety | anxiety | Pass |
| TC-07 | PHQ-9 minimal severity | nine responses all 0 | total = 0, severity = Minimal, no self-harm flag | 0 / Minimal / False | Pass |
| TC-08 | PHQ-9 severe + self-harm flag | nine responses all 3 | total = 27, severity = Severe, self-harm flag = True | 27 / Severe / True | Pass |
| TC-09 | GAD-7 severity thresholds | all 0, then all 3 | Minimal, then Severe | Minimal / Severe | Pass |
| TC-10 | Database user round-trip | create user "Test Student", MSc DS | stored name and course returned | matched | Pass |
| TC-11 | Mood-log round-trip | mood 4, sleep 7.5, note, sentiment 0.2 | one log with mood 4, sleep 7.5 | matched | Pass |
| TC-12 | Segmenter prediction | clearly thriving profile | healthy segment (Thriving/Coping) | Coping | Pass |
| TC-13 | Stress-classifier prediction | high-stress feature vector | stress level = High | High | Pass |
| TC-14 | Recommender ordering | emotion anxiety, segment At-Risk, stress High | scores non-increasing; calming/grounding present | ordered; 4-7-8 Breathing | Pass |
| TC-15 | Offline chat reply | "I'm really stressed about my exams" | source offline, risk none, substantive text | offline / none | Pass |
| TC-16 | Crisis routing through engine | "I want to kill myself" | source safety, risk crisis, helpline in text | safety / crisis | Pass |
| TC-17 | Self-harm flag isolated to item 9 | items 1-5 = 3, items 6-9 = 0 | self-harm flag = False | False | Pass |
| TC-18 | Malformed screening input rejected | five PHQ-9 responses | ValueError raised | ValueError | Pass |

## 6.7 Safety-Critical Testing

The safety layer is the part of MindMate that most clearly distinguishes it from an ordinary chatbot, and its testing was approached with a correspondingly different mindset. The governing observation is the **asymmetry of error cost**. A false positive, flagging a message as a crisis when the user is in no danger, produces, at worst, an unnecessary helpline message and a moment of friction. A false negative, failing to recognise genuine suicidal intent, could mean that a student who reached out received an ordinary, counselling-style reply instead of being directed to immediate help. Because the human cost of these two errors is so unequal, the detector is deliberately tuned to be conservative, preferring to over-flag rather than to miss, and the tests are written to enforce that bias.

Safety-critical testing in MindMate has three components:

- **Sensitivity to true crises.** A parameterised set of phrases expressing suicidal ideation or self-harm, direct statements of wanting to die, thoughts of ending one's life, intentions to hurt oneself, and statements that there is no reason to live, must each be classified as `crisis`. These verify that the detector responds to varied phrasings of the same underlying intent rather than to a single fixed wording.
- **Resistance to over-flagging.** The detector must not raise a crisis on everyday figurative language that happens to contain alarming words. "I'm killing it in my exams!" and "This deadline is killing me but I'll manage" both contain the word "killing" yet describe success and ordinary stress; both, along with neutral and empty input, are required to return a level other than `crisis`. This is the practical counterweight to conservatism: a detector that fired on every occurrence of a keyword would erode trust and quickly be ignored.
- **Correctness of the crisis response.** When a crisis is detected, the response itself is tested. The `crisis_response` output must contain a recognised helpline reference such as Tele-MANAS (14416), the Vandrevala Foundation, or AASRA, ensuring that the user is given an actionable route to support rather than an empty acknowledgement.

In addition to the conversational detector, the screening pathway contains its own safeguard. PHQ-9 item 9 asks specifically about thoughts of self-harm, and the scorer raises a dedicated `flags_self_harm` indicator whenever that item is answered with any non-zero response (Kroenke et al., 2001). The tests confirm both directions of this rule: the flag is raised when item 9 is non-zero and, critically, is *not* raised when other items are elevated but item 9 is zero. This guarantees that a high overall depression score never silently masks, nor falsely manufactures, a self-harm signal. Treating this behaviour as safety-critical, and testing it from both sides, reflects the principle that the components carrying the greatest risk deserve the most explicit verification (Pressman & Maxim, 2014).

## 6.8 Model Validation

The machine-learning components were validated separately from the functional tests, because their quality is measured statistically rather than as pass-or-fail behaviour. Validation followed a conventional hold-out methodology and was conducted on the synthetic datasets described in earlier chapters; all results below should therefore be read as a demonstration of the evaluation method on simulated data rather than as evidence about real students.

The **stress classifier**, a multinomial logistic-regression model, was evaluated on a stratified 25% hold-out test set of 150 samples drawn from the 600-student synthetic dataset. On this unseen set it achieved an **accuracy of 0.900** and a **macro-averaged F1 score of 0.889**. The macro average is reported alongside accuracy because it weights each stress class equally and so is not flattered by the larger High class. Per-class performance was strongest for the High class (precision 0.932, recall 0.958) and weakest for the Moderate class (precision 0.810, recall 0.829), which is expected, since the moderate band lies between the two extremes and is the most easily confused. The use of a held-out split, rather than evaluation on the training data, follows standard machine-learning practice for obtaining an unbiased estimate of generalisation performance (Géron, 2019; Pedregosa et al., 2011).

[[FIG: fig_confusion_matrix.png | Figure 6.1: Confusion matrix of the stress classifier on the test set]]

The confusion matrix in Figure 6.1 confirms this pattern: the off-diagonal errors are concentrated at the boundaries between adjacent stress levels rather than between Low and High, which is the most benign failure mode for a screening-style classifier.

The **K-Means student segmenter** was assessed using the silhouette coefficient, an internal measure of how compact and well separated the discovered clusters are (Rousseeuw, 1987). With four clusters the model produced a **silhouette score of 0.209**. This is a modest value, and it is reported candidly: it indicates that the four support segments overlap considerably rather than forming sharply distinct groups, which is realistic for continuous wellbeing data where students lie along a spectrum rather than in discrete types. The segments remain useful for tailoring tone and recommendations, but the silhouette score is a reminder not to treat the segment boundaries as crisp. The cluster labels themselves are assigned by ranking the cluster centroids on a composite wellbeing index, so that each label carries a meaningful, interpretable ordering rather than an arbitrary cluster number.

## 6.9 Results and Defects

The complete suite of 34 pytest tests executes successfully, with no failures and no errors, on Python 3.12. Coverage spans every layer identified as carrying meaningful risk: the safety detector and crisis response, sentiment and emotion analysis, PHQ-9 and GAD-7 scoring including the self-harm safeguard, the full database round-trip for users, mood logs, chat history, segments, and screening results, the three machine-learning models, and the assistant's offline and safety routing paths. The model-validation results, 0.900 accuracy and 0.889 macro-F1 for the stress classifier and a 0.209 silhouette for the segmenter, were reproduced from fixed seeds and are consistent with the figures reported elsewhere in this report.

Several defects and limitations were identified during testing and are recorded honestly here, since acknowledging them is part of responsible evaluation:

- **Rule-based crisis detection can miss indirect phrasing.** The detector recognises explicit and lexically clear expressions of risk, but it is fundamentally pattern-based. A student who signals distress obliquely, through metaphor, sarcasm, coded language, or a slow accumulation of despair across several messages, may not trigger a `crisis` classification. This is the most serious limitation of the safety layer, and it is the reason the system is positioned strictly as a source of support and signposting rather than as a clinical or emergency tool.
- **Conservatism produces occasional false positives.** The deliberate bias toward flagging means that some messages will be treated as elevated or crisis when the user is not in difficulty. This is an accepted trade-off given the cost asymmetry, but it is a real behaviour that an alert user may notice.
- **Sentiment and emotion analysis are lexicon-bound.** VADER and the custom emotion keywords perform well on direct, English-language statements of feeling but are weaker on irony, mixed emotion, and code-switched or colloquial text, so the detected emotion that drives recommendations is occasionally imperfect.
- **Validation rests on synthetic data.** The reported model metrics demonstrate that the evaluation pipeline is sound, but they are obtained from simulated students. They do not, and cannot, establish clinical effectiveness with real users; that would require a properly governed study with human participants.
- **Presentation logic is verified manually.** The Streamlit interface is checked by walkthrough rather than by automated UI tests, so visual or interaction regressions in the front end could escape the automated suite.

Taken together, the results give reasonable confidence that MindMate behaves as designed within the boundaries it claims for itself, while the recorded limitations make clear where that confidence stops. The most consequential of these, the inability of a rule-based detector to catch every indirect expression of risk, directly informs the future-work proposals discussed in the concluding chapter, where a more capable, learning-based safety layer is considered as a priority for any move beyond a synthetic, demonstrative setting (Sommerville, 2016).
