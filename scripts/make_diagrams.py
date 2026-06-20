"""Generate the structural diagrams for the project report using Graphviz.

Produces (in ``docs/figures/``):
    * System architecture
    * Entity-Relationship diagram
    * Data-Flow diagram (Level 0 and Level 1)
    * Use-case diagram
    * Class diagram
    * Sequence diagram (chat + safety flow)
    * Activity diagram (message handling with crisis branch)

Requires the Graphviz ``dot`` binary on PATH.
"""
from __future__ import annotations

from pathlib import Path

from graphviz import Digraph

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

FONT = "Helvetica"


def _render(g: Digraph, name: str) -> None:
    g.attr(fontname=FONT)
    out = g.render(filename=str(FIG_DIR / name), format="png", cleanup=True)
    print(f"  wrote {Path(out).relative_to(FIG_DIR.parent.parent)}")


def architecture():
    g = Digraph("architecture")
    g.attr(rankdir="TB", fontname=FONT, label="Figure: MindMate system architecture",
           labelloc="t")
    g.attr("node", fontname=FONT, style="filled", fontsize="11")

    with g.subgraph(name="cluster_ui") as c:
        c.attr(label="Presentation Layer (Streamlit)", style="rounded", color="#1f77b4")
        for n in ["Home", "Chat", "Mood Tracker", "Self-Check", "Insights", "Resources"]:
            c.node(n, shape="box", fillcolor="#dbe9f6")

    with g.subgraph(name="cluster_app") as c:
        c.attr(label="Application Layer", style="rounded", color="#2ca02c")
        c.node("Service", "Service Layer\n(service.py)", shape="box", fillcolor="#d8f0d8")
        c.node("Chat Engine", "Chat Engine\n(assistant)", shape="box", fillcolor="#d8f0d8")
        c.node("Safety", "Safety / Crisis\nScreening", shape="box", fillcolor="#ffe0e0")
        c.node("Features", "Feature\nEngineering", shape="box", fillcolor="#d8f0d8")

    with g.subgraph(name="cluster_ml") as c:
        c.attr(label="Machine-Learning Layer", style="rounded", color="#ff7f0e")
        c.node("Sentiment", "Sentiment\n(VADER)", shape="box", fillcolor="#ffead0")
        c.node("Segmenter", "Segmenter\n(K-Means)", shape="box", fillcolor="#ffead0")
        c.node("Stress", "Stress Classifier\n(Logistic Reg.)", shape="box", fillcolor="#ffead0")
        c.node("Recommender", "Recommender\n(hybrid)", shape="box", fillcolor="#ffead0")

    g.node("DB", "SQLite Database", shape="cylinder", fillcolor="#eeeeee")
    g.node("OpenAI", "OpenAI API\n(optional)", shape="component", fillcolor="#f0e0ff")

    for n in ["Home", "Chat", "Mood Tracker", "Self-Check", "Insights", "Resources"]:
        g.edge(n, "Service")
    g.edge("Chat", "Chat Engine")
    g.edge("Chat Engine", "Safety")
    g.edge("Chat Engine", "OpenAI", style="dashed")
    g.edge("Service", "Features")
    g.edge("Features", "Segmenter")
    g.edge("Features", "Stress")
    g.edge("Chat Engine", "Sentiment")
    g.edge("Service", "Recommender")
    g.edge("Segmenter", "Recommender")
    g.edge("Stress", "Recommender")
    for n in ["Service", "Chat Engine", "Features"]:
        g.edge(n, "DB")
    _render(g, "diagram_architecture")


def er_diagram():
    g = Digraph("er")
    g.attr(rankdir="LR", fontname=FONT, label="Figure: Entity-Relationship diagram",
           labelloc="t")
    g.attr("node", shape="record", fontname=FONT, fontsize="10", style="filled",
           fillcolor="#eef5ff")

    g.node("users", "{users|id (PK)\\lname\\lage\\lcourse\\lyear_of_study\\lsegment\\lcreated_at\\l}")
    g.node("mood", "{mood_logs|id (PK)\\luser_id (FK)\\lmood_score\\lenergy_score\\lsleep_hours\\lnote\\lsentiment\\llogged_at\\l}")
    g.node("chat", "{chat_messages|id (PK)\\luser_id (FK)\\lrole\\lcontent\\lsentiment\\lrisk_level\\lcreated_at\\l}")
    g.node("screen", "{screening_results|id (PK)\\luser_id (FK)\\linstrument\\ltotal_score\\lseverity\\ltaken_at\\l}")
    g.node("rec", "{recommendation_logs|id (PK)\\luser_id (FK)\\lstrategy\\lcategory\\lcontext\\lcreated_at\\l}")

    for child, lbl in [("mood", "1..N"), ("chat", "1..N"), ("screen", "1..N"), ("rec", "1..N")]:
        g.edge("users", child, label=lbl, arrowhead="crow", arrowtail="none", dir="both")
    _render(g, "diagram_er")


def dfd_level0():
    g = Digraph("dfd0")
    g.attr(rankdir="LR", fontname=FONT, label="Figure: Data-Flow Diagram (Level 0 / Context)",
           labelloc="t")
    g.attr("node", fontname=FONT, fontsize="11")
    g.node("student", "Student", shape="box", style="filled", fillcolor="#dbe9f6")
    g.node("system", "MindMate\nWellbeing System", shape="circle", style="filled",
           fillcolor="#d8f0d8", width="1.6")
    g.node("openai", "OpenAI API", shape="box", style="filled", fillcolor="#f0e0ff")
    g.edge("student", "system", label="messages, mood,\nself-checks")
    g.edge("system", "student", label="support, insights,\nrecommendations")
    g.edge("system", "openai", label="prompt", style="dashed")
    g.edge("openai", "system", label="reply", style="dashed")
    _render(g, "diagram_dfd_level0")


def dfd_level1():
    g = Digraph("dfd1")
    g.attr(rankdir="LR", fontname=FONT, label="Figure: Data-Flow Diagram (Level 1)",
           labelloc="t")
    g.attr("node", fontname=FONT, fontsize="10")
    g.node("student", "Student", shape="box", style="filled", fillcolor="#dbe9f6")
    for pid, lbl in [("p1", "1.0\nConverse"), ("p2", "2.0\nTrack Mood"),
                     ("p3", "3.0\nSelf-Check"), ("p4", "4.0\nPersonalise"),
                     ("p5", "5.0\nRecommend")]:
        g.node(pid, lbl, shape="circle", style="filled", fillcolor="#d8f0d8", width="1.1")
    g.node("d1", "D1 | mood_logs", shape="box", style="filled", fillcolor="#eeeeee")
    g.node("d2", "D2 | chat_messages", shape="box", style="filled", fillcolor="#eeeeee")
    g.node("d3", "D3 | screening_results", shape="box", style="filled", fillcolor="#eeeeee")

    g.edge("student", "p1", label="message")
    g.edge("p1", "d2")
    g.edge("p1", "student", label="reply")
    g.edge("student", "p2", label="mood entry")
    g.edge("p2", "d1")
    g.edge("student", "p3", label="answers")
    g.edge("p3", "d3")
    g.edge("d1", "p4")
    g.edge("d2", "p4")
    g.edge("d3", "p4")
    g.edge("p4", "p5", label="segment,\nstress level")
    g.edge("p5", "student", label="strategies")
    _render(g, "diagram_dfd_level1")


def use_case():
    g = Digraph("usecase")
    g.attr(rankdir="LR", fontname=FONT, label="Figure: Use-Case Diagram", labelloc="t")
    g.node("student", "Student", shape="box", style="filled", fillcolor="#dbe9f6")
    with g.subgraph(name="cluster_sys") as c:
        c.attr(label="MindMate", style="rounded")
        for uc in ["Chat with assistant", "Log mood", "Take self-check (PHQ-9/GAD-7)",
                   "View personalised insights", "Browse resources",
                   "Receive crisis support"]:
            c.node(uc, uc, shape="ellipse", style="filled", fillcolor="#fff3d6")
    for uc in ["Chat with assistant", "Log mood", "Take self-check (PHQ-9/GAD-7)",
               "View personalised insights", "Browse resources"]:
        g.edge("student", uc)
    g.edge("Chat with assistant", "Receive crisis support",
           label="«extend»", style="dashed")
    _render(g, "diagram_usecase")


def class_diagram():
    g = Digraph("classes")
    g.attr(rankdir="TB", fontname=FONT, label="Figure: Core Class Diagram", labelloc="t")
    g.attr("node", shape="record", fontname=FONT, fontsize="9", style="filled",
           fillcolor="#eef5ff")
    g.node("Settings", "{Settings|+openai_api_key\\l+openai_model\\l+db_path\\l|+llm_enabled()\\l}")
    g.node("StudentSegmenter", "{StudentSegmenter|+pipeline\\l+cluster_to_label\\l|+predict()\\l+strategy_for()\\l+save()/load()\\l}")
    g.node("StressClassifier", "{StressClassifier|+pipeline\\l+classes\\l|+predict()\\l+predict_proba()\\l}")
    g.node("SentimentResult", "{SentimentResult|+compound\\l+label\\l+emotion\\l|+as_dict()\\l}")
    g.node("RiskAssessment", "{RiskAssessment|+level\\l+matched\\l|+is_crisis()\\l}")
    g.node("AssistantReply", "{AssistantReply|+text\\l+risk_level\\l+sentiment\\l+source\\l}")
    g.node("Recommender", "{Recommender|+score_strategy()\\l+recommend()\\l}")
    g.edge("AssistantReply", "SentimentResult", arrowhead="diamond", dir="back")
    g.edge("AssistantReply", "RiskAssessment", arrowhead="diamond", dir="back")
    g.edge("Recommender", "StudentSegmenter", style="dashed", label="uses")
    g.edge("Recommender", "StressClassifier", style="dashed", label="uses")
    _render(g, "diagram_class")


def sequence_diagram():
    """A sequence diagram emulated with a left-to-right lifeline layout."""
    g = Digraph("sequence")
    g.attr(fontname=FONT, label="Figure: Sequence Diagram, sending a chat message",
           labelloc="t", rankdir="TB")
    g.attr("node", fontname=FONT, fontsize="10", shape="box", style="filled",
           fillcolor="#dbe9f6")
    steps = [
        ("Student", "ChatUI", "1: enter message"),
        ("ChatUI", "Safety", "2: assess_risk()"),
        ("Safety", "ChatUI", "3: risk level"),
        ("ChatUI", "Sentiment", "4: analyze()"),
        ("ChatUI", "Service", "5: profile_for_user()"),
        ("Service", "ML Models", "6: predict segment + stress"),
        ("ChatUI", "ChatEngine", "7: generate_reply()"),
        ("ChatEngine", "OpenAI/Offline", "8: build response"),
        ("ChatEngine", "Database", "9: persist messages"),
        ("ChatUI", "Student", "10: display reply"),
    ]
    for i, (src, dst, lbl) in enumerate(steps):
        g.node(f"{src}", src)
        g.node(f"{dst}", dst)
        g.edge(src, dst, label=lbl, fontsize="9")
    _render(g, "diagram_sequence")


def activity_diagram():
    g = Digraph("activity")
    g.attr(fontname=FONT, label="Figure: Activity Diagram, message handling with safety branch",
           labelloc="t", rankdir="TB")
    g.attr("node", fontname=FONT, fontsize="10")
    g.node("start", "", shape="circle", style="filled", fillcolor="black", width="0.2")
    g.node("recv", "Receive user message", shape="box", style="rounded,filled", fillcolor="#dbe9f6")
    g.node("risk", "Assess risk", shape="box", style="rounded,filled", fillcolor="#dbe9f6")
    g.node("dec", "Crisis detected?", shape="diamond", style="filled", fillcolor="#fff3d6")
    g.node("crisis", "Show crisis response\n+ helplines", shape="box", style="rounded,filled", fillcolor="#ffe0e0")
    g.node("personalise", "Compute segment\n+ stress level", shape="box", style="rounded,filled", fillcolor="#dbe9f6")
    g.node("gen", "Generate supportive reply\n(OpenAI / offline)", shape="box", style="rounded,filled", fillcolor="#dbe9f6")
    g.node("persist", "Persist + display", shape="box", style="rounded,filled", fillcolor="#dbe9f6")
    g.node("end", "", shape="doublecircle", style="filled", fillcolor="black", width="0.2")

    g.edge("start", "recv")
    g.edge("recv", "risk")
    g.edge("risk", "dec")
    g.edge("dec", "crisis", label="yes")
    g.edge("dec", "personalise", label="no")
    g.edge("crisis", "persist")
    g.edge("personalise", "gen")
    g.edge("gen", "persist")
    g.edge("persist", "end")
    _render(g, "diagram_activity")


def main():
    print("Generating diagrams...")
    architecture()
    er_diagram()
    dfd_level0()
    dfd_level1()
    use_case()
    class_diagram()
    sequence_diagram()
    activity_diagram()
    print(f"Done. Diagrams in {FIG_DIR}")


if __name__ == "__main__":
    main()
