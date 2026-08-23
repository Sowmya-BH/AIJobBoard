"""Interactive CLI demonstrating the full HITL loop.

    python run.py --resume path/to/resume.txt --domain "Data Science"

Works with or without a Gemini key (falls back to deterministic text).
"""
import argparse
import uuid
from langgraph.types import Command
from agent.graph import build_graph


def read_resume(path):
    if not path:
        return ("Data scientist with 2 years experience. Skills: Python, "
                "PyTorch, SQL, Pandas, scikit-learn, NLP, Docker.")
    if path.lower().endswith((".txt", ".md")):
        return open(path, encoding="utf-8", errors="ignore").read()
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
    if path.lower().endswith(".docx"):
        import docx
        return "\n".join(p.text for p in docx.Document(path).paragraphs)
    return open(path, encoding="utf-8", errors="ignore").read()


def show_matches(matches):
    print("\n=== TOP MATCHES ===")
    for i, m in enumerate(matches[:12]):
        print(f"[{i}] {m['score']:>5}  {m['title'][:46]:46} | {m['company'][:20]:20} | {m['location'][:18]}")
        print(f"        matched: {', '.join(m['matched'][:6])}")
    print()


def show_ats(a):
    print(f"\n=== ATS: {a['job_title']} @ {a['company']} ===")
    print(f"Score: {a['ats_score']}  (confidence: {a['confidence']})")
    print("\nWhat matched:")
    [print("  +", x) for x in a["matched"]]
    print("\nWhat is missing:")
    [print("  -", x) for x in a["missing"]]
    print("\nActionable advice:")
    [print("  *", x) for x in a["advice"]]
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume")
    ap.add_argument("--domain", default=None, help="Data Science | Web Development")
    ap.add_argument("--location", default=None)
    args = ap.parse_args()

    graph = build_graph()
    cfg = {"configurable": {"thread_id": str(uuid.uuid4())}}

    state = {"resume_text": read_resume(args.resume),
             "domain": args.domain, "location": args.location}

    # run until the first interrupt (job selection)
    result = graph.invoke(state, cfg)
    show_matches(result["__interrupt__"][0].value["matches"])

    idx = input("Pick a match index [0]: ").strip() or "0"
    matches = result["__interrupt__"][0].value["matches"]
    job_id = matches[int(idx)]["id"]
    result = graph.invoke(Command(resume=job_id), cfg)

    # HITL review loop
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        show_ats(payload["ats"])
        print("Actions: add_info | ask | cover_letter | tailored_resume | interview_questions | done")
        action = input("> action: ").strip() or "done"
        cmd = {"action": action}
        if action in ("add_info", "ask"):
            cmd["text"] = input("  text: ").strip()
        result = graph.invoke(Command(resume=cmd), cfg)

        if result.get("answer"):
            print("\nANSWER:", result["answer"], "\n")
        arts = result.get("artifacts", {})
        for k, v in arts.items():
            print(f"\n--- {k.upper()} ---\n{v}\n")

    print("Done.")


if __name__ == "__main__":
    main()
