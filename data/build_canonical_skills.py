"""OFFLINE: add a `canonical_skills` column to the jobs DB using spaCy + Gensim.

Runs on your machine, NOT on Render (spaCy/Gensim are heavy and must never be
imported by the 512 MB app). It:
  1. trains a Gensim Phrases model on the whole corpus so multi-word skills like
     "machine learning" / "natural language processing" become single tokens
     (this is what stops "washing machine" from matching "machine learning"),
  2. runs spaCy to lemmatize and keep only NOUN/PROPN tokens (+ a tech whitelist),
     so "I am coding in Python" -> python (the verb "coding" is dropped),
  3. writes the cleaned, de-duplicated skill set to jobs.canonical_skills.

Setup + run:
    pip install spacy gensim
    python -m spacy download en_core_web_sm
    python data/build_canonical_skills.py --db data/app.db

Then (optional) point matching at the cleaner column — see the note printed at
the end. Re-run any time the job data changes.
"""
import argparse
import os
import re
import sys

# Tokens spaCy may mis-tag (short tools/acronyms) but that we always want to keep.
TECH_WHITELIST = {
    "python", "sql", "r", "java", "c++", "c#", ".net", "js", "ts", "go", "rust",
    "aws", "gcp", "azure", "docker", "kubernetes", "k8s", "spark", "hadoop",
    "pandas", "numpy", "pytorch", "tensorflow", "keras", "sklearn", "scikit-learn",
    "nlp", "llm", "rag", "etl", "api", "rest", "graphql", "html", "css", "react",
    "node", "node.js", "django", "flask", "fastapi", "airflow", "tableau", "powerbi",
    "excel", "git", "ci", "cd", "linux", "bash", "mongodb", "postgresql", "mysql",
    "redis", "kafka", "snowflake", "databricks", "bigquery", "sagemaker",
}
_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,30}")
_MAX_SKILLS = 40


def tokenize(text: str):
    out = []
    for t in _TOKEN_RE.findall((text or "").lower()):
        t = t.strip(".-")
        if len(t) > 1:
            out.append(t)
    return out


def job_text(r):
    return f"{r['title'] or ''}. {r['skills'] or ''}. {(r['description'] or '')[:2000]}"


def load_spacy():
    try:
        import spacy
    except ImportError:
        sys.exit("pip install spacy  (and: python -m spacy download en_core_web_sm)")
    try:
        # tagger+lemmatizer only — parser/ner not needed, much faster
        return spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except OSError:
        sys.exit("Missing model. Run: python -m spacy download en_core_web_sm")


def noun_lemmas(doc):
    keep = set()
    for tok in doc:
        if tok.is_stop or tok.is_punct or tok.is_space or len(tok.text) < 2:
            continue
        if tok.pos_ in ("NOUN", "PROPN") or tok.lower_ in TECH_WHITELIST:
            keep.add(tok.lemma_.lower().strip(".-"))
    return keep


def main(limit, min_count, threshold):
    from pymongo import MongoClient
    uri = os.environ.get("MONGODB_URI")
    if not uri:
        sys.exit("Set MONGODB_URI first.")
    coll = MongoClient(uri)[os.environ.get("MONGODB_DB", "jobscout")].jobs

    cur = coll.find({}, {"title": 1, "skills": 1, "description": 1})
    if limit:
        cur = cur.limit(limit)
    rows = [{"id": d["_id"], "title": d.get("title", ""),
             "skills": ", ".join(d.get("skills", []) or []),
             "description": d.get("description", "")} for d in cur]
    print(f"{len(rows)} jobs to process")

    # 1) learn multi-word phrases across the corpus
    from gensim.models.phrases import Phrases, ENGLISH_CONNECTOR_WORDS
    print("training phrase model...")
    sentences = [tokenize(job_text(r)) for r in rows]
    bigram = Phrases(sentences, min_count=min_count, threshold=threshold,
                     connector_words=ENGLISH_CONNECTOR_WORDS)
    trigram = Phrases(bigram[sentences], min_count=min_count, threshold=threshold,
                      connector_words=ENGLISH_CONNECTOR_WORDS)

    # 2) spaCy lemmatize/POS in a streaming pipe, combine with phrases
    from pymongo import UpdateOne
    nlp = load_spacy()
    texts = [job_text(r) for r in rows]
    ops = []
    print("extracting canonical skills (spaCy pipe)...")
    for i, doc in enumerate(nlp.pipe(texts, batch_size=200)):
        allowed = noun_lemmas(doc)
        phrased = trigram[bigram[sentences[i]]]
        canon = set()
        for tok in phrased:
            if "_" in tok:
                canon.add(tok.replace("_", " "))
            elif tok in allowed:
                canon.add(tok)
        for s in re.split(r"[,;|]", (rows[i]["skills"] or "")):
            s = s.strip().lower()
            if s:
                canon.add(s)
        canon = sorted(c for c in canon if 1 < len(c) < 40)[:_MAX_SKILLS]
        ops.append(UpdateOne({"_id": rows[i]["id"]},
                             {"$set": {"canonical_skills": canon,
                                       "canonical_skills_lc": [c.lower() for c in canon]}}))
        if (i + 1) % 5000 == 0:
            coll.bulk_write(ops, ordered=False); ops = []
            print(f"  {i + 1}/{len(rows)}")
    if ops:
        coll.bulk_write(ops, ordered=False)
    print(f"done — wrote canonical_skills for {len(rows)} jobs")
    print("\nTo USE it: have agent/db read canonical_skills_lc for matching when present.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=5,
                    help="min corpus frequency for a phrase")
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="higher = fewer, stronger phrases")
    a = ap.parse_args()
    main(a.limit, a.min_count, a.threshold)