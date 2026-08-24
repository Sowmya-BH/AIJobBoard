"""ResumeHQ scorer on Hugging Face Spaces — Gradio SDK, ZeroGPU-compatible.

Exposes ATS / HR / Explain as a Gradio UI and as API endpoints (api_name=...),
callable from the main app via gradio_client. Scoring functions are wrapped with
@spaces.GPU so ZeroGPU allocates a GPU for the call (no-op off ZeroGPU).
"""
import gradio as gr

# ZeroGPU decorator — falls back to a no-op when the `spaces` package is absent
# (e.g. local/CPU runs), supporting both @GPU and @GPU(duration=...).
try:
    import spaces
    GPU = spaces.GPU
except Exception:
    def GPU(*args, **kwargs):
        if args and callable(args[0]):
            return args[0]
        def deco(fn):
            return fn
        return deco

# No Docker build step here, so fetch NLTK data + warm the model at startup.
try:
    import nltk
    for _p in ("wordnet", "omw-1.4", "punkt", "stopwords"):
        try:
            nltk.download(_p, quiet=True)
        except Exception:
            pass
except Exception:
    pass

import ats_scorer          # ResumeHQ local engines (flat module names)
import hr_scorer

# Load SBERT at module level (ZeroGPU allows CUDA placement outside GPU fns).
try:
    ats_scorer.get_sbert_model()
except Exception:
    pass


@GPU(duration=120)
def score_ats(resume_text, jd_text):
    r = ats_scorer.calculate_ats_score(resume_text or "", jd_text or "")
    rating, likelihood, _ = ats_scorer.get_likelihood_rating(r["total_score"])
    r["rating"], r["likelihood"] = rating, likelihood
    return r


@GPU(duration=120)
def score_hr(resume_text, jd_text):
    res = hr_scorer.calculate_hr_score_from_text(resume_text or "", jd_text or "")
    return hr_scorer.result_to_dict(res)


@GPU(duration=120)
def explain(resume_text, jd_text):
    ats = ats_scorer.calculate_ats_score(resume_text or "", jd_text or "")
    missing = ats.get("missing_keywords", [])
    return {
        "current_score": round(ats.get("total_score", 0), 1),
        "explanation": {
            "top_missing_keywords": missing[:10],
            "suggestion": ("Add these missing keywords to your Core Competencies "
                           "or bullets where you can honestly claim them."),
        },
    }


def health():
    return {"ok": True, "engine": "local"}


DEMO_RESUME = "Data scientist, 2 yrs. Skills: Python, SQL, Pandas, scikit-learn."
DEMO_JD = "Data Scientist. Required: Python, SQL, AWS, PyTorch, Docker, NLP."

with gr.Blocks(title="ResumeHQ Scorer") as demo:
    gr.Markdown("# ResumeHQ Scorer\nLocal SBERT scoring · ZeroGPU · UI + API.")
    with gr.Row():
        resume = gr.Textbox(label="Resume text", lines=8, value=DEMO_RESUME)
        jd = gr.Textbox(label="Job description text", lines=8, value=DEMO_JD)
    with gr.Row():
        b_ats = gr.Button("Score ATS", variant="primary")
        b_hr = gr.Button("Score HR")
        b_exp = gr.Button("Explain")
    out = gr.JSON(label="Result")

    b_ats.click(score_ats, [resume, jd], out, api_name="score_ats")
    b_hr.click(score_hr, [resume, jd], out, api_name="score_hr")
    b_exp.click(explain, [resume, jd], out, api_name="explain")
    # API-only liveness endpoint
    gr.Button("health", visible=False).click(lambda: health(), None, out,
                                             api_name="health")

if __name__ == "__main__":
    demo.launch()