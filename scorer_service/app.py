"""Self-hosted ResumeHQ scorer sidecar (+ visual API at /).

Runs ResumeHQ's LOCAL scoring engines (torch/SBERT) in its own process and
exposes them over HTTP. Local scoring => no cloud 5-score cap, no API key.

    GET  /              visual tester (HTML form -> calls the POST endpoints)
    GET  /info          JSON endpoint descriptor
    GET  /health        liveness
    POST /score/ats     {resume_text, jd_text}
    POST /score/hr      {resume_text, jd_text}
    POST /explain       {resume_text, jd_text}
    POST /cover-letter  {resume_text, jd_text, company_name?, job_title?}

Run (own container, full deps):
    pip install resumehq fastapi "uvicorn[standard]"
    uvicorn app:app --host 0.0.0.0 --port 8100
"""
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import ats_scorer          # ResumeHQ local engines — flat module names
import hr_scorer

app = FastAPI(title="ResumeHQ Scorer Sidecar")


class ScoreReq(BaseModel):
    resume_text: str
    jd_text: str


class CoverReq(BaseModel):
    resume_text: str
    jd_text: str
    company_name: str = ""
    job_title: str = ""


@app.on_event("startup")
def _warm():
    try:
        ats_scorer.get_sbert_model()   # load SBERT once so first request is fast
    except Exception:
        pass


@app.get("/info")
def info():
    return {
        "service": "ResumeHQ scorer sidecar",
        "engine": "local", "quota": "unlimited",
        "endpoints": ["POST /score/ats", "POST /score/hr", "POST /explain",
                      "POST /cover-letter", "GET /health", "GET / (visual)"],
    }


@app.get("/health")
def health():
    return {"ok": True, "engine": "local", "quota": "unlimited"}


@app.post("/score/ats")
def score_ats(req: ScoreReq):
    result = ats_scorer.calculate_ats_score(req.resume_text, req.jd_text)
    rating, likelihood, _ = ats_scorer.get_likelihood_rating(result["total_score"])
    result["rating"] = rating
    result["likelihood"] = likelihood
    return result


@app.post("/score/hr")
def score_hr(req: ScoreReq):
    res = hr_scorer.calculate_hr_score_from_text(req.resume_text, req.jd_text)
    return hr_scorer.result_to_dict(res)


@app.post("/explain")
def explain(req: ScoreReq):
    ats = ats_scorer.calculate_ats_score(req.resume_text, req.jd_text)
    missing = ats.get("missing_keywords", [])
    return {
        "current_score": round(ats.get("total_score", 0), 1),
        "explanation": {
            "top_missing_keywords": missing[:10],
            "suggestion": ("Add these missing keywords to your Core Competencies "
                           "or bullet points where you can honestly claim them."),
        },
    }


@app.post("/cover-letter")
def cover_letter(req: CoverReq):
    try:
        from llm_scorer import ANTHROPIC_AVAILABLE
        from llm_scorer import generate_cover_letter as _gen
    except Exception as e:
        return {"error": "llm_unavailable", "message": str(e)}
    if not ANTHROPIC_AVAILABLE:
        return {"error": "no_api_key",
                "message": "Set ANTHROPIC_API_KEY in the scorer container for cover letters."}
    return _gen(req.resume_text, req.jd_text,
                company_name=req.company_name, job_title=req.job_title)


# ---------------------------------------------------------------------------
# Visual API: a browser tester served from the sidecar itself.
# ---------------------------------------------------------------------------
_UI = """<!doctype html><html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>ResumeHQ Scorer</title><style>
:root{--bg:#0f1115;--card:#171a21;--line:#262b36;--fg:#e6e8ec;--mut:#9aa3b2;--acc:#5b8cff;--good:#3fb06a;--bad:#e2554b;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.55 system-ui,Segoe UI,Roboto,sans-serif}
.wrap{max-width:860px;margin:0 auto;padding:24px 18px 80px}h1{font-size:19px;margin:0 0 2px}
.sub{color:var(--mut);font-size:13px;margin-bottom:18px}.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px}
label{display:block;font-size:12px;color:var(--mut);margin:0 0 5px}textarea{width:100%;min-height:120px;background:#0d0f14;color:var(--fg);border:1px solid var(--line);border-radius:8px;padding:10px;font:inherit;resize:vertical}
.row{display:flex;gap:12px;flex-wrap:wrap}.row>div{flex:1;min-width:240px}
.actions{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
button{background:var(--acc);color:#fff;border:0;border-radius:8px;padding:9px 15px;font:inherit;font-weight:500;cursor:pointer}
button.ghost{background:transparent;border:1px solid var(--line);color:var(--fg)}button:disabled{opacity:.5}
.big{font-size:34px;font-weight:700}.unit{font-size:15px;color:var(--mut)}
.pill{display:inline-block;font-size:12px;padding:3px 9px;border-radius:20px;margin:3px 5px 0 0;border:1px solid var(--line)}
.pill.g{border-color:#2c5c3f;color:#8fe0ac}.pill.b{border-color:#5c2c2c;color:#f0a19b}
.out{white-space:pre-wrap;background:#0d0f14;border:1px solid var(--line);border-radius:8px;padding:13px;margin-top:12px;font-size:13px}
h3{font-size:13px;margin:14px 0 4px;color:var(--mut)}.tag{color:var(--mut);font-size:13px}.err{color:var(--bad);font-size:13px}
</style></head><body><div class="wrap">
<h1>ResumeHQ Scorer — visual API</h1>
<div class="sub">Local scoring · unlimited · calls this service's own POST endpoints.</div>
<div class="card">
  <div class="row">
    <div><label>Resume text</label><textarea id="resume">Data scientist, 2 yrs. Skills: Python, SQL, Pandas, scikit-learn.</textarea></div>
    <div><label>Job description text</label><textarea id="jd">Data Scientist. Required skills: Python, SQL, AWS, PyTorch, Docker, NLP.</textarea></div>
  </div>
  <div class="row" style="margin-top:10px"><div><label>Company (cover letter)</label><textarea id="company" style="min-height:38px">Acme</textarea></div>
    <div><label>Job title (cover letter)</label><textarea id="title" style="min-height:38px">Data Scientist</textarea></div></div>
  <div class="actions">
    <button onclick="run('/score/ats','ats')">Score ATS</button>
    <button class="ghost" onclick="run('/score/hr','hr')">Score HR</button>
    <button class="ghost" onclick="run('/explain','explain')">Explain</button>
    <button class="ghost" onclick="run('/cover-letter','cover')">Cover letter</button>
  </div>
</div>
<div class="card" id="result" style="display:none"></div>
</div>
<script>
const $=id=>document.getElementById(id);
function body(kind){const b={resume_text:$("resume").value,jd_text:$("jd").value};
  if(kind==="cover"){b.company_name=$("company").value;b.job_title=$("title").value;}return b;}
async function run(url,kind){
  const R=$("result");R.style.display="block";R.innerHTML='<span class="tag">scoring… (first call loads the model)</span>';
  try{const r=await fetch(url,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body(kind))});
    const j=await r.json();render(kind,j);}
  catch(e){R.innerHTML='<span class="err">Request failed: '+e+'</span>';}
}
function pills(arr,cls){return (arr||[]).map(k=>'<span class="pill '+cls+'">'+esc(k)+'</span>').join("")||'<span class="tag">—</span>';}
function esc(s){return String(s).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));}
function render(kind,j){const R=$("result");
  if(j.error){R.innerHTML='<div class="err">'+esc(j.error)+': '+esc(j.message||"")+'</div>';return;}
  if(kind==="ats"){R.innerHTML='<div class="big">'+(j.total_score??"?")+'<span class="unit">/100</span></div>'+
    '<div class="tag">'+esc(j.rating||"")+' · '+esc(j.likelihood||"")+'</div>'+
    '<h3>Matched keywords</h3>'+pills(j.matched_keywords,"g")+
    '<h3>Missing keywords</h3>'+pills(j.missing_keywords,"b");}
  else if(kind==="hr"){R.innerHTML='<div class="big">'+(j.overall_score??"?")+'<span class="unit">/100</span></div>'+
    '<div class="tag">'+esc(j.recommendation||"")+'</div>'+
    '<h3>Strengths</h3>'+pills(j.strengths,"g")+'<h3>Concerns</h3>'+pills(j.concerns,"b")+
    '<h3>Suggested interview questions</h3><div class="out">'+esc((j.suggested_questions||[]).map((q,i)=>(i+1)+". "+q).join("\\n")||"—")+'</div>';}
  else if(kind==="explain"){const e=j.explanation||{};R.innerHTML='<div class="big">'+(j.current_score??"?")+'<span class="unit">/100</span></div>'+
    '<h3>Top missing keywords</h3>'+pills(e.top_missing_keywords,"b")+'<div class="out">'+esc(e.suggestion||"")+'</div>';}
  else{R.innerHTML='<div class="out">'+esc(j.full_text||JSON.stringify(j,null,2))+'</div>';}
}
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def ui():
    return _UI
