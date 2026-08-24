---
title: ResumeHQ Scorer
emoji: 🧮
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10.13"
app_file: app.py
pinned: false
---

# ResumeHQ Scorer (Gradio + ZeroGPU)

Local ResumeHQ (SBERT/torch) ATS + HR scoring, exposed as a Gradio UI **and**
as an API. ZeroGPU requires the Gradio SDK, so the scoring functions are wrapped
with `@spaces.GPU`.

## Enable ZeroGPU
Space → **Settings → Hardware → ZeroGPU (Nvidia)**. Free accounts in good
standing (verified email, > 30 days old) may host up to 2 ZeroGPU Spaces.

## API (called by the main app)
Every button is also an API endpoint. From the main app:

```python
from gradio_client import Client
c = Client("https://<username>-resumehq-scorer.hf.space")   # or "<username>/resumehq-scorer"
ats = c.predict(resume_text, jd_text, api_name="/score_ats")   # -> dict
hr  = c.predict(resume_text, jd_text, api_name="/score_hr")
exp = c.predict(resume_text, jd_text, api_name="/explain")
```