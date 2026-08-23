"""Shared LangGraph state. One dict flows through every node."""
from typing import TypedDict, Optional
from typing_extensions import NotRequired


class Profile(TypedDict, total=False):
    raw_text: str            # extracted resume text
    skills: list             # normalised skill tokens
    exp_years: Optional[int]
    seniority: str           # e.g. "junior" | "mid" | "senior"
    summary: str
    extra_info: list         # user-added facts from the HITL loop


class JobMatch(TypedDict):
    id: str
    title: str
    company: str
    location: str
    domain: str
    score: float
    matched: list
    missing: list


class ATSReport(TypedDict, total=False):
    job_id: str
    ats_score: float
    matched: list            # {"what": ...} plain strings
    missing: list
    advice: list
    confidence: str          # high | medium | low


class GraphState(TypedDict, total=False):
    # inputs
    resume_path: NotRequired[str]
    resume_text: NotRequired[str]
    domain: NotRequired[str]
    location: NotRequired[str]
    llm_creds: NotRequired[dict]   # user's BYO key {provider,model,api_key,base_url}

    # produced in parallel branches
    profile: Profile
    candidate_pool_ids: NotRequired[list]   # from scout branch
    matches: list            # list[JobMatch]
    jd_text: NotRequired[str]               # grounded JD built from the selected job
    how_to_add: NotRequired[str]            # #3: how to fold added info into the resume

    # HITL selection + loop
    selected_job_id: NotRequired[str]
    ats: ATSReport
    next_action: NotRequired[str]   # add_info | ask | cover_letter | tailored_resume | interview_questions | done
    user_edit: NotRequired[str]
    user_question: NotRequired[str]

    # outputs of the extras node
    answer: NotRequired[str]
    artifacts: dict          # {"cover_letter": ..., "tailored_resume": ..., "interview_questions": ...}
# """Shared LangGraph state. One dict flows through every node."""
# from typing import TypedDict, Optional
# from typing_extensions import NotRequired


# class Profile(TypedDict, total=False):
#     raw_text: str            # extracted resume text
#     skills: list             # normalised skill tokens
#     exp_years: Optional[int]
#     seniority: str           # e.g. "junior" | "mid" | "senior"
#     summary: str
#     extra_info: list         # user-added facts from the HITL loop


# class JobMatch(TypedDict):
#     id: str
#     title: str
#     company: str
#     location: str
#     domain: str
#     score: float
#     matched: list
#     missing: list


# class ATSReport(TypedDict, total=False):
#     job_id: str
#     ats_score: float
#     matched: list            # {"what": ...} plain strings
#     missing: list
#     advice: list
#     confidence: str          # high | medium | low


# class GraphState(TypedDict, total=False):
#     # inputs
#     resume_path: NotRequired[str]
#     resume_text: NotRequired[str]
#     domain: NotRequired[str]
#     location: NotRequired[str]

#     # produced in parallel branches
#     profile: Profile
#     candidate_pool_ids: NotRequired[list]   # from scout branch
#     matches: list            # list[JobMatch]
#     jd_text: NotRequired[str]               # grounded JD built from the selected job
#     how_to_add: NotRequired[str]            # #3: how to fold added info into the resume

#     # HITL selection + loop
#     selected_job_id: NotRequired[str]
#     ats: ATSReport
#     next_action: NotRequired[str]   # add_info | ask | cover_letter | tailored_resume | interview_questions | done
#     user_edit: NotRequired[str]
#     user_question: NotRequired[str]

#     # outputs of the extras node
#     answer: NotRequired[str]
#     artifacts: dict          # {"cover_letter": ..., "tailored_resume": ..., "interview_questions": ...}
