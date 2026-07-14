"""
Research Agent (Agent 1) — single-file LangGraph implementation.

Objective: Establish the state-of-the-art AI benchmark for a target business domain.

Pipeline (LangGraph StateGraph):
    load_prompt -> ingest -> extract -> persist

Capabilities (mapped to the spec):
  * Framework Alignment ...... SCOR / eTOM / APQC reference codes
  * Multi-Modal Ingestion .... PDF / HTML / PPTX via Unstructured.io
  * Cross-Domain Synthesis ... `cross_domain_source` per finding
  * Abstraction Mapping ...... L1 through L4
  * Taxonomy Tagging ......... intervention_category + ai_subcategory
  * Destination .............. `market_intelligence` table (PostgreSQL)
  * Prompt source ............ latest Approved `prompt_candidates` row for
                               target_agent='Research_Agent', else system default

Requirements:
    pip install langgraph langchain-anthropic pydantic sqlalchemy psycopg2-binary \
                python-dotenv "unstructured[pdf,html,pptx]"

Env (.env):
    ANTHROPIC_API_KEY=...
    DATABASE_URL=postgresql+psycopg2://user:pass@localhost:5432/market_intel
    RESEARCH_AGENT_MODEL=claude-sonnet-4-20250514        # optional
    UNSTRUCTURED_API_KEY=                                 # optional (hosted API)

Usage:
    python research_agent_langgraph.py init-db
    python research_agent_langgraph.py run --domain "Supply Chain" report.pdf deck.pptx https://x.com/wp.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from enum import Enum
from typing import Optional, TypedDict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from sqlalchemy import (
    BigInteger, DateTime, ForeignKey, Integer, Numeric, String, Text,
    create_engine, func, select,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph

load_dotenv()

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
TARGET_AGENT = "Research_Agent"
MODEL = os.getenv("RESEARCH_AGENT_MODEL", "claude-sonnet-4-20250514")
TEMPERATURE = float(os.getenv("RESEARCH_AGENT_TEMPERATURE", "0.1"))
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://postgres:postgres@localhost:5432/market_intel")
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY", "")
UNSTRUCTURED_API_URL = os.getenv("UNSTRUCTURED_API_URL", "")

# --------------------------------------------------------------------------- #
# Vocabularies: Abstraction (L1-L4), Frameworks, Taxonomy
# --------------------------------------------------------------------------- #
class AbstractionLevel(str, Enum):
    L1 = "L1"  # domain / value chain
    L2 = "L2"  # process group
    L3 = "L3"  # process / activity
    L4 = "L4"  # task / implementation


class Framework(str, Enum):
    SCOR = "SCOR"
    ETOM = "eTOM"
    APQC = "APQC"
    OTHER = "Other"


FRAMEWORK_REFS = {
    Framework.SCOR: {"Plan": "sP", "Source": "sS", "Make": "sM", "Deliver": "sD", "Return": "sR", "Enable": "sE"},
    Framework.ETOM: {"Strategy, Infrastructure & Product": "1.1", "Operations": "1.2", "Enterprise Management": "1.3"},
    Framework.APQC: {"Manage Supply Chain": "4.0", "Manage Customer Service": "7.0"},
}

INTERVENTION_CATEGORIES = [
    "Demand Forecasting", "Inventory Optimization", "Predictive Maintenance",
    "Process Automation", "Quality Assurance", "Customer Experience",
    "Risk & Fraud Management", "Pricing & Revenue", "Knowledge Management", "Decision Support",
]
AI_SUBCATEGORIES = [
    "Classical ML", "Deep Learning", "Time-Series Forecasting", "Computer Vision",
    "NLP / LLM", "Reinforcement Learning", "Optimization / OR", "Recommender Systems",
    "Anomaly Detection", "Generative AI",
]

# --------------------------------------------------------------------------- #
# Structured output contract
# --------------------------------------------------------------------------- #
class ResearchFinding(BaseModel):
    business_domain: str
    title: Optional[str] = None
    finding: str = Field(..., description="SOTA benchmark or intervention.")
    framework: Framework = Framework.OTHER
    framework_reference: Optional[str] = Field(None, description="e.g. SCOR 'sS1.2' or eTOM '1.1.9'")
    abstraction_level: AbstractionLevel = AbstractionLevel.L2
    intervention_category: str
    ai_subcategory: str
    cross_domain_source: Optional[str] = Field(None, description="Domain the intervention was adapted from.")
    confidence: float = Field(0.5, ge=0.0, le=1.0)


class ResearchOutput(BaseModel):
    findings: list[ResearchFinding] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Database: prompt_candidates (source) + market_intelligence (destination)
# --------------------------------------------------------------------------- #
class Base(DeclarativeBase):
    pass


class PromptCandidate(Base):
    __tablename__ = "prompt_candidates"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    target_agent: Mapped[str] = mapped_column(Text, nullable=False)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, default="Draft", nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    approved_at: Mapped[Optional[dt.datetime]] = mapped_column(DateTime(timezone=True), nullable=True)


class MarketIntelligence(Base):
    __tablename__ = "market_intelligence"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    business_domain: Mapped[str] = mapped_column(Text, nullable=False)
    source_uri: Mapped[Optional[str]] = mapped_column(Text)
    source_type: Mapped[Optional[str]] = mapped_column(Text)
    title: Mapped[Optional[str]] = mapped_column(Text)
    finding: Mapped[str] = mapped_column(Text, nullable=False)
    framework: Mapped[Optional[str]] = mapped_column(Text)
    framework_reference: Mapped[Optional[str]] = mapped_column(Text)
    abstraction_level: Mapped[Optional[str]] = mapped_column(String(4))
    intervention_category: Mapped[Optional[str]] = mapped_column(Text)
    ai_subcategory: Mapped[Optional[str]] = mapped_column(Text)
    cross_domain_source: Mapped[Optional[str]] = mapped_column(Text)
    confidence: Mapped[Optional[float]] = mapped_column(Numeric(3, 2))
    raw_metadata: Mapped[Optional[dict]] = mapped_column(JSONB)
    prompt_candidate_id: Mapped[Optional[int]] = mapped_column(BigInteger, ForeignKey("prompt_candidates.id"))
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


engine = create_engine(DATABASE_URL, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


def init_db() -> None:
    Base.metadata.create_all(engine)


# --------------------------------------------------------------------------- #
# Multi-Modal Ingestion (Unstructured.io)
# --------------------------------------------------------------------------- #
_SOURCE_TYPES = {".pdf": "pdf", ".html": "html", ".htm": "html", ".ppt": "pptx", ".pptx": "pptx"}


def _source_type(uri: str) -> str:
    return _SOURCE_TYPES.get(os.path.splitext(uri.lower())[1], "unknown")


def ingest(uri: str) -> tuple[str, str]:
    """Return (text, source_type) for one PDF/HTML/PPTX source."""
    stype = _source_type(uri)
    if UNSTRUCTURED_API_KEY:
        from unstructured.partition.api import partition_via_api
        els = partition_via_api(filename=uri, api_key=UNSTRUCTURED_API_KEY, api_url=UNSTRUCTURED_API_URL or None)
    elif stype == "pdf":
        from unstructured.partition.pdf import partition_pdf
        els = partition_pdf(filename=uri)
    elif stype == "html":
        from unstructured.partition.html import partition_html
        els = partition_html(url=uri) if uri.startswith(("http://", "https://")) else partition_html(filename=uri)
    elif stype == "pptx":
        from unstructured.partition.pptx import partition_pptx
        els = partition_pptx(filename=uri)
    else:
        from unstructured.partition.auto import partition
        els = partition(filename=uri)
    text = "\n\n".join(getattr(e, "text", "") for e in els if getattr(e, "text", ""))
    return text, stype


# --------------------------------------------------------------------------- #
# Prompt sourcing
# --------------------------------------------------------------------------- #
SYSTEM_DEFAULT_PROMPT = """\
You are the Research Agent. Establish the state-of-the-art AI benchmark for the
target business domain from the supplied document text. For every finding:
 - Framework Alignment: map to a process framework (SCOR, eTOM, APQC) with a reference code.
 - Cross-Domain Synthesis: note any domain a proven intervention was adapted from.
 - Abstraction Mapping: assign an abstraction level L1 (domain) .. L4 (task).
 - Taxonomy Tagging: set intervention_category and ai_subcategory from the allowed lists.
Return calibrated, concrete findings only."""


def get_active_prompt() -> tuple[str, Optional[int]]:
    """Latest Approved prompt_candidates row for this agent, else the system default."""
    with SessionLocal() as s:
        stmt = (
            select(PromptCandidate)
            .where(PromptCandidate.target_agent == TARGET_AGENT, PromptCandidate.status == "Approved")
            .order_by(PromptCandidate.approved_at.desc().nullslast(),
                      PromptCandidate.version.desc(), PromptCandidate.id.desc())
            .limit(1)
        )
        row = s.execute(stmt).scalar_one_or_none()
    return (SYSTEM_DEFAULT_PROMPT, None) if row is None else (row.prompt_text, row.id)


# --------------------------------------------------------------------------- #
# LangGraph state + nodes
# --------------------------------------------------------------------------- #
class AgentState(TypedDict, total=False):
    business_domain: str
    uri: str
    persist: bool
    system_prompt: str
    prompt_candidate_id: Optional[int]
    source_type: str
    text: str
    output: ResearchOutput
    row_ids: list[int]


_llm = ChatAnthropic(model=MODEL, temperature=TEMPERATURE)
_extractor = _llm.with_structured_output(ResearchOutput)

_REFERENCE = (
    f"Abstraction levels: {', '.join(l.value for l in AbstractionLevel)} "
    "(L1 domain -> L4 task).\n"
    f"Frameworks: {', '.join(f.value for f in Framework)}.\n"
    f"intervention_category options: {', '.join(INTERVENTION_CATEGORIES)}.\n"
    f"ai_subcategory options: {', '.join(AI_SUBCATEGORIES)}."
)

_PROMPT = ChatPromptTemplate.from_messages([
    ("system", "{system_prompt}\n\nREFERENCE:\n{reference}"),
    ("human",
     "Target business domain: {domain}\nSource URI: {uri}\nSource type: {stype}\n\n"
     "Document content:\n\"\"\"\n{content}\n\"\"\"\n\n"
     "Extract the SOTA benchmarks/interventions as structured findings."),
])


def node_load_prompt(state: AgentState) -> AgentState:
    prompt, pid = get_active_prompt()
    return {"system_prompt": prompt, "prompt_candidate_id": pid}


def node_ingest(state: AgentState) -> AgentState:
    text, stype = ingest(state["uri"])
    return {"text": text, "source_type": stype}


def node_extract(state: AgentState) -> AgentState:
    if not state.get("text", "").strip():
        return {"output": ResearchOutput(findings=[])}
    messages = _PROMPT.format_messages(
        system_prompt=state["system_prompt"], reference=_REFERENCE,
        domain=state["business_domain"], uri=state["uri"],
        stype=state["source_type"], content=state["text"][:60_000],
    )
    output: ResearchOutput = _extractor.invoke(messages)
    for f in output.findings:
        if not f.business_domain:
            f.business_domain = state["business_domain"]
    return {"output": output}


def node_persist(state: AgentState) -> AgentState:
    if not state.get("persist", True):
        return {"row_ids": []}
    ids: list[int] = []
    with SessionLocal() as s:
        for f in state["output"].findings:
            row = MarketIntelligence(
                business_domain=f.business_domain, source_uri=state["uri"],
                source_type=state["source_type"], title=f.title, finding=f.finding,
                framework=f.framework.value, framework_reference=f.framework_reference,
                abstraction_level=f.abstraction_level.value,
                intervention_category=f.intervention_category, ai_subcategory=f.ai_subcategory,
                cross_domain_source=f.cross_domain_source, confidence=f.confidence,
                raw_metadata={"title": f.title} if f.title else None,
                prompt_candidate_id=state.get("prompt_candidate_id"),
            )
            s.add(row); s.flush(); ids.append(row.id)
        s.commit()
    return {"row_ids": ids}


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("load_prompt", node_load_prompt)
    g.add_node("ingest", node_ingest)
    g.add_node("extract", node_extract)
    g.add_node("persist", node_persist)
    g.add_edge(START, "load_prompt")
    g.add_edge("load_prompt", "ingest")
    g.add_edge("ingest", "extract")
    g.add_edge("extract", "persist")
    g.add_edge("persist", END)
    return g.compile()


GRAPH = build_graph()


def run(uris: list[str], business_domain: str, persist: bool = True) -> list[dict]:
    results = []
    for uri in uris:
        final = GRAPH.invoke({"uri": uri, "business_domain": business_domain, "persist": persist})
        results.append({
            "uri": uri,
            "prompt": "system default" if final.get("prompt_candidate_id") is None
                      else f"candidate #{final['prompt_candidate_id']}",
            "findings": len(final.get("output", ResearchOutput()).findings),
            "row_ids": final.get("row_ids", []),
        })
    return results


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _seed_prompt(text: str) -> None:
    with SessionLocal() as s:
        pc = PromptCandidate(target_agent=TARGET_AGENT, prompt_text=text, status="Approved",
                             approved_at=dt.datetime.now(dt.timezone.utc))
        s.add(pc); s.commit()
        print(f"Seeded Approved prompt id={pc.id}.")


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="research_agent_langgraph")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    sp = sub.add_parser("seed-prompt"); sp.add_argument("text")
    rp = sub.add_parser("run")
    rp.add_argument("--domain", required=True)
    rp.add_argument("--dry-run", action="store_true")
    rp.add_argument("sources", nargs="+")
    args = p.parse_args(argv)

    if args.cmd == "init-db":
        init_db(); print("Schema created / verified."); return 0
    if args.cmd == "seed-prompt":
        _seed_prompt(args.text); return 0
    if args.cmd == "run":
        for r in run(args.sources, args.domain, persist=not args.dry_run):
            print(f"[{r['prompt']}] {r['uri']}: {r['findings']} findings -> rows {r['row_ids']}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
