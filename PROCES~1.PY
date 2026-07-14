"""
Agent 2A: Process Modeler Agent  (single-file LangGraph implementation)
=======================================================================

Objective : Build the intended ("as-documented") process model from internal
            documentation (SOPs, manuals, Visio diagrams).

Behaviour (per design spec):
  1. Prompt source        : latest Approved row in `prompt_registry` for
                            target_agent = 'Process_Modeler_Agent'.
  2. Recursive LLM
     Decomposition        : analyses SOPs / manuals / Visio (.vsdx) files and
                            recursively decomposes processes down to an
                            L4 (activity) / L5 (task) process map.
  3. Persona / System /
     Policy extraction    : documented actors, applications and policies.
  4. Destination          : `documented_process_model` table (PostgreSQL).

Requirements:
    pip install langgraph langchain-anthropic langchain-core sqlalchemy \
                psycopg2-binary pydantic
    # optional parsers:
    pip install pypdf python-docx

Environment:
    ANTHROPIC_API_KEY   - Anthropic key
    PMA_DATABASE_URL    - e.g. postgresql+psycopg2://user:pwd@host:5432/dbname
    PMA_MODEL           - default: claude-sonnet-4-5
    PMA_DOCS_DIR        - folder containing SOPs / manuals / .vsdx files
    PMA_MAX_LEVEL       - deepest level to decompose to (default 5)

Run:
    python process_modeler_agent.py --docs ./internal_docs [--seed-demo]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import uuid
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Literal, Optional, TypedDict

from pydantic import BaseModel, Field
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

# --------------------------------------------------------------------------
# Configuration / logging
# --------------------------------------------------------------------------

MODEL_NAME = os.getenv("PMA_MODEL", "claude-sonnet-4-5")
DATABASE_URL = os.getenv(
    "PMA_DATABASE_URL",
    "postgresql+psycopg2://postgres:postgres@localhost:5432/process_mining",
)
TARGET_AGENT = "Process_Modeler_Agent"
MAX_LEVEL = int(os.getenv("PMA_MAX_LEVEL", "5"))       # decompose to L5
CORPUS_CHAR_BUDGET = 60_000                            # context guard
FALLBACK_SYSTEM_PROMPT = (
    "You are the Process Modeler Agent. You analyse internal documentation "
    "(SOPs, manuals, Visio process diagrams) and build the intended business "
    "process model as a hierarchy from L1 (value chain) to L5 (task). "
    "Be faithful to the documents: model what is DOCUMENTED, not what you "
    "assume happens. Use crisp verb-first names for processes."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")
log = logging.getLogger("process_modeler_agent")

# --------------------------------------------------------------------------
# Structured-output schemas (LLM contracts)
# --------------------------------------------------------------------------

class SubProcess(BaseModel):
    """One child process/activity/task produced by decomposition."""
    name: str = Field(description="Short verb-first name, e.g. 'Validate invoice header'")
    description: str = Field(description="What happens in this step, per the documentation")
    inputs: List[str] = Field(default_factory=list, description="Documented inputs/triggers")
    outputs: List[str] = Field(default_factory=list, description="Documented outputs/results")
    performed_by: List[str] = Field(default_factory=list, description="Roles/personas executing it")
    systems_used: List[str] = Field(default_factory=list, description="Applications/systems used")
    source_evidence: str = Field(default="", description="Short quote/reference from the docs")


class DecompositionResult(BaseModel):
    """LLM answer for: decompose <process> one level down."""
    is_atomic: bool = Field(
        description="True if the documentation gives no further breakdown "
                    "(a leaf task); then sub_processes must be empty."
    )
    rationale: str = Field(description="One sentence on why decomposed / atomic")
    sub_processes: List[SubProcess] = Field(default_factory=list)


class RootProcesses(BaseModel):
    """LLM answer for: identify the top-level processes in the corpus."""
    domain_name: str = Field(description="Name of the overall business domain, e.g. 'Accounts Payable'")
    domain_description: str = Field(description="1-2 sentence description of the domain")
    processes: List[SubProcess] = Field(description="The major (L2) processes found in the documents")


class Persona(BaseModel):
    name: str = Field(description="Role/persona name, e.g. 'AP Clerk'")
    responsibilities: str = Field(description="Documented responsibilities")


class SystemEntity(BaseModel):
    name: str = Field(description="System/application name, e.g. 'SAP ECC', 'DocuWare'")
    role_in_process: str = Field(description="What the system is used for")


class Policy(BaseModel):
    name: str = Field(description="Policy/control/business-rule name")
    statement: str = Field(description="The rule as stated in the documentation")
    applies_to: List[str] = Field(default_factory=list, description="Process steps it constrains")


class EntityExtraction(BaseModel):
    """LLM answer for documented Persona / System / Policy extraction."""
    personas: List[Persona] = Field(default_factory=list)
    systems: List[SystemEntity] = Field(default_factory=list)
    policies: List[Policy] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Document ingestion (SOPs, manuals, Visio)
# --------------------------------------------------------------------------

def _read_txt(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        log.warning("pypdf not installed — skipping %s", path.name)
        return ""
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except ImportError:
        log.warning("python-docx not installed — skipping %s", path.name)
        return ""
    d = docx.Document(str(path))
    parts = [p.text for p in d.paragraphs if p.text.strip()]
    for tbl in d.tables:
        for row in tbl.rows:
            parts.append(" | ".join(c.text.strip() for c in row.cells))
    return "\n".join(parts)


def _read_vsdx(path: Path) -> str:
    """
    Extract shape texts + connections from a Visio .vsdx file (it is a zip of
    XML). No extra dependency needed. Produces a readable pseudo-diagram:
        [Shape text] ; connector: A -> B
    """
    texts: List[str] = []
    edges: List[str] = []
    ns = {"v": "http://schemas.microsoft.com/office/visio/2012/main"}
    try:
        with zipfile.ZipFile(path) as z:
            pages = [n for n in z.namelist()
                     if re.match(r"visio/pages/page\d+\.xml$", n)]
            for page in sorted(pages):
                root = ET.fromstring(z.read(page))
                shape_names: dict[str, str] = {}
                for shape in root.iter(f"{{{ns['v']}}}Shape"):
                    sid = shape.get("ID", "")
                    txt = " ".join(
                        t.text.strip() for t in shape.iter(f"{{{ns['v']}}}Text")
                        if t.text and t.text.strip()
                    )
                    if txt:
                        shape_names[sid] = txt
                        texts.append(f"[Shape] {txt}")
                for conn in root.iter(f"{{{ns['v']}}}Connect"):
                    frm, to = conn.get("FromSheet", ""), conn.get("ToSheet", "")
                    a, b = shape_names.get(frm), shape_names.get(to)
                    if a and b:
                        edges.append(f"[Flow] {a} -> {b}")
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not parse Visio file %s: %s", path.name, exc)
    return "\n".join(texts + edges)


_PARSERS = {
    ".txt": _read_txt, ".md": _read_txt,
    ".pdf": _read_pdf, ".docx": _read_docx, ".vsdx": _read_vsdx,
}


def load_documents(docs_dir: str) -> List[dict]:
    """Read every supported file under docs_dir into {name, type, content}."""
    docs: List[dict] = []
    base = Path(docs_dir)
    if not base.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")
    for path in sorted(base.rglob("*")):
        parser = _PARSERS.get(path.suffix.lower())
        if not parser or not path.is_file():
            continue
        content = parser(path)
        if content.strip():
            docs.append({
                "name": path.name,
                "type": "visio_diagram" if path.suffix.lower() == ".vsdx" else "document",
                "content": content,
            })
            log.info("Ingested %-40s (%d chars)", path.name, len(content))
    return docs


def build_corpus(docs: List[dict], budget: int = CORPUS_CHAR_BUDGET) -> str:
    """Concatenate documents into one tagged corpus within a char budget."""
    if not docs:
        return ""
    per_doc = max(2_000, budget // len(docs))
    blocks = []
    for d in docs:
        body = d["content"][:per_doc]
        blocks.append(f"<document name='{d['name']}' type='{d['type']}'>\n{body}\n</document>")
    return "\n\n".join(blocks)


# --------------------------------------------------------------------------
# Database layer (PostgreSQL)
# --------------------------------------------------------------------------

DDL = """
CREATE TABLE IF NOT EXISTS prompt_registry (
    id            SERIAL PRIMARY KEY,
    target_agent  TEXT        NOT NULL,
    prompt_text   TEXT        NOT NULL,
    version       INTEGER     NOT NULL DEFAULT 1,
    status        TEXT        NOT NULL DEFAULT 'Draft',   -- Draft/Approved/Retired
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS documented_process_model (
    id                 SERIAL PRIMARY KEY,
    run_id             UUID        NOT NULL,
    process_id         TEXT        NOT NULL,     -- e.g. '1.2.3.1'
    parent_process_id  TEXT,
    level              INTEGER     NOT NULL,     -- 1..5 (L1..L5)
    name               TEXT        NOT NULL,
    description        TEXT,
    sequence           INTEGER,
    inputs             JSONB       NOT NULL DEFAULT '[]',
    outputs            JSONB       NOT NULL DEFAULT '[]',
    performed_by       JSONB       NOT NULL DEFAULT '[]',
    systems_used       JSONB       NOT NULL DEFAULT '[]',
    source_evidence    TEXT,
    is_leaf            BOOLEAN     NOT NULL DEFAULT FALSE,
    personas           JSONB,                    -- filled on the L1 row
    systems            JSONB,                    -- filled on the L1 row
    policies           JSONB,                    -- filled on the L1 row
    source_documents   JSONB       NOT NULL DEFAULT '[]',
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dpm_run ON documented_process_model (run_id);
"""


def get_engine() -> Engine:
    return create_engine(DATABASE_URL, pool_pre_ping=True)


def ensure_schema(engine: Engine) -> None:
    with engine.begin() as conn:
        for stmt in DDL.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))


def fetch_agent_prompt(engine: Engine) -> str:
    """Prompt source: latest Approved row for target_agent='Process_Modeler_Agent'."""
    sql = text("""
        SELECT prompt_text
          FROM prompt_registry
         WHERE target_agent = :agent AND status = 'Approved'
         ORDER BY version DESC, created_at DESC
         LIMIT 1
    """)
    with engine.connect() as conn:
        row = conn.execute(sql, {"agent": TARGET_AGENT}).fetchone()
    if row:
        log.info("Loaded Approved prompt for %s from prompt_registry", TARGET_AGENT)
        return row[0]
    log.warning("No Approved prompt found for %s — using fallback", TARGET_AGENT)
    return FALLBACK_SYSTEM_PROMPT


def seed_demo_prompt(engine: Engine) -> None:
    """Insert an Approved prompt row so the pipeline can run end-to-end."""
    with engine.begin() as conn:
        conn.execute(
            text("""INSERT INTO prompt_registry (target_agent, prompt_text, version, status)
                    VALUES (:a, :p, 1, 'Approved')"""),
            {"a": TARGET_AGENT, "p": FALLBACK_SYSTEM_PROMPT},
        )
    log.info("Seeded demo Approved prompt for %s", TARGET_AGENT)


# --------------------------------------------------------------------------
# LangGraph state
# --------------------------------------------------------------------------

class ProcessNode(TypedDict):
    process_id: str            # hierarchical id '1.2.3'
    parent_process_id: Optional[str]
    level: int                 # 1..5
    name: str
    description: str
    sequence: int
    inputs: List[str]
    outputs: List[str]
    performed_by: List[str]
    systems_used: List[str]
    source_evidence: str
    is_leaf: bool


class ModelerState(TypedDict, total=False):
    run_id: str
    docs_dir: str
    documents: List[dict]
    corpus: str
    system_prompt: str
    frontier: List[ProcessNode]        # queue of nodes awaiting decomposition
    process_nodes: List[ProcessNode]   # final flat tree
    personas: List[dict]
    systems: List[dict]
    policies: List[dict]
    rows_written: int
    errors: List[str]


# --------------------------------------------------------------------------
# LLM helpers
# --------------------------------------------------------------------------

def make_llm() -> ChatAnthropic:
    return ChatAnthropic(model=MODEL_NAME, temperature=0, max_tokens=8_000)


LLM = None  # lazily created so importing this file needs no API key


def llm() -> ChatAnthropic:
    global LLM
    if LLM is None:
        LLM = make_llm()
    return LLM


# --------------------------------------------------------------------------
# Graph nodes
# --------------------------------------------------------------------------

def node_load_prompt(state: ModelerState) -> dict:
    """Fetch the governing system prompt from the prompt registry."""
    engine = get_engine()
    ensure_schema(engine)
    return {"system_prompt": fetch_agent_prompt(engine)}


def node_ingest_documents(state: ModelerState) -> dict:
    """Parse SOPs, manuals and Visio diagrams into a tagged corpus."""
    docs = load_documents(state["docs_dir"])
    if not docs:
        raise RuntimeError(f"No parseable documents found in {state['docs_dir']}")
    return {"documents": docs, "corpus": build_corpus(docs)}


def node_seed_root(state: ModelerState) -> dict:
    """
    L1/L2 bootstrap: ask the LLM for the domain (L1) and its major
    processes (L2). L2 nodes go onto the decomposition frontier.
    """
    structured = llm().with_structured_output(RootProcesses)
    result: RootProcesses = structured.invoke([
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=(
            "Below is the internal documentation corpus (SOPs, manuals and "
            "flattened Visio diagrams). Identify the overall business domain "
            "(L1) and its major documented processes (L2).\n\n"
            f"{state['corpus']}"
        )),
    ])

    root: ProcessNode = {
        "process_id": "1", "parent_process_id": None, "level": 1,
        "name": result.domain_name, "description": result.domain_description,
        "sequence": 1, "inputs": [], "outputs": [], "performed_by": [],
        "systems_used": [], "source_evidence": "", "is_leaf": False,
    }
    frontier: List[ProcessNode] = []
    for i, sp in enumerate(result.processes, start=1):
        frontier.append({
            "process_id": f"1.{i}", "parent_process_id": "1", "level": 2,
            "name": sp.name, "description": sp.description, "sequence": i,
            "inputs": sp.inputs, "outputs": sp.outputs,
            "performed_by": sp.performed_by, "systems_used": sp.systems_used,
            "source_evidence": sp.source_evidence, "is_leaf": False,
        })
    log.info("L1 '%s' with %d L2 processes", result.domain_name, len(frontier))
    return {"process_nodes": [root], "frontier": frontier}


def node_decompose(state: ModelerState) -> dict:
    """
    Recursive LLM Decomposition — one frontier node per graph step.
    Pops the next process, asks the LLM to break it one level down,
    and pushes children back onto the frontier until L{MAX_LEVEL} or the
    documentation bottoms out (is_atomic).
    """
    frontier = list(state["frontier"])
    nodes = list(state["process_nodes"])
    current = frontier.pop(0)

    ancestry = " > ".join(
        n["name"] for n in nodes
        if current["process_id"].startswith(n["process_id"] + ".")
        or n["process_id"] == current["process_id"]
    )

    structured = llm().with_structured_output(DecompositionResult)
    result: DecompositionResult = structured.invoke([
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=(
            f"You are decomposing one node of the process hierarchy.\n"
            f"Path      : {ancestry}\n"
            f"Process   : {current['name']} (L{current['level']})\n"
            f"Description: {current['description']}\n\n"
            f"Break this process EXACTLY ONE level down into its documented "
            f"L{current['level'] + 1} steps, in execution order. Only use "
            f"steps supported by the documentation. If the documentation "
            f"provides no further breakdown, mark it atomic.\n\n"
            f"Documentation corpus:\n{state['corpus']}"
        )),
    ])

    if result.is_atomic or not result.sub_processes:
        current["is_leaf"] = True
        log.info("  %s L%d '%s' -> atomic (%s)",
                 current["process_id"], current["level"], current["name"],
                 result.rationale[:80])
    else:
        children: List[ProcessNode] = []
        for i, sp in enumerate(result.sub_processes, start=1):
            child: ProcessNode = {
                "process_id": f"{current['process_id']}.{i}",
                "parent_process_id": current["process_id"],
                "level": current["level"] + 1,
                "name": sp.name, "description": sp.description, "sequence": i,
                "inputs": sp.inputs, "outputs": sp.outputs,
                "performed_by": sp.performed_by,
                "systems_used": sp.systems_used,
                "source_evidence": sp.source_evidence,
                "is_leaf": current["level"] + 1 >= MAX_LEVEL,
            }
            children.append(child)
        # only children below MAX_LEVEL keep recursing
        frontier.extend(c for c in children if c["level"] < MAX_LEVEL)
        nodes.extend(children)
        log.info("  %s L%d '%s' -> %d x L%d",
                 current["process_id"], current["level"], current["name"],
                 len(children), current["level"] + 1)

    # replace/record the current node
    nodes = [n for n in nodes if n["process_id"] != current["process_id"]]
    nodes.append(current)
    return {"frontier": frontier, "process_nodes": nodes}


def route_decompose(state: ModelerState) -> Literal["decompose", "extract_entities"]:
    """Loop until the frontier is empty."""
    return "decompose" if state.get("frontier") else "extract_entities"


def node_extract_entities(state: ModelerState) -> dict:
    """Documented Persona / System / Policy extraction."""
    structured = llm().with_structured_output(EntityExtraction)
    result: EntityExtraction = structured.invoke([
        SystemMessage(content=state["system_prompt"]),
        HumanMessage(content=(
            "From the documentation corpus below, extract every DOCUMENTED:\n"
            "1. Persona (roles who perform work)\n"
            "2. System (applications / tools used)\n"
            "3. Policy (business rules, controls, SLAs, compliance constraints)\n\n"
            f"Documentation corpus:\n{state['corpus']}"
        )),
    ])
    log.info("Extracted %d personas, %d systems, %d policies",
             len(result.personas), len(result.systems), len(result.policies))
    return {
        "personas": [p.model_dump() for p in result.personas],
        "systems": [s.model_dump() for s in result.systems],
        "policies": [p.model_dump() for p in result.policies],
    }


def node_persist(state: ModelerState) -> dict:
    """Destination: documented_process_model table."""
    engine = get_engine()
    ensure_schema(engine)
    src_docs = json.dumps([d["name"] for d in state["documents"]])
    ordered = sorted(
        state["process_nodes"],
        key=lambda n: [int(x) for x in n["process_id"].split(".")],
    )
    insert = text("""
        INSERT INTO documented_process_model
            (run_id, process_id, parent_process_id, level, name, description,
             sequence, inputs, outputs, performed_by, systems_used,
             source_evidence, is_leaf, personas, systems, policies,
             source_documents)
        VALUES
            (:run_id, :process_id, :parent, :level, :name, :description,
             :sequence, :inputs, :outputs, :performed_by, :systems_used,
             :evidence, :is_leaf, :personas, :systems, :policies, :src)
    """)
    with engine.begin() as conn:
        for n in ordered:
            is_root = n["level"] == 1
            conn.execute(insert, {
                "run_id": state["run_id"],
                "process_id": n["process_id"],
                "parent": n["parent_process_id"],
                "level": n["level"],
                "name": n["name"],
                "description": n["description"],
                "sequence": n["sequence"],
                "inputs": json.dumps(n["inputs"]),
                "outputs": json.dumps(n["outputs"]),
                "performed_by": json.dumps(n["performed_by"]),
                "systems_used": json.dumps(n["systems_used"]),
                "evidence": n["source_evidence"],
                "is_leaf": n["is_leaf"],
                "personas": json.dumps(state.get("personas", [])) if is_root else None,
                "systems": json.dumps(state.get("systems", [])) if is_root else None,
                "policies": json.dumps(state.get("policies", [])) if is_root else None,
                "src": src_docs,
            })
    log.info("Persisted %d rows to documented_process_model (run_id=%s)",
             len(ordered), state["run_id"])
    return {"rows_written": len(ordered)}


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

def build_graph():
    g = StateGraph(ModelerState)
    g.add_node("load_prompt", node_load_prompt)
    g.add_node("ingest_documents", node_ingest_documents)
    g.add_node("seed_root", node_seed_root)
    g.add_node("decompose", node_decompose)
    g.add_node("extract_entities", node_extract_entities)
    g.add_node("persist", node_persist)

    g.add_edge(START, "load_prompt")
    g.add_edge("load_prompt", "ingest_documents")
    g.add_edge("ingest_documents", "seed_root")
    g.add_edge("seed_root", "decompose")
    g.add_conditional_edges("decompose", route_decompose,
                            {"decompose": "decompose",
                             "extract_entities": "extract_entities"})
    g.add_edge("extract_entities", "persist")
    g.add_edge("persist", END)
    return g.compile()


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description="Agent 2A: Process Modeler Agent")
    ap.add_argument("--docs", default=os.getenv("PMA_DOCS_DIR", "./internal_docs"),
                    help="Directory with SOPs / manuals / .vsdx files")
    ap.add_argument("--seed-demo", action="store_true",
                    help="Insert a demo Approved prompt into prompt_registry first")
    args = ap.parse_args()

    if args.seed_demo:
        engine = get_engine()
        ensure_schema(engine)
        seed_demo_prompt(engine)

    graph = build_graph()
    run_id = str(uuid.uuid4())
    log.info("Starting Process Modeler run %s over %s", run_id, args.docs)

    final = graph.invoke(
        {"run_id": run_id, "docs_dir": args.docs},
        # generous limit: the decompose node loops once per process node
        config={"recursion_limit": 500},
    )

    print("\n================ PROCESS MODELER RUN COMPLETE ================")
    print(f"run_id        : {run_id}")
    print(f"rows written  : {final.get('rows_written')}")
    print(f"personas      : {len(final.get('personas', []))}")
    print(f"systems       : {len(final.get('systems', []))}")
    print(f"policies      : {len(final.get('policies', []))}")
    print("\nProcess map:")
    for n in sorted(final["process_nodes"],
                    key=lambda n: [int(x) for x in n["process_id"].split(".")]):
        indent = "    " * (n["level"] - 1)
        leaf = " (leaf)" if n["is_leaf"] else ""
        print(f"{indent}L{n['level']} {n['process_id']}  {n['name']}{leaf}")


if __name__ == "__main__":
    main()
