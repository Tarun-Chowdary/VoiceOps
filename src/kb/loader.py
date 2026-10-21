"""Load kb/*.md into section-level chunks that keep their [SECTION-ID] and metadata.

Why section-level chunks: the grader checks retrieval by section ID (e.g. RET-2),
so each chunk must be exactly one documented section, with its source metadata
(doc_type, status, effective_from) preserved for precedence handling later.
"""
import re
from dataclasses import dataclass, field
from src import config

# KB-00.1 precedence: lower number wins when documents disagree
DOC_PRIORITY = {"policy": 1, "agent_guide": 2, "help": 3, "faq": 4, "marketing": 5, "meta": 0}

_HEADING = re.compile(r"^#{2,3}\s*\[([A-Za-z0-9.\-]+)\]\s*(.*)$")


@dataclass
class Chunk:
    section_id: str
    heading: str
    text: str                      # heading + body (what we show the LLM)
    doc_id: str
    doc_title: str
    doc_type: str
    status: str
    effective_from: str
    file: str
    priority: int
    meta: dict = field(default_factory=dict)

    @property
    def search_text(self) -> str:
        """Text used for retrieval: includes doc title so 'returns policy' matters."""
        return f"{self.doc_title} {self.heading} {self.text}"


def _front_matter(raw: str):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def load_chunks(kb_dir=None) -> list[Chunk]:
    kb_dir = kb_dir or config.KB_DIR
    chunks: list[Chunk] = []
    for path in sorted(kb_dir.glob("*.md")):
        meta, body = _front_matter(path.read_text(encoding="utf-8"))
        cur = None
        for line in body.splitlines():
            h = _HEADING.match(line)
            if h:
                if cur:
                    chunks.append(cur)
                cur = Chunk(
                    section_id=h.group(1), heading=h.group(2).strip(),
                    text=line.strip(), doc_id=meta.get("doc_id", path.stem),
                    doc_title=meta.get("title", path.stem),
                    doc_type=meta.get("doc_type", "faq"),
                    status=meta.get("status", "current"),
                    effective_from=meta.get("effective_from", ""),
                    file=path.name,
                    priority=DOC_PRIORITY.get(meta.get("doc_type", "faq"), 4),
                    meta=meta)
            elif cur:
                cur.text += "\n" + line
        if cur:
            chunks.append(cur)
    for c in chunks:
        c.text = c.text.strip()
    return chunks
