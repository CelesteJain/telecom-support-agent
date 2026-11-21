"""Write the agent's workflow diagram (Mermaid) straight from the compiled graph,
so the diagram in the README can never drift from the code."""
from pathlib import Path

from agent.graph import build_graph

OUT = Path("docs/architecture.mmd")

if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(build_graph().get_graph().draw_mermaid(), encoding="utf-8")
    print(f"Wrote {OUT}")