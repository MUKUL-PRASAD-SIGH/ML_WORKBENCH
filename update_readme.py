#!/usr/bin/env python3
"""
update_readme.py
────────────────
Auto-updates the Projects table in the root README.md whenever
a new project folder is added to this repo.

HOW TO USE
──────────
  python update_readme.py

HOW IT WORKS
────────────
1. Scans every top-level folder (excluding hidden dirs, __pycache__, etc.)
2. Reads each folder's README.md (if it exists) to extract metadata
3. Rebuilds the Projects table between <!-- PROJECTS_START --> and <!-- PROJECTS_END -->
4. Writes the updated README.md back

ADDING METADATA TO A PROJECT
─────────────────────────────
In your project's README.md, add a YAML-like front-matter block at the top
(between triple dashes) or just let the script auto-detect from the file.

For custom control, add a `.project.json` in your project folder:
{
  "name": "Sentiment Analyser",
  "description": "Short one-liner description",
  "stack": "Python · Streamlit · HuggingFace",
  "demo_url": "https://sentilyticz.streamlit.app/",
  "status": "✅ Live"
}
"""

import os
import re
import json
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
README_PATH = ROOT / "README.md"

# Folders to skip when scanning for projects
SKIP_FOLDERS = {
    ".git", ".github", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".idea", ".vscode", "dist", "build", ".agents",
    "_agents", ".agent", "_agent",
}

# Marker comments in README.md that wrap the auto-managed table
START_MARKER = "<!-- PROJECTS_START -->"
END_MARKER   = "<!-- PROJECTS_END -->"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def get_project_meta(folder: Path) -> dict | None:
    """
    Extract project metadata from a folder.
    Priority: .project.json > README.md heuristics > folder name fallback
    """

    # 1. Try .project.json (explicit metadata)
    meta_file = folder / ".project.json"
    if meta_file.exists():
        try:
            with open(meta_file, encoding="utf-8") as f:
                data = json.load(f)
            return {
                "name":        data.get("name",        folder.name.replace("_", " ")),
                "description": data.get("description", "No description yet."),
                "stack":       data.get("stack",        "Python"),
                "demo_url":    data.get("demo_url",     ""),
                "status":      data.get("status",       "🔲 WIP"),
                "folder":      folder.name,
            }
        except (json.JSONDecodeError, OSError):
            pass  # fall through to README heuristics

    # 2. Try README.md heuristics
    readme = folder / "README.md"
    if readme.exists():
        try:
            content = readme.read_text(encoding="utf-8")

            # Extract first H1 as name
            name_match = re.search(r"^#\s+(.+)", content, re.MULTILINE)
            name = name_match.group(1).strip() if name_match else folder.name.replace("_", " ")

            # Extract Live Demo URL (look for common patterns)
            demo_match = re.search(
                r"\*\*Live Demo[:\*]*\*?\*?\s*:?\s*\[?[^\]]*\]?\(?(https?://[^\s)>\"]+)",
                content, re.IGNORECASE
            )
            if not demo_match:
                demo_match = re.search(r"(https?://[^\s)>\"]+streamlit\.app[^\s)>\"]*)", content)
            if not demo_match:
                demo_match = re.search(r"(https?://[^\s)>\"]+\.app[^\s)>\"]*)", content)
            demo_url = demo_match.group(1).strip() if demo_match else ""

            # Extract stack line
            stack_match = re.search(
                r"\*\*Core Stack[:\*]*\*?\*?\s*:?\s*(.+)", content, re.IGNORECASE
            )
            if not stack_match:
                stack_match = re.search(r"Stack[:\s]+(.+)", content, re.IGNORECASE)
            stack = stack_match.group(1).strip() if stack_match else "Python"

            # Extract first meaningful paragraph as description
            # Strip markdown headers, badges, and blank lines
            lines = content.split("\n")
            desc = ""
            for line in lines:
                line = line.strip()
                if (line and
                        not line.startswith("#") and
                        not line.startswith(">") and
                        not line.startswith("!") and
                        not line.startswith("|") and
                        not line.startswith("```") and
                        not line.startswith("[![") and
                        len(line) > 40):
                    # Remove markdown bold/italic/link syntax for cleanliness
                    desc = re.sub(r"\*\*?([^*]+)\*\*?", r"\1", line)
                    desc = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", desc)
                    desc = desc[:200] + ("…" if len(desc) > 200 else "")
                    break

            status = "✅ Live" if demo_url else "🔲 WIP"

            return {
                "name":        name,
                "description": desc or "See project README for details.",
                "stack":       stack,
                "demo_url":    demo_url,
                "status":      status,
                "folder":      folder.name,
            }
        except OSError:
            pass

    # 3. Fallback — bare minimum from folder name only
    return {
        "name":        folder.name.replace("_", " "),
        "description": "Documentation coming soon.",
        "stack":       "Python",
        "demo_url":    "",
        "status":      "🔲 WIP",
        "folder":      folder.name,
    }


def scan_projects() -> list[dict]:
    """Return sorted list of project metadata dicts."""
    projects = []
    for entry in sorted(ROOT.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in SKIP_FOLDERS or entry.name.startswith("."):
            continue
        # A folder qualifies as a project if it has .project.json, README.md,
        # or any code file directly in its root. We NEVER recurse into subdirs
        # to avoid hanging on large model caches (e.g. finetuned_distilbert/).
        CODE_EXTS = {".py", ".js", ".ts", ".ipynb", ".html", ".json"}
        root_files = [f for f in entry.iterdir() if f.is_file()]
        has_project_marker = (entry / ".project.json").exists() or (entry / "README.md").exists()
        has_code = has_project_marker or any(f.suffix in CODE_EXTS for f in root_files)
        if not has_code:
            continue
        meta = get_project_meta(entry)
        if meta:
            projects.append(meta)
    return projects


def build_table(projects: list[dict]) -> str:
    """Render the markdown table string for the projects list."""
    if not projects:
        return "_No projects yet — first one coming soon!_"

    rows = ["| # | Project | Description | Stack | Live Demo | Status |",
            "|---|---------|-------------|-------|-----------|--------|"]

    for i, p in enumerate(projects, start=1):
        num  = f"{i:02d}"
        name = f"[**{p['name']}**](./{p['folder']}/)"
        desc = p["description"]
        stack = p["stack"]
        demo  = (f"[**🌐 Try it live →**]({p['demo_url']})" if p["demo_url"]
                 else "Coming soon")
        status = p["status"]
        rows.append(f"| {num} | {name} | {desc} | {stack} | {demo} | {status} |")

    return "\n".join(rows)


def update_readme(projects: list[dict]) -> bool:
    """Patch README.md between the marker comments. Returns True if changed."""
    if not README_PATH.exists():
        print(f"❌  README not found at {README_PATH}")
        return False

    content = README_PATH.read_text(encoding="utf-8")

    start_idx = content.find(START_MARKER)
    end_idx   = content.find(END_MARKER)

    if start_idx == -1 or end_idx == -1:
        print("❌  Markers <!-- PROJECTS_START --> / <!-- PROJECTS_END --> not found in README.md")
        return False

    new_table  = build_table(projects)
    new_block  = f"{START_MARKER}\n{new_table}\n{END_MARKER}"
    old_block  = content[start_idx : end_idx + len(END_MARKER)]

    if old_block == new_block:
        print("✅  README is already up-to-date. No changes needed.")
        return False

    new_content = content[:start_idx] + new_block + content[end_idx + len(END_MARKER):]
    README_PATH.write_text(new_content, encoding="utf-8")
    print(f"✅  README updated with {len(projects)} project(s).")
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    print("🔍  Scanning for projects …")
    projects = scan_projects()

    if projects:
        print(f"📦  Found {len(projects)} project(s):")
        for p in projects:
            demo_label = p['demo_url'] if p['demo_url'] else "no demo yet"
            print(f"     • {p['name']}  ({p['folder']})  →  {demo_label}")
    else:
        print("     No project folders detected yet.")

    print()
    update_readme(projects)


if __name__ == "__main__":
    main()
