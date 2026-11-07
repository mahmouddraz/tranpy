#!/usr/bin/env python3
"""Number audit of the manuscript prose (papers/Access-2026-DE-TSA).

Every numeric token in ``sections/*.tex`` must be traceable to one of
  1. ``tables/numbers.json`` (value -> source, written by scripts/paper_tables.py),
  2. a generated table ``tables/*.tex`` (same numbers, rounded as printed),
  3. ``audit_derived.yaml`` -> ``derived``: a formula over numbers.json keys that is
     re-evaluated here and must reproduce the printed value,
  4. ``audit_derived.yaml`` -> ``sourced``: a number with a file:line source that the
     table generator does not carry (hardware, model facts, knees.json),
or it is listed under UNMATCHED and the exit code is 1.

Tokens that are not quantities (citation keys, labels, model version strings such as
"TabPFN 3.5", voltage-level names such as "220/380 kV", dates, list markers, the
constants 0 and 1 inside formulas) are removed by the ``ignore`` patterns of the YAML
file and listed as IGNORED so the filter stays visible.

The audit also lists every ``\\todo`` of ``main.tex``, ``sections/*.tex`` and
``tables/*.tex`` and checks that each is a row of ``TODO.md``.

Usage (see the manuscript Makefile, ``make audit``):
  paper_audit.py --sections sections --numbers tables/numbers.json --tables tables \\
                 --derived audit_derived.yaml --todo TODO.md --main main.tex --out audit.md
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

# ----------------------------------------------------------------------------------------
# LaTeX -> plain text with numbers preserved
# ----------------------------------------------------------------------------------------
_BRACED = r"\{((?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*)\}"  # one balanced {...} (3 levels)


def strip_comments(tex: str) -> str:
    out = []
    for line in tex.splitlines():
        i, s = 0, line
        while True:
            j = s.find("%", i)
            if j < 0:
                break
            if j > 0 and s[j - 1] == "\\":
                i = j + 1
                continue
            s = s[:j]
            break
        out.append(s)
    return "\n".join(out)


def drop_cmd(tex: str, name: str, nargs: int = 1, keep: tuple[int, ...] = ()) -> str:
    """Remove ``\\name[opt]{a}{b}...``; keep the arguments listed in ``keep`` (0-based)."""
    rx = re.compile(r"\\" + name + r"\*?(?:\[[^\]]*\])?" + _BRACED * nargs)
    return rx.sub(lambda m: " ".join(m.group(i + 1) for i in keep) if keep else " ", tex)


def normalise(tex: str) -> str:
    t = strip_comments(tex)
    t = drop_cmd(t, "cite")
    t = drop_cmd(t, "ref")
    t = drop_cmd(t, "eqref")
    t = drop_cmd(t, "label")
    t = drop_cmd(t, "todo")                      # placeholders are audited via TODO.md
    t = drop_cmd(t, "texttt")                    # file and column names
    t = drop_cmd(t, "paperinputtablewide", 2)    # fallback caption: not rendered
    t = drop_cmd(t, "paperinputtable", 2)
    t = drop_cmd(t, "paperfigurewide", 3, keep=(2,))  # rendered caption only
    t = drop_cmd(t, "paperfigure", 3, keep=(2,))
    t = drop_cmd(t, "PARstart", 2)
    t = drop_cmd(t, "begin")
    t = drop_cmd(t, "end")
    t = drop_cmd(t, "section")
    t = drop_cmd(t, "subsection")
    t = drop_cmd(t, "textbf", 1, keep=(0,))
    t = drop_cmd(t, "textit", 1, keep=(0,))
    t = drop_cmd(t, "emph", 1, keep=(0,))
    t = drop_cmd(t, "footnote", 1, keep=(0,))
    t = drop_cmd(t, "num", 1, keep=(0,))
    t = drop_cmd(t, "mathrm", 1, keep=(0,))
    t = drop_cmd(t, "text", 1, keep=(0,))
    t = re.sub(r"\\times\s*10\^\{(-?\d+)\}", r"e\1", t)   # 1.6\times10^{-23} -> 1.6e-23
    t = re.sub(r"\\times\s*10\^(-?\d)", r"e\1", t)
    t = re.sub(r"\^\{?\\circ\}?", "", t)                   # 180^\circ -> 180
    t = t.replace("{,}", ",").replace("\\,", " ").replace("\\;", " ").replace("\\!", "")
    t = t.replace("~", " ").replace("\\%", "%")
    t = t.replace("\\&", "&").replace("\\_", "_").replace("$", " ")
    t = re.sub(r"\\[a-zA-Z]+", " ", t)                     # remaining control words
    t = re.sub(r"[{}]", " ", t)
    return t


NUM_RX = re.compile(
    r"(?<![\w.^\-])"                        # not glued to a word, a decimal or an exponent
    r"(-?)"
    r"(\d{1,3}(?:,\d{3})+(?:\.\d+)?"        # 11,159  2,048  47,625
    r"|\d+\.\d+(?:e-?\d+)?"                 # 0.867  1.6e-23  4.9e-5
    r"|\d+e-?\d+"                           # 1e-5
    r"|\d+)"                                # 18
    r"(?![\w])"                             # not glued to a trailing letter (2H, 9bus)
)


@dataclass
class Token:
    file: str
    line: int
    text: str          # as printed, e.g. "11,159" or "0.867"
    value: float
    context: str
    status: str = ""
    source: str = ""


def to_value(text: str) -> float:
    return float(text.replace(",", "").replace("−", "-"))


def decimals(text: str) -> int:
    t = text.replace(",", "")
    if "e" in t:
        mant, exp = t.split("e")
        return (len(mant.split(".")[1]) if "." in mant else 0) - int(exp)
    return len(t.split(".")[1]) if "." in t else 0


def same(printed: str, value: float, allow_round: bool = False) -> bool:
    """A printed token matches a source value when it is equal, or (``allow_round``) when the
    token has at least two decimals and the source value rounds to it."""
    p = to_value(printed)
    if p == value:
        return True
    d = decimals(printed)
    if not allow_round or d < 2:
        return False
    return abs(p - value) <= 0.5 * 10 ** (-d) + 1e-9 * abs(value)


def tokens_of(path: Path, ignore: list[tuple[re.Pattern[str], str]],
              ignored_out: list[tuple[str, int, str, str]]) -> list[Token]:
    raw = path.read_text()
    toks: list[Token] = []
    for ln, line in enumerate(normalise(raw).splitlines(), start=1):
        s = line
        for rx, why in ignore:
            for m in rx.finditer(s):
                if re.search(r"\d", m.group(0)):
                    ignored_out.append((path.name, ln, m.group(0).strip(), why))
            s = rx.sub(" ", s)
        for m in NUM_RX.finditer(s):
            text = m.group(1) + m.group(2)
            a, b = max(0, m.start() - 80), min(len(s), m.end() + 80)
            ctx = re.sub(r"\s+", " ", s[a:b]).strip()
            toks.append(Token(path.name, ln, text, to_value(text), ctx))
    return toks


# ----------------------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------------------
def load_numbers(path: Path) -> dict[str, dict]:
    recs = json.loads(path.read_text())["numbers"]
    for k, r in recs.items():
        r["_values"] = []
        v = r["value"]
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            r["_values"].append(float(v))
        txt = normalise(str(r["text"]))
        for m in NUM_RX.finditer(txt):
            r["_values"].append(to_value(m.group(1) + m.group(2)))
    return recs


def table_numbers(folder: Path) -> list[tuple[float, str, str]]:
    out = []
    for f in sorted(folder.glob("*.tex")):
        txt = normalise(f.read_text())
        label = re.search(r"\\label\{(tab:\w+)\}", f.read_text())
        lab = label.group(1) if label else f.name
        for m in NUM_RX.finditer(txt):
            text = m.group(1) + m.group(2)
            out.append((to_value(text), text, f"tables/{f.name} ({lab})"))
    return out


def eval_formula(formula: str, recs: dict[str, dict]) -> float:
    def repl(m: re.Match[str]) -> str:
        key = m.group(0)
        if key not in recs:
            raise KeyError(key)
        v = recs[key]["value"]
        if isinstance(v, str):
            v = to_value(normalise(v).split()[0])
        return repr(float(v))
    expr = re.sub(r"[A-Za-z_][A-Za-z0-9_.]*", lambda m: repl(m) if m.group(0) not in
                  ("abs", "min", "max", "round", "sqrt", "log10") else m.group(0), formula)
    return float(eval(expr, {"__builtins__": {}}, {"abs": abs, "min": min, "max": max,
                                                    "round": round, "sqrt": math.sqrt,
                                                    "log10": math.log10}))


@dataclass
class Derived:
    text: str
    formula: str
    note: str
    value: float | None = None
    ok: bool = False
    error: str = ""


@dataclass
class Sourced:
    text: str
    source: str
    context: str = ""
    rx: re.Pattern[str] | None = field(default=None, repr=False)


def load_yaml(path: Path, recs: dict[str, dict]) -> tuple[list[Derived], list[Sourced],
                                                            list[tuple[re.Pattern[str], str]]]:
    y = yaml.safe_load(path.read_text()) if path.exists() else {}
    derived: list[Derived] = []
    for d in y.get("derived", []) or []:
        item = Derived(str(d["text"]), str(d["formula"]), str(d.get("note", "")))
        try:
            item.value = eval_formula(item.formula, recs)
            d = decimals(item.text)
            item.ok = abs(to_value(item.text) - item.value) <= 0.5 * 10 ** (-d) + 1e-9
            if not item.ok:
                item.error = f"formula gives {item.value:.6g}, prose prints {item.text}"
        except KeyError as e:
            item.error = f"unknown numbers.json key {e}"
        except Exception as e:  # noqa: BLE001
            item.error = f"{type(e).__name__}: {e}"
        derived.append(item)
    sourced: list[Sourced] = []
    for s in y.get("sourced", []) or []:
        item = Sourced(str(s["text"]), str(s["source"]), str(s.get("context", "")))
        if item.context:
            item.rx = re.compile(item.context)
        sourced.append(item)
    ignore = [(re.compile(i["pattern"]), str(i.get("why", ""))) for i in (y.get("ignore", []) or [])]
    return derived, sourced, ignore


# ----------------------------------------------------------------------------------------
# Matching
# ----------------------------------------------------------------------------------------
def match(tok: Token, recs: dict[str, dict], tabnums: list[tuple[float, str, str]],
          derived: list[Derived], sourced: list[Sourced]) -> None:
    # 1. the audited YAML (derived formulas, sourced numbers) takes precedence: it is explicit
    for s in sourced:
        if s.rx is not None and to_value(s.text) == tok.value and s.rx.search(tok.context):
            tok.status, tok.source = "sourced", s.source
            return
    for d in derived:
        if d.ok and to_value(d.text) == tok.value:
            tok.status, tok.source = "derived", f"{d.formula}" + (f" ({d.note})" if d.note else "")
            return
    for s in sourced:
        if s.rx is None and to_value(s.text) == tok.value:
            tok.status, tok.source = "sourced", s.source
            return
    # 2. numbers.json / generated tables: exact value first, then rounding to >= 2 decimals
    for allow_round in (False, True):
        hits = []
        for k, r in recs.items():
            vals = r["_values"]
            if not vals:
                continue
            main = vals[0]
            verbatim = 0 if tok.text in normalise(str(r["text"])) else 1
            if same(tok.text, main, allow_round):
                hits.append((verbatim, abs(tok.value - main), 0, k))
            else:
                for v in vals[1:]:
                    if same(tok.text, v, allow_round):
                        hits.append((verbatim, abs(tok.value - v), 1, k))
        if hits:
            hits.sort()
            k = hits[0][3]
            r = recs[k]
            line = f":{r['line']}" if r.get("line") else ""
            tok.status = "numbers.json" if not allow_round else "numbers.json (rounded)"
            tok.source = f"{k} = {r['text']} <- {r['source']}{line}"
            others = [h[3] for h in hits[1:4] if h[3] != k]
            if allow_round and others:
                tok.source += " (also rounds from " + ", ".join(others) + ")"
            return
        for v, text, where in tabnums:
            if same(tok.text, v, allow_round):
                tok.status = "table" if not allow_round else "table (rounded)"
                tok.source = f"{text} in {where}"
                return
    tok.status = "UNMATCHED"


# ----------------------------------------------------------------------------------------
# \todo <-> TODO.md
# ----------------------------------------------------------------------------------------
def find_todos(files: list[Path]) -> list[tuple[str, int, str]]:
    out = []
    rx = re.compile(r"\\todo(?:\[[^\]]*\])?" + _BRACED)
    for f in files:
        for ln, line in enumerate(strip_comments(f.read_text()).splitlines(), start=1):
            for m in rx.finditer(line):
                out.append((str(f), ln, m.group(1)))
    return out


def todo_listed(text: str, todo_md: str) -> bool:
    plain = re.sub(r"\\[a-zA-Z]+|[{}\\]", "", text)
    key = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", plain.lower())).strip()
    md = re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]", " ", todo_md.lower()))
    head = " ".join(key.split()[:4])
    return head in md


# ----------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sections", type=Path, default=Path("sections"))
    ap.add_argument("--numbers", type=Path, default=Path("tables/numbers.json"))
    ap.add_argument("--tables", type=Path, default=Path("tables"))
    ap.add_argument("--derived", type=Path, default=Path("audit_derived.yaml"))
    ap.add_argument("--todo", type=Path, default=Path("TODO.md"))
    ap.add_argument("--main", type=Path, default=Path("main.tex"))
    ap.add_argument("--out", type=Path, default=Path("audit.md"))
    a = ap.parse_args(argv)

    recs = load_numbers(a.numbers)
    tabnums = table_numbers(a.tables)
    derived, sourced, ignore = load_yaml(a.derived, recs)

    ignored: list[tuple[str, int, str, str]] = []
    toks: list[Token] = []
    for f in sorted(a.sections.glob("*.tex")):
        toks.extend(tokens_of(f, ignore, ignored))
    for t in toks:
        match(t, recs, tabnums, derived, sourced)

    todo_files = [a.main] + sorted(a.sections.glob("*.tex")) + sorted(a.tables.glob("*.tex"))
    todos = find_todos([f for f in todo_files if f.exists()])
    todo_md = a.todo.read_text() if a.todo.exists() else ""
    todo_missing = [t for t in todos if not todo_listed(t[2], todo_md)]
    bad_derived = [d for d in derived if not d.ok]
    unmatched = [t for t in toks if t.status == "UNMATCHED"]

    counts = {s: sum(1 for t in toks if t.status == s)
              for s in ("numbers.json", "numbers.json (rounded)", "table", "table (rounded)",
                        "derived", "sourced", "UNMATCHED")}

    L: list[str] = []
    L.append("# Number audit\n")
    L.append(f"Prose files: `{a.sections}/*.tex`; sources: `{a.numbers}` ({len(recs)} keys), "
             f"`{a.tables}/*.tex` ({len(tabnums)} printed numbers), `{a.derived}` "
             f"({len(derived)} derived, {len(sourced)} sourced, {len(ignore)} ignore patterns).\n")
    L.append("| status | tokens |\n|---|---|")
    for k, v in counts.items():
        L.append(f"| {k} | {v} |")
    L.append(f"| ignored (not quantities) | {len(ignored)} |")
    L.append(f"| `\\todo` found / not in TODO.md | {len(todos)} / {len(todo_missing)} |")
    L.append(f"| derived formulas failing | {len(bad_derived)} |\n")

    L.append("## UNMATCHED\n")
    if unmatched:
        L.append("| file | line | number | context |\n|---|---|---|---|")
        for t in unmatched:
            L.append(f"| {t.file} | {t.line} | `{t.text}` | {t.context} |")
    else:
        L.append("none")
    L.append("")

    if bad_derived:
        L.append("## Derived formulas that do not reproduce the prose\n")
        for d in bad_derived:
            L.append(f"- `{d.text}` = `{d.formula}`: {d.error}")
        L.append("")

    L.append("## Derived numbers (formula re-evaluated)\n")
    L.append("| printed | formula | value | note |\n|---|---|---|---|")
    for d in derived:
        L.append(f"| {d.text} | `{d.formula}` | {d.value if d.value is None else f'{d.value:.6g}'} | {d.note} |")
    L.append("")

    L.append("## `\\todo` placeholders\n")
    L.append("| file | line | text | in TODO.md |\n|---|---|---|---|")
    for f, ln, text in todos:
        L.append(f"| {f} | {ln} | {text} | {'no' if (f, ln, text) in todo_missing else 'yes'} |")
    L.append("")

    L.append("## Every number with its source\n")
    L.append("| file | line | number | status | source or formula |\n|---|---|---|---|---|")
    for t in toks:
        L.append(f"| {t.file} | {t.line} | `{t.text}` | {t.status} | {t.source} |")
    L.append("")

    L.append("## Ignored tokens (not quantities)\n")
    L.append("| file | line | text | why |\n|---|---|---|---|")
    for f, ln, text, why in ignored:
        L.append(f"| {f} | {ln} | `{text}` | {why} |")
    L.append("")

    a.out.write_text("\n".join(L))
    print(f"audit: {len(toks)} numbers; " + ", ".join(f"{k} {v}" for k, v in counts.items())
          + f"; ignored {len(ignored)}; todo {len(todos)} ({len(todo_missing)} missing in TODO.md)"
          + f"; derived failing {len(bad_derived)} -> {a.out}")
    for t in unmatched:
        print(f"  UNMATCHED {t.file}:{t.line} {t.text!r}  {t.context}")
    for d in bad_derived:
        print(f"  DERIVED {d.text!r}: {d.error}")
    for f, ln, text in todo_missing:
        print(f"  TODO not listed: {f}:{ln} {text}")
    return 1 if (unmatched or bad_derived or todo_missing) else 0


if __name__ == "__main__":
    sys.exit(main())
