import json
import re
from datasets import load_dataset
from tqdm import tqdm
import random
import os

OUT_DIR = "data"
TRAIN_OUTFILE = os.path.join(OUT_DIR, "math_train.json")
VALID_OUTFILE = os.path.join(OUT_DIR, "math_valid.json")
VALID_RATIO = 0.10
SEED = 42
random.seed(SEED)

# Precompile regexes
MATRIX_RE = re.compile(r"\\begin\{(pmatrix|bmatrix|Bmatrix)\}(.*?)\\end\{\1\}", re.S)
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
FRAC_RE = re.compile(r"\\frac\s*\{\s*([^{}]+?)\s*\}\s*\{\s*([^{}]+?)\s*\}")
BRACES_RE = re.compile(r"\{([^{}]+)\}")
MATH_DELIM_RE = re.compile(r"\${1,2}|\\\(|\\\)|\\\[|\\\]")
UNKNOWN_CMD_RE = re.compile(r"\\(?!pi\b|sin\b|cos\b|tan\b|ln\b|sqrt\b|sum\b|dbinom\b)[A-Za-z]+\*?")
SUM_RE = re.compile(r"\\sum_\{([^{}]+)\}\^\{([^{}]+)\}")

LATEX_SYMBOLS = {
        r"\left": "",
        r"\right": "",
        r"\le": "<=",
        r"\ge": ">=",
        r"\theta": "theta",
        r"\phi": "phi",
        r"\alpha": "alpha",
        r"\beta": "beta",
        r"\gamma": "gamma",
        r"\infty": "infinity",
        r"\pi": "pi",
        r"\equiv": "≡",
        r"^\circ": "°",
        r"\pm": "+/-"
    }

ANSWER_PATTERNS = [
    re.compile(r"(?i)^\s*answer[:\s]+(.+)$", flags=re.M),
    re.compile(r"(?i)final answer[:\s]+(.+)$", flags=re.M),
    re.compile(r"(?i)the answer is[:\s]+(.+)$", flags=re.M),
]

def convert_matrix(s: str) -> str:
    def repl(m):
        rows = [row.strip() for row in m.group(2).strip().split(r"\\") if row.strip()]
        matrix_list = [[clean_latex(entry.strip()) for entry in row.split("&")] for row in rows]
        return str(matrix_list)
    return MATRIX_RE.sub(repl, s)

def clean_latex(text: str) -> str:
    if not text:
        return ""
    s = text

    # Combine LaTeX symbols and Greek letters in one pass
    s = re.sub(r"\\pmod\s*\{\s*([^{}]+?)\s*\}", r"(mod \1)", s)
    symbol_re = re.compile("|".join(re.escape(k) for k in LATEX_SYMBOLS.keys()))
    s = symbol_re.sub(lambda m: LATEX_SYMBOLS[m.group(0)], s)
    # [asy] blocks
    s = re.sub(r"\[asy\].*?\[/asy\]", "[diagram]", s, flags=re.S)

    s = convert_matrix(s) 
    s = s.replace("&=", "=").replace("\pm"," +/- ")
    s = s.replace("\\\\", " \n")
    s = MATH_DELIM_RE.sub("", s)

    # --- Convert summations \sum_{i=a}^{b} ---
    s = SUM_RE.sub(r"summation from \1 to \2", s)

    # --- Functions like sin, cos, tan, ln, sqrt, dbinom ---
    def repl_func(m):
        func = m.group(1)
        arg = m.group(2) or m.group(3)
        return f"{func}({arg})"
    s = re.sub(r"\\(sin|cos|tan|ln|sqrt|dbinom)\{([^{}]+)\}", repl_func, s)
    s = re.sub(r"\\(sin|cos|tan|ln|sqrt|dbinom)([A-Za-z0-9\(\)])", repl_func, s)

    # Fractions
    while True:
        m = FRAC_RE.search(s)
        if not m: break
        s = s[:m.start()] + f"({m.group(1)})/({m.group(2)})" + s[m.end():]

    # Exponents
    s = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", s)   # braced
    s = re.sub(r"\^([A-Za-z0-9\+\-])", r"^(\1)", s)  # single-char
    # Subscripts
    s = re.sub(r"_\{([^{}]+)\}", r"_(\1)", s)
    s = re.sub(r"_([A-Za-z0-9])", r"_(\1)", s)

    # Operators
    s = s.replace(r"\cdot", "*").replace(r"\times", "*").replace(r"\div", "/")
    s = re.sub(r"\\boxed\s*\{([^{}]+)\}", r"(\1)", s)
    s = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", "", s)
    s = s.replace(r"\item", "- ").replace(r"\qquad", " ").replace(r"\quad", " ")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

    # Remove unknown LaTeX commands
    s = UNKNOWN_CMD_RE.sub("", s)

    # Flatten braces
    s = BRACES_RE.sub(r"(\1)", s)

    # Space fix
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace("\\","")

    return s

def extract_answer(solution_text: str) -> str:
    if not solution_text: return ""
    m = BOXED_RE.search(solution_text)
    if m: return clean_latex(m.group(1))
    for pat in ANSWER_PATTERNS:
        m = pat.search(solution_text)
        if m: return clean_latex(m.group(1).splitlines()[0])
    lines = [ln.strip() for ln in solution_text.splitlines() if ln.strip()]
    if lines and len(lines[-1]) <= 200: return clean_latex(lines[-1])
    return ""

def split_steps(solution_text: str):
    if not solution_text: return []
    raw_steps = [s.strip() for s in clean_latex(solution_text).split("\n") if s.strip()]
    return [s for step in raw_steps for s in re.split(r'(?<=[\.\?\!])\s+', step) if s]

def main():
    print("Loading MATH dataset...")
    ds = load_dataset("qwedsacf/competition_math")

    rows = []
    for ex in tqdm(ds["train"]):
        problem = ex.get("problem") or ex.get("question") or ""
        solution = ex.get("solution") or ex.get("solution_text") or ""
        question_plain = clean_latex(problem)
        answer_plain = extract_answer(solution)
        steps = split_steps(solution)
        if not steps and solution.strip():
            steps = [clean_latex(solution)]
        rows.append({"question": question_plain, "answer": answer_plain, "steps": steps})

    random.shuffle(rows)
    n_valid = int(len(rows) * VALID_RATIO)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(TRAIN_OUTFILE, "w", encoding="utf-8") as f:
        json.dump(rows[n_valid:], f, ensure_ascii=False, indent=2)
    with open(VALID_OUTFILE, "w", encoding="utf-8") as f:
        json.dump(rows[:n_valid], f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rows[n_valid:])} train and {len(rows[:n_valid])} valid examples to {OUT_DIR}")

if __name__ == "__main__":
    main()
