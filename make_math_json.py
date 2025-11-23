import json, re, os, random
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "data"
TRAIN_OUTFILE = os.path.join(OUT_DIR, "math_train.json")
VALID_OUTFILE = os.path.join(OUT_DIR, "math_valid.json")
VALID_RATIO = 0.1
SEED = 42
random.seed(SEED)

# Precompile regexes
MATRIX_RE = re.compile(r"\\begin\{(pmatrix|bmatrix|Bmatrix)\}(.*?)\\end\{\1\}", re.S)
FRAC_RE = re.compile(r"\\frac\s*\{([^{}]+?)\}\s*\{([^{}]+?)\}")
SUM_RE = re.compile(r"\\sum_\{([^{}]+)\}\^\{([^{}]+)\}")
FUNC_RE = re.compile(r"\\(sin|cos|tan|ln|sqrt|dbinom)(\{([^{}]+)\}|([A-Za-z0-9θφ]))")
BRACES_RE = re.compile(r"\{([^{}]+)\}")
BOXED_RE = re.compile(r"\\boxed\s*\{([^{}]+)\}")
UNKNOWN_CMD_RE = re.compile(r"\\(?!pi\b|theta\b|phi\b|alpha\b|beta\b|gamma\b|infty\b|sin\b|cos\b|tan\b|ln\b|sqrt\b|sum\b|dbinom\b)[A-Za-z]+\*?")
MATH_DELIM_RE = re.compile(r"\${1,2}|\\\(|\\\)|\\\[|\\\]")

ANSWER_PATTERNS = [
    re.compile(r"(?i)^\s*answer[:\s]+(.+)$", flags=re.M),
    re.compile(r"(?i)final answer[:\s]+(.+)$", flags=re.M),
    re.compile(r"(?i)the answer is[:\s]+(.+)$", flags=re.M),
]

# --- Matrix conversion without recursive clean_latex ---
def convert_matrix_fast(s: str) -> str:
    def repl(m):
        rows = [row.strip() for row in m.group(2).strip().split(r"\\") if row.strip()]
        matrix_list = []
        for row in rows:
            entries = [e.strip() for e in row.split("&")]
            matrix_list.append(entries)
        return str(matrix_list)
    return MATRIX_RE.sub(repl, s)

# --- Main clean function ---
def clean_latex_fast(s: str) -> str:
    if not s: return ""

    # diagrams and symbols
    s = re.sub(r"\[asy\].*?\[/asy\]", "[diagram]", s, flags=re.S)
    s = s.replace("^\circ", "°")
    s = s.replace("&=", "=").replace("\pm", " +/- ")
    s = s.replace("\\\\", " \n")
    s = s.replace(r"\equiv", "≡")
    s = re.sub(r"\\pmod\s*\{([^{}]+)\}", r"(mod \1)", s)

    # remove math delimiters
    s = MATH_DELIM_RE.sub("", s)

    # sum replacement
    s = SUM_RE.sub(r"summation from \1 to \2", s)

    # Greek letters
    greek = {"\\theta":"theta","\\phi":"phi","\\alpha":"alpha","\\beta":"beta","\\gamma":"gamma","\\infty":"infinity","\\pi":"pi"}
    for k,v in greek.items():
        s = s.replace(k,v)

    # functions
    def func_repl(m):
        arg = m.group(3) or m.group(4)
        return f"{m.group(1)}({arg})"
    s = FUNC_RE.sub(func_repl, s)

    # fractions
    s = FRAC_RE.sub(lambda m: f"({m.group(1)})/({m.group(2)})", s)

    # exponents and subscripts
    s = re.sub(r"\^\{([^{}]+)\}", r"^(\1)", s)
    s = re.sub(r"\^([A-Za-z0-9\+\-])", r"^(\1)", s)
    s = re.sub(r"_\{([^{}]+)\}", r"_(\1)", s)
    s = re.sub(r"_([A-Za-z0-9])", r"_(\1)", s)

    # operators
    s = s.replace(r"\cdot","*").replace(r"\times","*").replace(r"\div","/")

    # boxed expressions
    s = BOXED_RE.sub(r"(\1)", s)

    # remove environments and simple LaTeX
    s = re.sub(r"\\begin\{[^}]*\}|\\end\{[^}]*\}", "", s)
    s = s.replace(r"\item", "- ").replace(r"\qquad"," ").replace(r"\quad"," ")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

    # unknown commands
    s = UNKNOWN_CMD_RE.sub("", s)

    # matrices
    s = convert_matrix_fast(s)

    # flatten braces
    s = BRACES_RE.sub(r"(\1)", s)

    # spacing
    s = s.replace(")(", ") (")
    s = re.sub(r"\s+"," ",s).strip()
    s = s.replace("\\","")
    return s

def extract_answer(solution_text: str) -> str:
    if not solution_text: return ""
    m = BOXED_RE.search(solution_text)
    if m: return clean_latex_fast(m.group(1))
    for pat in ANSWER_PATTERNS:
        m = pat.search(solution_text)
        if m: return clean_latex_fast(m.group(1).splitlines()[0])
    lines = [ln.strip() for ln in solution_text.splitlines() if ln.strip()]
    if lines and len(lines[-1])<=200: return clean_latex_fast(lines[-1])
    return ""

def split_steps(solution_text: str):
    if not solution_text: return []
    raw = [s.strip() for s in clean_latex_fast(solution_text).split("\n") if s.strip()]
    steps = [s for step in raw for s in re.split(r'(?<=[\.\?\!])\s+', step) if s]
    return steps

def main():
    print("Loading MATH dataset...")
    ds = load_dataset("qwedsacf/competition_math")

    rows = []
    for ex in tqdm(ds["train"]):
        problem = ex.get("problem") or ex.get("question") or ""
        solution = ex.get("solution") or ex.get("solution_text") or ""
        q = clean_latex_fast(problem)
        a = extract_answer(solution)
        steps = split_steps(solution)
        if not steps and solution.strip(): steps = [clean_latex_fast(solution)]
        rows.append({"question":q,"answer":a,"steps":steps})

    random.shuffle(rows)
    n_valid = int(len(rows)*VALID_RATIO)
    os.makedirs(OUT_DIR,exist_ok=True)
    with open(TRAIN_OUTFILE,"w",encoding="utf-8") as f: json.dump(rows[n_valid:],f,ensure_ascii=False,indent=2)
    with open(VALID_OUTFILE,"w",encoding="utf-8") as f: json.dump(rows[:n_valid],f,ensure_ascii=False,indent=2)
    print(f"Wrote {len(rows[n_valid:])} train and {len(rows[:n_valid])} valid examples to {OUT_DIR}")

if __name__=="__main__":
    main()
