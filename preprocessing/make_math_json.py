import json, re, os, random
from datasets import load_dataset
from tqdm import tqdm

OUT_DIR = "data"
TRAIN_OUTFILE = os.path.join(OUT_DIR, "math_train.json")
VALID_OUTFILE = os.path.join(OUT_DIR, "math_valid.json")
TEST_OUTFILE = os.path.join(OUT_DIR, "math_test.json")
VALID_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42
random.seed(SEED)

# Keep these for fallback patterns
ANSWER_PATTERNS = [
    re.compile(r"(?i)^\s*answer[:\s]+(.+)$", flags=re.M),
    re.compile(r"(?i)final answer[:\s]+(.+)$", flags=re.M),
    re.compile(r"(?i)the answer is[:\s]+(.+)$", flags=re.M),
]

def _clean_wrappers(s: str) -> str:
    """Clean only simple external wrappers ($...$, \(..\), \[..\]) and collapse spaces.
       Do NOT strip internal LaTeX content or inner braces."""
    if not s:
        return s
    s = s.strip()
    # Remove surrounding dollar delimiters $...$
    if s.startswith("$") and s.endswith("$"):
        s = s[1:-1].strip()
    # Remove \(...\) or \[...\]
    if s.startswith(r"\(") and s.endswith(r"\)"):
        s = s[2:-2].strip()
    if s.startswith(r"\[") and s.endswith(r"\]"):
        s = s[2:-2].strip()
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def _find_braced_content(text: str, open_brace_idx: int) -> str:
    """Given the index of a '{' in text, return the substring content inside the matching braces.
       Handles nested braces by counting. Returns '' if not properly closed."""
    i = open_brace_idx
    if i >= len(text) or text[i] != "{":
        return ""
    depth = 0
    start = i + 1
    j = start
    while j < len(text):
        ch = text[j]
        # skip escaped braces like '\{' or '\}' by ignoring the backslash
        if ch == "\\" and j + 1 < len(text) and text[j+1] in "{}":
            j += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            if depth == 0:
                # found the matching closing brace
                return text[start:j]
            else:
                depth -= 1
        j += 1
    # if we get here, braces were not balanced; return empty
    return ""

def extract_answer(solution_text: str) -> str:
    if not solution_text:
        return ""

    s = solution_text

    # 1) Look for \boxed or \\boxed occurrences and parse balanced braces
    # We search for literal occurrences of '\boxed' (single-escaped) or '\\boxed' (double-escaped)
    # Use a simple search rather than a brittle regex.
    for marker in [r"\\boxed", r"\\\\boxed", r"\boxed"]:
        idx = s.find(marker)
        if idx != -1:
            # find first '{' after marker
            brace_idx = s.find("{", idx + len(marker))
            if brace_idx != -1:
                content = _find_braced_content(s, brace_idx)
                if content:
                    return _clean_wrappers(content)

    # Also handle cases like "$\boxed{...}$" where dollars surround the boxed expression
    m = re.search(r"\$\s*(?:\\boxed|\\\\boxed|\\boxed)\s*\{", s)
    if m:
        # find the brace after m.end()-1
        brace_idx = s.find("{", m.start())
        if brace_idx != -1:
            content = _find_braced_content(s, brace_idx)
            if content:
                return _clean_wrappers(content)

    # 2) Try common textual answer patterns like "Answer: ..."
    for pat in ANSWER_PATTERNS:
        m = pat.search(s)
        if m:
            ans = m.group(1).strip()
            return _clean_wrappers(ans)

    # 3) Fallback: last non-empty line if short enough
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        if len(last) <= 200:
            return _clean_wrappers(last)

    return ""

def split_steps(solution_text: str):
    if not solution_text: return []
    steps = [s.strip() for s in solution_text.splitlines() if s.strip()]
    return steps

def main():
    print("Loading MATH dataset...")
    ds = load_dataset("qwedsacf/competition_math")

    rows = []
    for ex in tqdm(ds["train"]):
        problem = ex.get("problem") or ex.get("question") or ""
        solution = ex.get("solution") or ex.get("solution_text") or ""
        q = problem  # skip LaTeX conversion
        a = extract_answer(solution)
        steps = split_steps(solution)
        if not steps and solution.strip():
            steps = [solution]
        rows.append({"question": q, "answer": a, "steps": steps})

    random.shuffle(rows)

    n_valid = int(len(rows) * VALID_RATIO)
    n_test = int(len(rows) * TEST_RATIO)

    train_rows = rows[n_valid + n_test:]
    valid_rows = rows[:n_valid]
    test_rows = rows[n_valid:n_valid + n_test]

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(TRAIN_OUTFILE, "w", encoding="utf-8") as f:
        json.dump(train_rows, f, ensure_ascii=False, indent=2)
    with open(VALID_OUTFILE, "w", encoding="utf-8") as f:
        json.dump(valid_rows, f, ensure_ascii=False, indent=2)
    with open(TEST_OUTFILE, "w", encoding="utf-8") as f:
        json.dump(test_rows, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(train_rows)} train, {len(valid_rows)} valid, and {len(test_rows)} test examples to {OUT_DIR}")

if __name__ == "__main__":
    main()
