from pathlib import Path
import json
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QUERIES_FILE = PROJECT_ROOT / "config" / "master_queries.json"
OUTPUT = PROJECT_ROOT / "data_v2" / "processed" / "query_actual_counts_v3.txt"


# ============================================================
# ROMAN URDU DICTIONARY — carefully curated, NO English words
# ============================================================
URDU = {
    # Question words
    "kya", "kab", "kis", "konsa", "konsi", "konse",
    "kitne", "kitni", "kitna", "kaise", "kyun", "kahan", "kaun",
    # Postpositions
    "ka", "ki", "ke", "ko", "se", "mein", "ne", "par", "tak",
    "baad", "pehle", "liye", "sath", "bina", "baghair",
    "khilaf", "mutabiq", "wajah", "sabab", "khatir", "taur",
    "zarurat", "zaroorat", "dur", "duran", "andar", "bahar",
    "upar", "niche",
    # Pronouns / particles
    "yeh", "ye", "woh", "wo", "jo", "jab", "tab", "ab", "phir",
    "kuch", "koi", "sab", "har", "ek", "bhi", "sirf", "bas",
    "yahan", "wahan", "idhar", "udhar", "na", "ya", "nahi",
    # Adverbs / degree
    "bahut", "thora", "zyada", "kam", "az", "abhi", "kal", "aaj",
    "wala", "wali", "wale", "bilkul", "sahi", "theek", "acha",
    "behtar", "aasan", "lazmi", "darkar", "mumkin",
    "zaroori", "chahiye", "mazeed", "taake", "waste",
    # Verb forms
    "hai", "hain", "hoga", "hogi", "honge", "tha", "thi",
    "hota", "hoti", "hote", "hona", "honi", "hone", "hua", "hui", "hue",
    "karna", "karni", "karne", "karo", "kiya", "kiye", "karke",
    "karta", "karti", "karte", "kar", "karwa", "karwani", "karwane",
    "lena", "leni", "lene", "liya", "liye", "leta", "leti", "lete",
    "dena", "deni", "dene", "diya", "diye", "dein", "deta", "deti", "dete",
    "milna", "milni", "milne", "milta", "milti", "milte",
    "milega", "milegi", "milenge", "mile", "mil",
    "raha", "rahi", "rahe", "gaya", "gayi", "gaye",
    "jayega", "jayegi", "jayenge", "jaye", "ja",
    "sakta", "sakti", "sakte", "sakna",
    "jata", "jati", "jate", "jana",
    "lagta", "lagti", "lagte", "lagna", "lage", "lagi",
    "aata", "aati", "aate", "aana",
    "banta", "banti", "bante", "banna", "banti",
    "nikal", "pohanch", "pohanchna", "nikalna",
    "ban", "banne", "bana", "bani",
    # Urdu-specific content
    "parhai", "padhai", "dars", "imtihan", "talib", "ilm", "jamia",
    "sanad", "kitab", "khana", "khane", "sawari", "saalana",
    "saal", "bhar", "mahina", "hafta", "haftay", "hafton",
    "kharcha", "ada", "raqam", "meyad", "mohlat", "shikayat",
    "saza", "haq", "ijazat", "hadd", "shart", "shartein", "mukammal",
    "hasil", "poore", "pure", "batana", "bata", "samajhna",
    "dekhna", "jaanna", "maloom", "khatam", "kholna", "khulegi",
    "chhorna", "chhoro", "chhor", "chhutti", "der", "bemar", "bimari",
    "tabdeel", "dakhla", "darkhwast", "jama", "muttafiq",
    "faislay", "faisla", "pehchaan", "sazaa",
    "shuru", "shuruat", "aakhri", "tareekh", "muddat", "waqt",
    "din", "raat", "subah", "shaam",
    "naam", "kaam", "ghar", "kitaben", "kisi", "kuch",
    "jane", "dene", "lene", "rehne", "karte", "karte",
    "usay", "isay", "inko", "unko", "humein", "tumhein",
}

# ============================================================
# ENGLISH STOPWORDS — pure English
# ============================================================
ENGLISH = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "do", "does", "did", "have", "has", "had",
    "of", "in", "on", "at", "for", "to", "with", "and", "or", "but",
    "if", "then", "so", "as", "by", "from", "into", "onto",
    "this", "that", "these", "those",
    "not", "no", "yes", "all", "any", "some", "each", "every",
    "must", "should", "can", "could", "will", "would", "may", "might",
    "shall", "how", "what", "when", "where", "which", "who", "why",
    "many", "much", "more", "most", "less", "long",
    "it", "its", "he", "she", "they", "we", "you", "i", "me",
    "his", "her", "their", "our", "your", "my",
    "after", "before", "during", "between", "through", "within",
    "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "first", "second", "third",
    "student", "students", "class", "classes", "exam", "exams",
    "examination", "university", "date", "start", "deadline",
    "final", "faculty", "library", "withdraw", "withdrawal",
    "penalty", "medical", "leave", "sick", "grade", "grading",
    "degree", "graduation", "requirement", "commencement",
    "duration", "process", "processing", "threshold", "minimum",
    "maximum", "allowed", "permitted", "required", "condition",
    "appeal", "appealing", "enrollment", "adjustment",
    "borrowing", "curfew", "residence", "night", "return",
    "annual", "yearly", "per", "year", "time", "timing",
    "policy", "application", "cutoff", "cycle", "submission",
    "late", "window", "grace", "period", "assignment", "assignments",
    "short", "long", "stay", "standing", "good", "attend",
    "attendances", "attend",
}

# ============================================================
# PROPER NOUNS / NUMBER / TECHNICAL — neutral, excluded from ratio
# ============================================================
PROPER = {
    "fall", "spring", "summer", "winter",
    "software", "engineering",
    "cgpa", "gpa", "id", "pm", "am",
}

TECHNICAL = {
    "semester", "course", "courses", "credit", "hours",
    "assignment", "plagiarism", "admission", "campus",
    "thesis", "internship", "scholarship", "hostel",
    "counselling", "feedback", "evaluation",
    "alumni", "merit", "laboratory", "lab", "sports",
    "transport", "shuttle", "prerequisite", "registration",
    "attendance", "grade", "grades", "exam", "imtihan",
}


def tokenize(text):
    return re.findall(r"[A-Za-z0-9]+", text.lower())


def classify(tok):
    if tok.isdigit():
        return "number"
    if tok in URDU:
        return "urdu"
    if tok in ENGLISH:
        return "english"
    if tok in PROPER:
        return "proper"
    if tok in TECHNICAL:
        return "technical"
    # Fallback: English morphology
    if re.search(r"(tion|sion|ing|ment|ness|able|ible|ous|ive|ly|ed)$", tok):
        return "english"
    # Unknown short word — assume English (conservative)
    return "english"


def analyze(q):
    tokens = tokenize(q)
    counts = {"urdu": 0, "english": 0, "proper": 0,
              "technical": 0, "number": 0}
    tagged = []
    for t in tokens:
        cls = classify(t)
        counts[cls] += 1
        tagged.append(f"{t}({cls[:3]})")

    core = counts["urdu"] + counts["english"]
    urdu_pct = counts["urdu"] / core if core else 0

    return {
        "tokens": len(tokens),
        "counts": counts,
        "urdu_pct": urdu_pct,
        "tagged": " ".join(tagged)
    }


def score(urdu_pct, level):
    if level == "L0":
        return max(0, 100 - int(urdu_pct * 500))
    if level == "L1":
        return max(0, 100 - int(abs(urdu_pct - 0.15) * 400))
    if level == "L2":
        return max(0, 100 - int(abs(urdu_pct - 0.45) * 200))
    if level == "L3":
        return max(0, 100 - int(abs(urdu_pct - 0.80) * 150))
    if level == "L4":
        return max(0, 100 - int(abs(urdu_pct - 0.90) * 120))
    return 0


def main():
    with open(QUERIES_FILE, "r", encoding="utf-8") as f:
        config = json.load(f)

    lines = ["ACTUAL COUNT v3 (FIXED)", "=" * 100]
    stats = {lvl: {"pass": 0, "warn": 0, "fail": 0, "scores": []}
             for lvl in ["L0", "L1", "L2", "L3", "L4"]}

    for doc_id in sorted(config.keys()):
        for level in ["L0", "L1", "L2", "L3", "L4"]:
            for v, q in enumerate(config[doc_id][level], start=1):
                a = analyze(q)
                s = score(a["urdu_pct"], level)

                if s >= 80:
                    verdict = "PASS"
                elif s >= 60:
                    verdict = "WARN"
                else:
                    verdict = "FAIL"

                stats[level][verdict.lower()] += 1
                stats[level]["scores"].append(s)

                c = a["counts"]
                line = (f"{doc_id} {level} V{v} | {verdict} score={s:3d} "
                        f"| urdu={c['urdu']}/{c['urdu']+c['english']} "
                        f"({a['urdu_pct']:.0%}) "
                        f"| eng={c['english']} "
                        f"| tech={c['technical']} "
                        f"| prop={c['proper']}")
                print(line)
                lines.append(line)

                if verdict != "PASS":
                    print(f"   query: {q}")
                    print(f"   tags:  {a['tagged']}")
                    lines.append(f"   query: {q}")
                    lines.append(f"   tags:  {a['tagged']}")

    print()
    print("=" * 100)
    print("SUMMARY")
    print("=" * 100)
    for lvl in ["L0", "L1", "L2", "L3", "L4"]:
        s = stats[lvl]
        tot = s["pass"] + s["warn"] + s["fail"]
        avg = sum(s["scores"]) / len(s["scores"]) if s["scores"] else 0
        line = (f"{lvl}: PASS={s['pass']:3d} WARN={s['warn']:3d} "
                f"FAIL={s['fail']:3d} | avg={avg:.1f}")
        print(line)
        lines.append(line)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nSaved: {OUTPUT}")


if __name__ == "__main__":
    main()