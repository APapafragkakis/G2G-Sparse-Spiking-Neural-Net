import os
import re
import csv
from glob import glob
from collections import defaultdict

LOGDIRS = [
    r"folder",
]

RE_P = re.compile(r"_p(?P<p>\d+(?:[._]\d+)?)_", re.IGNORECASE)

RE_TEST_ACC = re.compile(r"test_acc=(?P<acc>\d+(?:\.\d+)?)", re.IGNORECASE)
RE_OVERALL_FR = re.compile(r"Overall:\s*(?P<fr>\d+(?:\.\d+)?)", re.IGNORECASE)

def parse_one(path: str, source_dir: str):
    base = os.path.basename(path)

    mp = RE_P.search(base)
    p = float(mp.group("p").replace("_", ".")) if mp else None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()

    accs = [float(x) for x in RE_TEST_ACC.findall(text)]
    best_acc = max(accs) if accs else None

    mfr = RE_OVERALL_FR.search(text)
    overall_fr = float(mfr.group("fr")) if mfr else None

    if best_acc is None:
        return None

    return {
        "p": p,
        "best_test_acc": best_acc,
        "overall_fr": overall_fr,
        "file": base,
        "path": path,
        "source_dir": source_dir,
    }

def main():
    data = defaultdict(lambda: defaultdict(list))

    total_logs = 0
    for logdir in LOGDIRS:
        paths = sorted(glob(os.path.join(logdir, "*.log")))
        if not paths:
            print(f"[WARN] No .log found in: {logdir}")
            continue

        for pth in paths:
            r = parse_one(pth, source_dir=os.path.basename(os.path.normpath(logdir)))
            if not r:
                continue
            key_p = r["p"] if r["p"] is not None else -1.0
            data[r["source_dir"]][key_p].append(r)
            total_logs += 1

    if total_logs == 0:
        raise SystemExit("No parsable logs found (no 'test_acc=' in logs OR wrong folders).")

    for source_dir in sorted(data.keys()):
        print("\n" + "#" * 80)
        print(f"SOURCE: {source_dir}")
        print("#" * 80)

        for p in sorted(data[source_dir].keys()):
            label = "NO_P_IN_FILENAME" if p == -1.0 else f"{p:.2f}"
            print("\n" + "=" * 72)
            print(f"p' = {label}")
            print("=" * 72)

            best = max(data[source_dir][p], key=lambda x: x["best_test_acc"])
            fr = best["overall_fr"]
            if fr is None:
                print(f"BEST: acc={best['best_test_acc']:.4f} | file={best['file']}")
            else:
                print(f"BEST: acc={best['best_test_acc']:.4f} | FR={fr:.6f} | file={best['file']}")

            print("Files:")
            for r in sorted(data[source_dir][p], key=lambda x: x["best_test_acc"], reverse=True):
                fr_str = "NA" if r["overall_fr"] is None else f"{r['overall_fr']:.6f}"
                print(f"  acc={r['best_test_acc']:.4f}  FR={fr_str}  file={r['file']}")

    out_csv = "summary_static_ALL.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source_dir", "p", "best_test_acc", "overall_firing_rate", "file", "full_path"])
        for source_dir in sorted(data.keys()):
            for p in sorted(data[source_dir].keys()):
                label = "" if p == -1.0 else f"{p:.2f}"
                for r in data[source_dir][p]:
                    w.writerow([
                        source_dir,
                        label,
                        f"{r['best_test_acc']:.6f}",
                        "" if r["overall_fr"] is None else f"{r['overall_fr']:.6f}",
                        r["file"],
                        r["path"],
                    ])

    print(f"\nSaved: {out_csv}")

if __name__ == "__main__":
    main()
