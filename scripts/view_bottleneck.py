"""View bottleneck optimization results in a readable format.

Usage:
    python scripts/view_bottleneck.py                          # Default: results/two_bit, mmmu
    python scripts/view_bottleneck.py --results_dir results/two_bit_200
    python scripts/view_bottleneck.py --benchmark scienceqa
    python scripts/view_bottleneck.py --show_all               # Show every question
    python scripts/view_bottleneck.py --per_class 5            # 5 samples per answer class
"""

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(
    results_dir: str = "results/two_bit",
    benchmark: str = "mmmu",
    per_class: int = 3,
    show_all: bool = False,
):
    results_file = os.path.join(results_dir, f"{benchmark}_bottleneck.jsonl")
    if not os.path.exists(results_file):
        print(f"No results found at {results_file}")
        return

    # Load all results
    results = []
    by_answer = defaultdict(list)
    with open(results_file) as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            results.append(r)
            by_answer[r["ground_truth"]].append(r)

    total = len(results)
    correct = sum(r["correct"] for r in results)
    print(f"\n{'=' * 70}")
    print(f"  BOTTLENECK RESULTS: {benchmark.upper()}")
    print(f"  {total} questions, {correct}/{total} correct ({correct/total*100:.1f}%)")
    print(f"  {results[0].get('num_tokens', '?')} token(s), "
          f"train_expand={results[0].get('num_image_tokens', '?')}")
    print(f"{'=' * 70}")

    # --- Summary table ---
    print(f"\n  {'Answer':<8} {'Count':<7} {'Acc':<8} {'Avg Loss Drop':<15} {'Avg Time':<10}")
    print(f"  {'-' * 48}")
    for ans in sorted(by_answer.keys()):
        items = by_answer[ans]
        n = len(items)
        acc = sum(r["correct"] for r in items) / n * 100
        drops = [r["loss_reduction"] for r in items if r.get("loss_reduction") is not None]
        avg_drop = sum(drops) / len(drops) if drops else 0
        times = [r["optimization_time_s"] for r in items]
        avg_time = sum(times) / len(times) if times else 0
        print(f"  {ans:<8} {n:<7} {acc:>5.1f}%  {avg_drop:>+13.3f}   {avg_time:>7.1f}s")

    # --- Per-question details ---
    print(f"\n{'=' * 70}")
    print("  INDIVIDUAL TOKENS")
    print(f"{'=' * 70}")

    for ans in sorted(by_answer.keys()):
        items = by_answer[ans]
        show = items if show_all else items[:per_class]
        remaining = len(items) - len(show)

        print(f"\n  --- Answer {ans} ({len(items)} questions) ---")

        for r in show:
            qid = r["question_id"]
            q_text = r.get("question", "")[:80]
            gt = r["ground_truth"]
            pred = r["prediction"]
            status = "OK" if r["correct"] else "WRONG"
            loss_i = r.get("initial_loss", 0)
            loss_f = r.get("final_loss", 0)

            # Initial token decode
            init_decode = ""
            if r.get("initial_lm_decode"):
                top3 = []
                for tok_pos in r["initial_lm_decode"]:
                    if tok_pos:
                        top3.append(f"'{tok_pos[0][0]}'({tok_pos[0][1]:.3f})")
                init_decode = ", ".join(top3)

            # Final token decode
            final_decode = ""
            if r.get("final_lm_decode"):
                top5 = []
                for tok_pos in r["final_lm_decode"]:
                    for word, prob in tok_pos[:5]:
                        top5.append(f"'{word}'({prob:.3f})")
                final_decode = ", ".join(top5)

            # Snapshot evolution
            snap_str = ""
            if r.get("snapshots"):
                snap_parts = []
                for s in r["snapshots"]:
                    step = s["step"]
                    correct_mark = "+" if s.get("correct") else "-"
                    top1 = ""
                    if s.get("lm_head_top1"):
                        top1 = f"'{s['lm_head_top1'][0][0]}'"
                    snap_parts.append(f"s{step}:{correct_mark}{top1}")
                snap_str = " -> ".join(snap_parts)

            print(f"\n  [{qid}] {status}  gt={gt} pred={pred}  "
                  f"loss: {loss_i:.3f} -> {loss_f:.3f}")
            print(f"    Q: {q_text}...")
            print(f"    Initial token:  {init_decode}")
            print(f"    Final token:    {final_decode}")
            if snap_str:
                print(f"    Evolution:      {snap_str}")

        if remaining > 0:
            print(f"\n    ... and {remaining} more (use --show_all or --per_class {len(items)})")

    # --- Codebook analysis summary ---
    codebook_file = os.path.join(results_dir, f"{benchmark}_codebook_analysis.json")
    if os.path.exists(codebook_file):
        with open(codebook_file) as f:
            cb = json.loads(f.read())

        print(f"\n{'=' * 70}")
        print("  CODEBOOK ANALYSIS")
        print(f"{'=' * 70}")

        # LM head centroids
        if cb.get("lm_head_decoding"):
            print("\n  Class centroids (averaged token decoded through lm_head):")
            for ans, toks in sorted(cb["lm_head_decoding"].items()):
                if toks and toks[0]:
                    top5_str = ", ".join(
                        f"'{t['token']}'({t['prob']:.3f})" for t in toks[0][:5]
                    )
                    print(f"    {ans} -> {top5_str}")

        # Between-class distances
        if cb.get("between_class_distances"):
            print("\n  Between-class cosine similarities:")
            pairs = sorted(cb["between_class_distances"].items())
            for pair, d in pairs:
                bar_len = int(d["cosine_similarity"] * 30)
                bar = "#" * bar_len
                print(f"    {pair:<8} cos={d['cosine_similarity']:.3f}  {bar}")

        # Step accuracy
        if cb.get("step_accuracy_curve"):
            curve = cb["step_accuracy_curve"]
            print(f"\n  Step-accuracy curve:")
            for pt in curve:
                bar = "#" * int(pt["accuracy"] / 2)
                print(f"    step {pt['step']:>3}: {pt['accuracy']:>5.1f}% {bar}")

        # LM head evolution
        if cb.get("lm_head_evolution_by_answer"):
            print("\n  How the decoded word evolves during training:")
            for ans, steps in sorted(cb["lm_head_evolution_by_answer"].items()):
                parts = []
                for step in sorted(steps.keys(), key=int):
                    if steps[step] and steps[step][0]:
                        top = steps[step][0][0]
                        parts.append(f"s{step}:'{top['token']}'({top['fraction']:.0%})")
                if parts:
                    print(f"    {ans}: {' -> '.join(parts)}")

    print()


if __name__ == "__main__":
    import fire
    fire.Fire(main)
