"""Cache proof traces from deduction-solved problems.

Runs saturate_with_trace() on all JGEX problems and stores:
  - initial facts (pre-saturation)
  - deduced facts (post - pre saturation)
  - proof path facts (non-axiom facts on proof path)
  - goal text
  - problem name

Output: JSON file with one entry per solved problem.
Reusable for Summarizer training, analysis, etc.
"""

import argparse
import json
import os
import time

import geoprover

DEFAULT_PROBLEMS_FILE = "problems/jgex_ag_231.txt"
DEFAULT_OUTPUT = "data/proof_cache.json"


def load_problems(path: str) -> list[tuple[str, str]]:
    problems = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    for i in range(0, len(lines) - 1, 2):
        problems.append((lines[i], lines[i + 1]))
    return problems


def cache_proofs(problems_file: str, output_path: str) -> None:
    problems = load_problems(problems_file)
    print(f"Loaded {len(problems)} problems from {problems_file}")

    results = []
    solved = 0
    skipped = 0
    t0 = time.time()

    for i, (name, definition) in enumerate(problems):
        problem_text = f"{name}\n{definition}"
        t_start = time.time()
        try:
            state = geoprover.parse_problem(problem_text)
            initial_facts = sorted(state.facts_as_text_list())
            goal_text = state.goal_as_text()

            proved, trace = geoprover.saturate_with_trace(state)
            elapsed_ms = (time.time() - t_start) * 1000

            if not proved:
                print(f"  [{i+1}/{len(problems)}] {name}: not solved ({elapsed_ms:.0f}ms)")
                continue

            solved += 1
            post_facts = sorted(state.facts_as_text_list())
            deduced_facts = sorted(set(post_facts) - set(initial_facts))

            proof_path = trace.proof_path_facts()
            if proof_path is None:
                proof_path = []

            axioms = trace.axiom_facts()

            # Extract full proof steps for analysis
            proof_steps = trace.extract_proof()
            if proof_steps:
                num_proof_steps = len(proof_steps)
                num_axiom_steps = sum(
                    1 for _, rule, _ in proof_steps if rule == "Axiom"
                )
                num_derived_steps = num_proof_steps - num_axiom_steps
            else:
                num_proof_steps = 0
                num_axiom_steps = 0
                num_derived_steps = 0

            entry = {
                "name": name,
                "definition": definition,
                "goal": goal_text,
                "initial_facts": initial_facts,
                "deduced_facts": deduced_facts,
                "proof_path_facts": proof_path,
                "axiom_facts": axioms,
                "num_initial": len(initial_facts),
                "num_deduced": len(deduced_facts),
                "num_proof_path": len(proof_path),
                "num_total_post": len(post_facts),
                "num_proof_steps": num_proof_steps,
                "num_derived_steps": num_derived_steps,
                "num_axiom_steps": num_axiom_steps,
                "elapsed_ms": elapsed_ms,
            }
            results.append(entry)

            print(f"  [{i+1}/{len(problems)}] {name}: SOLVED "
                  f"(init={len(initial_facts)}, deduced={len(deduced_facts)}, "
                  f"proof_path={len(proof_path)}, {elapsed_ms:.0f}ms)")

        except Exception as e:
            skipped += 1
            print(f"  [{i+1}/{len(problems)}] {name}: ERROR {e}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Solved: {solved}/{len(problems)}")
    print(f"  Skipped/errors: {skipped}")

    # Stats
    if results:
        deduced_counts = [r["num_deduced"] for r in results]
        pp_counts = [r["num_proof_path"] for r in results]
        print(f"\nDeduced facts per problem:")
        print(f"  min={min(deduced_counts)}, max={max(deduced_counts)}, "
              f"mean={sum(deduced_counts)/len(deduced_counts):.0f}, "
              f"median={sorted(deduced_counts)[len(deduced_counts)//2]}")
        print(f"Proof path facts per problem:")
        print(f"  min={min(pp_counts)}, max={max(pp_counts)}, "
              f"mean={sum(pp_counts)/len(pp_counts):.0f}, "
              f"median={sorted(pp_counts)[len(pp_counts)//2]}")
        # Ratio: what fraction of deduced facts are on proof path
        ratios = [r["num_proof_path"] / r["num_deduced"]
                  for r in results if r["num_deduced"] > 0]
        if ratios:
            print(f"Proof path / deduced ratio:")
            print(f"  min={min(ratios):.3f}, max={max(ratios):.3f}, "
                  f"mean={sum(ratios)/len(ratios):.3f}")
        step_counts = [r["num_proof_steps"] for r in results]
        derived_counts2 = [r["num_derived_steps"] for r in results]
        print(f"Proof steps (total):")
        print(f"  min={min(step_counts)}, max={max(step_counts)}, "
              f"mean={sum(step_counts)/len(step_counts):.0f}, "
              f"median={sorted(step_counts)[len(step_counts)//2]}")
        print(f"Proof steps (derived only):")
        print(f"  min={min(derived_counts2)}, max={max(derived_counts2)}, "
              f"mean={sum(derived_counts2)/len(derived_counts2):.0f}, "
              f"median={sorted(derived_counts2)[len(derived_counts2)//2]}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {len(results)} proof caches to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache proof traces")
    parser.add_argument("--problems", default=DEFAULT_PROBLEMS_FILE)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    cache_proofs(args.problems, args.output)
