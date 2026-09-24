# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

DateSATBench is a **dataset repository plus the generators that produced it**. The shipped artifacts (the `constraints.json` / `constraints.jsonl` files) are the product; the generator scripts are provenance and are only re-run to extend or regenerate a dataset. Treat the committed constraint files as data — regenerating them costs LLM API calls and changes IDs.

The benchmark is consumed by the separate **DateSAT** solver repo (not vendored here), whose `eval/run_benchmarks.py --datesatbench-repo <path>` points at either `datesatbench/` or a variant under `datesatbench_bounded/`.

## Commands

All commands run from the repository root. `datesatbench/` is a Python package, so always use module execution (`python -m datesatbench....`), never `python path/to/script.py`.

```bash
pip install -e .          # base install (no runtime deps)
pip install -e ".[llm]"   # adds python-dotenv, openai, anthropic — needed by the LLM generators
```

There is no test suite, linter config, or CI in this repo.

### Regenerating datasets

```bash
# LLM-synthesized (needs ANTHROPIC_API_KEY or OPENAI_API_KEY, or a .env at repo root)
python -m datesatbench.llm_constraints.generator.constraint_generator \
    --num 10 --tags year_vs_days --output datesatbench/llm_constraints/constraints/1.json
python -m datesatbench.llm_constraints.generator.combine_constraints   # merges per-tag files -> constraints.json

# Legal-doc (4 stages; stage 1 needs raw_data/title26.xml, which is gitignored/absent)
python -m datesatbench.legal_doc_constraints.generator.parse
python -m datesatbench.legal_doc_constraints.generator.filter \
    datesatbench/legal_doc_constraints/processed_data/parsed.jsonl \
    --output datesatbench/legal_doc_constraints/processed_data/filtered.jsonl
python -m datesatbench.legal_doc_constraints.generator.random_select
python -m datesatbench.legal_doc_constraints.generator.llm_extractor \
    --input datesatbench/legal_doc_constraints/processed_data/selected.jsonl \
    --output datesatbench/legal_doc_constraints/constraints/1.jsonl --provider anthropic

# Grammar-fuzzer (needs the `fandango` CLI on PATH; step 2 needs solver results)
python -m datesatbench.grammar_constraints.generator.generate_constraints -n 100
python -m datesatbench.grammar_constraints.generator.pick_benchmarks
python -m datesatbench.grammar_constraints.generator.merge_benchmarks
```

### Bounded variants

```bash
python tools/inject_bounds.py                                              # regenerates both default variants
python tools/inject_bounds.py --min 1950/1/1 --max 2050/12/31 --name years_1950_2050
```

`--min`, `--max`, `--name` must be given together. Output mirrors the source layout under `datesatbench_bounded/<name>/` plus a `bound_manifest.json`.

## Architecture

### Three datasets, one schema

Every entry in every dataset — synthetic, legal, grammar, bounded — is a JSON object with `declarations` (`"name: date|int|bool"` strings) and `constraints` (DateSAT DSL expression strings). That shared shape is why the same solver runner and the same `tools/inject_bounds.py` work across all three. Per-dataset extras: `description` + `coverage_tags` (LLM), `description` + `provenance`/`parsed_id`/`filtered_id` (legal), `size` + `execution_time` (grammar).

| Dataset | Source | File | Entries | ID form |
|---|---|---|---|---|
| `llm_constraints` | LLM synthesis, 5 coverage tags | `constraints/constraints.json` | 100 | `llm-<tag>-<n>` |
| `legal_doc_constraints` | US Code Title 26 XML | `constraints/constraints.jsonl` | 200 | `legal-<n>` |
| `grammar_constraints` | fandango fuzzer over `grammar.fan` | `constraints/constraints.json` | 150 | `grammar-<sat\|unsat\|timeout>-<n>` |

Each dataset directory follows the same layout: `generator/` (code), `constraints/` (shipped output), and for legal also `raw_data/` → `processed_data/` staging.

### The DateSAT DSL

Constraints are strings in a small date/period language: `Date(y, m, d)`, `Period(years, months, days)`, `Date ± Period`, `Period ± Period`, `Period * int`, comparisons, `.year`/`.month`/`.day` property access, and `&& || ! ->`. The canonical spec lives in two places — `SYSTEM_PROMPT` in `llm_constraints/generator/constraint_generator.py` and `LEGAL_EXTRACTION_PROMPT` in `legal_doc_constraints/generator/llm_extractor.py`. **These two prompts must be kept in agreement**; a DSL change means editing both, and the grammar in `grammar_constraints/generator/grammar.fan` as well.

The LLM prompts restrict generated dates to **1900-03-01 … 2100-02-28**. Grammar-generated and bounded datasets do not share that restriction.

### Generate → validate → feed-back loop

Both LLM generators are closed loops, not one-shot calls: parse JSON → check schema → check constraint counts → validate by round-tripping through `datesat.constraint_parser.ConstraintParser.generate_builder_code()` → on any failure, append a structured error message to the prompt and retry. Hard API errors (401/rate limit) fail fast instead of retrying. Every attempt is appended to a timestamped `llm_calls_<ts>.jsonl` beside the output file.

**`datesat` is an optional import that is not installed here.** When absent, `_validate_constraints_with_parser()` returns a "Missing dependency" failure, so the feedback loop will burn all retries and produce nothing. Parser-backed validation requires the DateSAT solver repo to be importable.

`datesatbench/utils/llm.py` is the single LLM entry point (`LLMClient`) for both generators: provider auto-detection (Anthropic preferred over OpenAI), extended-thinking config per provider, and the fence-stripping / smart-quote-normalizing JSON repair used on every response. Providers are gated by module-level `ENABLE_OPENAI` / `ENABLE_ANTHROPIC` flags — OpenAI is currently disabled there, so `--provider openai` errors out until that flag is flipped.

### Bounded variants

`tools/inject_bounds.py` appends `<var> >= Date(...)` / `<var> <= Date(...)` for every `date`-typed declaration and records an `injected_bound` field per entry. The bound lives in the *benchmark text*, not the solver, so an unbounded solver run (`--bound none`) can be evaluated on bounded problems. Semantics deliberately differ from solver-level bounds: only declared variables are constrained, intermediate arithmetic results stay unbounded. Regenerating a variant is purely mechanical — re-run the tool rather than hand-editing files under `datesatbench_bounded/`.

## Gotchas

- `merge_benchmarks.py` globs every `*.json` in `grammar_constraints/constraints/` and excludes only `merged_benchmarks.json` — the existing `constraints.json` gets folded into its own replacement, duplicating entries. Delete or move it before merging.
- `combine_constraints.py` does the right thing (it excludes `constraints.json`), but it **reassigns every `id`** in alphabetical file order, preserving the old one as `generated_id`. Adding a new tag file renumbers downstream IDs, breaking cross-references to prior results.
- `pick_benchmarks.py` reads `grammar_constraints/results/naive_int.json`, which is produced by the DateSAT solver repo and is not committed here.
- `legal_doc_constraints/raw_data/title26.xml` is not committed; download it from the US Code site before running `parse.py`.
- `random_select.py` uses a fixed seed (42) so the 200-record selection is reproducible — don't change it without renumbering the whole legal dataset.
