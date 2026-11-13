# The datasets

Three datasets with one schema. 
Each folder holds the shipped instances (`constraints/`) and the code that produced them (`generator/`); 
its own README explains how to regenerate it.

| Folder | Source | File | Instances |
|---|---|---|---|
| `llm_constraints/` | Synthesized by an LLM, over 5 coverage tags | `constraints/constraints.json` | 100 |
| `legal_doc_constraints/` | Extracted by an LLM from US Code Title 26 | `constraints/constraints.jsonl` | 200 |
| `grammar_constraints/` | Sampled from a formal grammar with the fandango fuzzer | `constraints/constraints.json` | 500 |

## An instance

```json
{
  "id": "llm-logical_operators-1",
  "declarations": ["x: date", "y: date"],
  "constraints": ["x == Date(2000, 2, 29)", "y == x + Period(4, 0, 0)", "y.day == 29"]
}
```

- `declarations`: variables, each `date`, `int` or `bool`.
- `constraints`: expressions in the DateSAT language: `Date(y, m, d)`, `Period(years, months, days)`, `Date ± Period`, `Period * int`, comparisons, `.year` / `.month` / `.day`, and `&& || ! ->`.
- Each dataset adds fields of its own, such as `description`, `coverage_tags` (LLM), `provenance` (legal) and `size` (grammar).

## Bounds

Every date variable is bounded to years 1 to 9999 by constraints in the instance itself (`x >= Date(1, 1, 1)`, `x <= Date(9999, 12, 31)`), recorded in its `injected_bound` field.
`utils/bounds.py` adds, changes or removes them in place:

```bash
python -m datesatbench.utils.bounds inject
python -m datesatbench.utils.bounds remove
```

## Other files

- `utils/llm.py`: the LLM client both LLM-based generators use.
- `legal_doc_constraints/processed_data/`: the text of US Code Title 26 as the legal pipeline parsed it (`parsed.jsonl`), filtered it (`filtered.jsonl`) and selected from it (`selected.jsonl`). Only the original XML, `raw_data/title26.xml`, is not included; the folder's README says where to download it.
