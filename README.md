# DateSATBench Datasets

**Note:** All commands assume you are running from the repository root directory.

This repository contains DateSATBench datasets and generators for producing additional DateSATBench constraints.

## DateSATBench

DateSATBench contains 3 curated datasets. Please see the paper for details.

- **LLM-synthesized constraints**: See `datesatbench/llm_constraints/README.md`
- **Legally grounded constraints**: See `datesatbench/legal_doc_constraints/README.md`
- **Grammar-fuzzer constraints**: See `datesatbench/grammar_constraints/`

## Python imports / running scripts

The `datesatbench/` directory is a Python package. If you run the generator scripts, prefer module execution, e.g.:

```bash
python -m datesatbench.llm_constraints.generator.constraint_generator --help
```