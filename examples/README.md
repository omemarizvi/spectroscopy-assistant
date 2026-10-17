# Development example scenarios (D01–D08)

Put `AFS_Examples_and_Reference.docx` **here**, not in `data/`.

## Why not in `data/`?

`data/` is the **knowledge base** — the AFS technical materials the assistant
is allowed to reason from. The D01–D08 scenarios in this document are **test
inputs** (sample customer problems), not knowledge.

If they were ingested into the knowledge base, the assistant could retrieve an
old example and echo its answer back — which looks like memorisation /
case-specific hard-coding. The challenge rules explicitly forbid hard-coding,
so we keep inputs and knowledge strictly separate.

## How these are used

Run the development scenarios through the assistant to sanity-check answer
quality before the real evaluation:

```bash
python run_dev_examples.py examples/AFS_Examples_and_Reference.docx
```

Generated answers are written to `outputs/dev/`.
