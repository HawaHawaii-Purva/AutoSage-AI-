# Contributing to AutoSage AI

Keep domain logic testable and auditable. New safety rules should be added as deterministic checks first, then explained by the LLM.

Before pushing:

```bash
python -m compileall .
streamlit run app.py
```
