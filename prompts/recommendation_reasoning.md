# Recommendation Reasoning Prompt Contract

Inputs:

- user profile
- current context
- ranked candidates
- evidence snippets for each candidate

Generation requirements:

- Explain why each selected item fits the user now.
- Mention tradeoffs when relevant.
- Do not re-rank candidates unless explicitly asked by the ranking service.
- Avoid claims not supported by item metadata or retrieved evidence.
