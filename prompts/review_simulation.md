# Review Simulation Prompt Contract

Inputs:

- user profile
- target item profile
- predicted rating
- retrieved user review examples
- locale and tone constraints

Generation requirements:

- Produce a review consistent with the predicted rating.
- Preserve the user's inferred voice style.
- Use only facts present in the item profile or retrieved evidence.
- Avoid unsupported claims about availability, safety, price, or quality.
- Return structured JSON in production usage.

