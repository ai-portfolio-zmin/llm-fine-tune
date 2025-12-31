Template = """
You are a retrieval router for mixed search.

You are given a query and associated features, choose the best route from:
- "bm25"
- "dense"
- "hybrid"

Return Only a json object of the form: {{"route":"<bm25|dense|hybrid>"}}

Query: {query}
Features: {features}
Answer:
"""

