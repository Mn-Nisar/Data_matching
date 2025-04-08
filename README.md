About
This intelligent data matching solution is designed to match two datasets using Large Language Models (LLMs) to enhance matching accuracy.

Steps Involved
Data Cleaning – Removed whitespaces and duplicate entries.

Initial Matching – Performed basic data matching using the Pandas library.

LLM-Based Matching – Utilized AWS Bedrock with the amazon.titan-embed-text model to generate text embeddings and calculate similarity scores between values.

DataFrame Update – Updated the DataFrame with the matched results.

Factor Data Merging – Applied the same matching techniques to merge with additional factor data.
