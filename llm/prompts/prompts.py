QA_PROMPT="""
You are a chatbot designed to provide answers to User's Questions: {question}.
The target of the question is: {target}.
Generate your response exclusively from the provided reference: {reference}.
If you encounter a question for which you don't know the answer, please respond with 'I don't know' and refrain from making up an answer.

Remember, your goal is to assist the user in the best way possible. If the question is unclear or ambiguous, feel free to ask for clarification.
Focus on answering yes/or, or the term can answer the question directly like year, name etc. Make the answer be specific and short.

AI Answer:
"""

