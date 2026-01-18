def generate_answer(context, query):
    prompt = f"""
Context:
{''.join(context)}

Question:
{query}

Answer:
"""
    # Placeholder for LLM (OpenAI / local LLM)
    return "Based on the context, " + context[0]

