from retrieve import retrieve
from generate import generate_answer

def rag_pipeline(query):
    context = retrieve(query)
    return generate_answer(context, query)

