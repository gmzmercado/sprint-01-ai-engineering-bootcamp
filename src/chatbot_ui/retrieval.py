# Refer to 02-RAG-Preprocessing.ipynb

import openai
from langsmith import traceable, get_current_run_tree
from qdrant_client import QdrantClient
from chatbot_ui.core.config import config

# Flow: get embedding -> retrieve context from Qdrant -> format context -> build prompt for LLM -> generate answer
# This is what naive RAG looks like.

# Start client for Qdrant
qdrant_client = QdrantClient(
    url="http://qdrant:6333",
)

# Get the embedding for the text
@traceable(
    name="embed_query",
    run_type="embedding",
    metadata={
        "ls_provider": config.EMBEDDING_MODEL_PROVIDER,
        "ls_model_name": config.EMBEDDING_MODEL,
    }
)
def get_embedding(text, model=config.EMBEDDING_MODEL):
    response = openai.embeddings.create(
        input=[text],
        model=model
    )

    # Add additional metadata usage like input and total tokens
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    
    return response.data[0].embedding

# Retrieve the data from the database
# named from retrieve_data to retrieve_context
@traceable(
    name="retrieve_top_n",
    run_type="retriever"
)
def retrieve_context(query, qdrant_client, top_k=5):
    query_embedding = get_embedding(query)
    results = qdrant_client.query_points(
        collection_name=config.QDRANT_COLLECTION_NAME,
        query=query_embedding,
        limit=10
    )

    # Obtain retrieved items and their similarity scores
    retrieved_context_ids = []
    retrieved_context = []
    similarity_scores = []

    for result in results.points:
        retrieved_context_ids.append(result.id)
        retrieved_context.append(result.payload['text'])
        similarity_scores.append(result.score)

    # return results # returns a list of points. 
    return {
        "retrieved_context_ids": retrieved_context_ids,
        "retrieved_context": retrieved_context,
        "similarity_scores": similarity_scores
    }

# Format the context
@traceable(
    name="format_retriever_context",
    run_type="prompt"
)
def process_context(context):
    formatted_context = ""
    for chunk in context["retrieved_context"]:
        formatted_context += f"- {chunk}\n"
    return formatted_context

# Prompt will get the retrieved context and the user's question
@traceable(
    name="render_prompt",
    run_type="prompt"
)
def build_prompt(context, question):

    processed_context = process_context(context)

    prompt = f"""
You are a shopping assistant that can answer questions about the products in stock.

You will be given a question and a list of context.

Instructions:
- You need to answer the question based on the provided context only.
- Never use word context and refer to it as the available products

Context:
{processed_context}

Question:
{question}
"""
    return prompt

# Now, we generate the answer
@traceable(
    name="generate_answer",
    run_type="llm",
    metadata={
        "ls_provider": config.GENERATION_MODEL_PROVIDER,
        "ls_model_name": config.GENERATION_MODEL,
    }
)
def generate_answer(prompt):
    response = openai.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
        max_tokens=500
    )

    # Add additional metadata usage like input and total tokens
    current_run = get_current_run_tree()
    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage.prompt_tokens,
            "output_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }

    return response.choices[0].message.content

@traceable(
    name="rag_pipeline"
)
def rag_pipeline(question, qdrant_client, top_k=10):
    retrieved_context = retrieve_context(question, qdrant_client, top_k)
    prompt = build_prompt(retrieved_context, question)
    answer = generate_answer(prompt)

    final_result = {
        "question": question,
        "answer": answer,
        "context": retrieved_context,
        "retrieved_context_ids": retrieved_context["retrieved_context_ids"],
        "retrieved_context": retrieved_context["retrieved_context"],
        "similarity_scores": retrieved_context["similarity_scores"]
    }

    return final_result