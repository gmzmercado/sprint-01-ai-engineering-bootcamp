import os

from chatbot_ui.retrieval import rag_pipeline
from chatbot_ui.core.config import config

from langsmith import Client

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Updated imports for RAGAS v0.2.15
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from ragas import SingleTurnSample 
from ragas.metrics import Faithfulness
from ragas.metrics import ResponseRelevancy
from ragas.metrics import LLMContextPrecisionWithoutReference
from ragas.metrics import LLMContextRecall
from ragas.metrics import NonLLMContextRecall

# Import QdrantClient (was missing from original code)
from qdrant_client import QdrantClient

ragas_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o"))
ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

qdrant_client = QdrantClient(
    url="http://localhost:6333",
)

# `run` is the RAG execution, and `example` is the reference
# How do we extract the required data?
# This specific metric looks for how well the answer takes in the context provided
async def ragas_faithfulness(run, example):
    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )
    scorer = Faithfulness(llm=ragas_llm)
    return await scorer.single_turn_ascore(sample)

# This specific metric looks for how well the answer is relevant to the question
async def ragas_response_relevancy(run, example):
    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )
    scorer = ResponseRelevancy(llm=ragas_llm, embeddings=ragas_embeddings)
    return await scorer.single_turn_ascore(sample)

# This specific metric looks for how well the answer is relevant to the question, but it uses the LLM to do the evaluation
# Higher chunks mean fewer relevant chunks
async def ragas_context_precision(run, example):
    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )
    scorer = LLMContextPrecisionWithoutReference(llm=ragas_llm)
    return await scorer.single_turn_ascore(sample)


async def ragas_context_recall_llm_based(run, example):
    sample = SingleTurnSample(
            user_input=run.outputs["question"],
            response=run.outputs["answer"],
            reference=example.outputs["ground_truth"],
            retrieved_contexts=run.outputs["retrieved_context"]
        )
    scorer = LLMContextRecall(llm=ragas_llm)
    return await scorer.single_turn_ascore(sample)

async def ragas_context_recall_non_llm(run, example):
    sample = SingleTurnSample(
            retrieved_contexts=run.outputs["retrieved_context"],
            reference_contexts=example.outputs["contexts"]
        )
    scorer = NonLLMContextRecall()
    return await scorer.single_turn_ascore(sample)

ls_client = Client(api_key=os.environ["LANGSMITH_API_KEY"])

results = ls_client.evaluate(
    lambda x: rag_pipeline(x["question"], qdrant_client),
    data="rag-evaluation-dataset",
    evaluators=[
        ragas_faithfulness,
        ragas_response_relevancy,
        ragas_context_precision,
        ragas_context_recall_llm_based,
        ragas_context_recall_non_llm,
    ], # list of functions that will be executed
    experiment_prefix="rag-evaluation-dataset",
)