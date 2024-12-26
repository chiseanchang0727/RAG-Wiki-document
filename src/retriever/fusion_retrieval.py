from configs.rag_config import RAGConfig
from src.retriever.base import CombineRetriever
from llm.bots.qa_bot import QABot
from llm.prompts.prompts import QA_PROMPT

def fusion_retriever(target, question, chunked_data, llm: QABot, config: RAGConfig, vectorstore):


    langchain_retriever = CombineRetriever(vectorstore=vectorstore, chunked_data=chunked_data, config=config)
    file_name, file_content = langchain_retriever.get_relevant_docs(question)

    question = question
    reference = file_content

    llm_answer = llm.get_answer(target=target, question=question, reference=reference, prompt=QA_PROMPT)


    return file_name, llm_answer