"""
RAG prompts: system + user with context.
"""
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

SYSTEM_RAG = """You are an AML/KYC assistant. Answer only using the provided context (complaint and regulatory excerpts). If the context does not contain enough information, say so. Do not invent facts or citations. Quote or cite the context when relevant."""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_RAG),
    ("human", """Context:
{context}

Question: {question}

Answer (based only on the context above):"""),
])
