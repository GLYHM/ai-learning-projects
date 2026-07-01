"""
LangChain RAG 示例 — 对照手写版 07&08-rag 理解

你手写的 RAG 做了什么：
  1. 扫描 .md 文件，按 ## 标题切块
  2. all-MiniLM-L6-v2（ONNX）向量化
  3. 存入 ChromaDB
  4. 用户提问 → 向量搜索 Top3 → 拼 Prompt → LLM 回答

LangChain 帮你做了什么：
  1. DirectoryLoader + RecursiveCharacterTextSplitter（一行代码搞定加载+切分）
  2. HuggingFaceEmbeddings 直接包 sentence-transformers（不用手动调 ONNX）
  3. Chroma 一行搞定存取查
  4. | 管道符把检索+拼接+LLM 串成流水线

运行方式:
  set DEEPSEEK_API_KEY=你的Key
  set HF_ENDPOINT=https://hf-mirror.com
  python 11-langchain-rag.py
"""
import os, sys
sys.stdout.reconfigure(encoding="utf-8")

# 国内镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    temperature=0,
)

# ==== 1. 加载文档 ====
from langchain_community.document_loaders import DirectoryLoader, TextLoader

loader = DirectoryLoader(
    os.path.expanduser("~/Documents/ObsidianVault/AI学习"),
    glob="*.md",
    loader_cls=TextLoader,
    loader_kwargs={"encoding": "utf-8"},
    show_progress=True,
)
docs = loader.load()
print(f"加载了 {len(docs)} 个文档")

# ==== 2. 切分文本 ====
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n## ", "\n### ", "\n# ", "\n", "。", ".", " "],
)
chunks = splitter.split_documents(docs)
print(f"切成了 {len(chunks)} 个文本块")

# ==== 3. 向量化 + 存入 ====
# 方案A：中文模型 shibing624/text2vec-base-chinese（推荐，本地运行）
# 方案B：英文模型 sentence-transformers/all-MiniLM-L6-v2（备选，搜中文效果差）
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

NAME = "shibing624/text2vec-base-chinese"  # 换成英文模型试试对比
embeddings = HuggingFaceEmbeddings(
    model_name=NAME,
    model_kwargs={"device": "cpu"},
)

vectordb = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./lc_chroma_db",
)
print(f"向量库已构建，{vectordb._collection.count()} 条记录")

# ==== 4. LangChain 管道：检索 + 拼接 + LLM ====
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

template = """你是一个知识库助手。根据以下资料回答问题。如果资料中找不到答案，就说"资料中未提及"。

资料：
{context}

问题：{question}

回答："""

prompt = ChatPromptTemplate.from_template(template)

chain = (
    {
        "context": vectordb.as_retriever(search_kwargs={"k": 3}),
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ==== 5. 测试 ====
print("\n" + "=" * 50)
print("测试 RAG 问答")
print("=" * 50)

questions = [
    "什么是 RAG？",
    "Agent 循环有几个步骤？",
    "笔记生成器支持哪些输入方式？",
]

for q in questions:
    print(f"\n问: {q}")
    answer = chain.invoke(q)
    print(f"答: {answer[:300]}")
