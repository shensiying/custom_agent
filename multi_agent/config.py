# config.py — 多 Agent 共享配置
import os
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = "sk-5c712263b8c84fd9a4aa79947f10dd15"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3307)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "RootPass123!"),
    "database": os.getenv("MYSQL_DATABASE", "agent_db"),
}

DATABASE_URI = (
    f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}"
    f"@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}?charset=utf8mb4"
)

RAG_BASE_URL = "http://127.0.0.1:8001"
TOP_K = 3
MAX_ITERATIONS = 8
CURRENT_USER_ID = "user_001"

# ============================================================
# 记忆系统配置
# ============================================================

# PostgreSQL（PostgresSaver 短期记忆 + PostgresStore 长期记忆）
PG_HOST = "localhost"
PG_PORT = 5432
PG_USER = "root"
PG_PASSWORD = "Yxxxxxxxx"
PG_DATABASE = "mydb"

# 短期记忆参数（token 阈值方案）
MAX_RECENT_ROUNDS = 6                    # 保留最近 N 轮对话（兜底用）
SUMMARIZE_TOKEN_THRESHOLD = 3000         # 消息总 token 数超过此值触发概括
TARGET_TOKENS_AFTER_SUMMARY = 1500       # 概括后目标 token 数（摘要 + 最近消息）
MAX_SUMMARY_TOKENS = 200                 # 摘要本身的目标长度（token）
TOKENIZER_MODEL = "cl100k_base"          # tiktoken 编码（兼容 DeepSeek）
