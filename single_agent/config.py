# config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ==================== DeepSeek 配置 ====================
DEEPSEEK_API_KEY = "sk-5c712263b8c84fd9a4aa79947f10dd15"  # os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"  # deepseek-chat 支持工具调用，deepseek-reasoner 不支持
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"

# ==================== MySQL 配置 ====================
# 格式：mysql+pymysql://用户名:密码@主机:端口/数据库名
MYSQL_CONFIG = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "port": int(os.getenv("MYSQL_PORT", 3307)),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "RootPass123!"),
    "database": os.getenv("MYSQL_DATABASE", "agent_db"),
}

DATABASE_URI = f"mysql+pymysql://{MYSQL_CONFIG['user']}:{MYSQL_CONFIG['password']}@{MYSQL_CONFIG['host']}:{MYSQL_CONFIG['port']}/{MYSQL_CONFIG['database']}?charset=utf8mb4"

# ==================== Agent 配置 ====================
# 记忆 token 限制（超过则自动总结旧对话）
MEMORY_MAX_TOKEN_LIMIT = 2000

# Agent 最大推理次数
MAX_ITERATIONS = 8

# 是否打印详细思考过程
VERBOSE = True

# 当前模拟用户 ID（实际可从会话上下文获取）
CURRENT_USER_ID = "user_001"