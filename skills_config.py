"""主智能体本地 skills 配置 —— 共享定义（单一事实来源）

main.py（CLI 演示）与 server.py（网页聊天）共用同一份配置：
把 /Users/wuming/Downloads/test/skills 挂载为主智能体的 skill 目录。

skills 源路径相对 backend 根目录解析：以 skill 目录的父目录为虚拟根，
"/skills" 即映射到 SKILLS_DIR；virtual_mode=True 把文件访问沙箱限制在根目录内。
"""

from pathlib import Path

from deepagents.backends import FilesystemBackend

# 主智能体的 skill 目录：每个子目录是一个 skill（内含 SKILL.md）
SKILLS_DIR = Path("/Users/wuming/Downloads/test/skills")

# 传给 create_deep_agent(skills=...) 的源路径（虚拟路径，相对 backend 根目录）
SKILLS_SOURCE = "/skills"

# 无状态只读配置，可被多个智能体实例共享
skills_backend = FilesystemBackend(root_dir=SKILLS_DIR.parent, virtual_mode=True)