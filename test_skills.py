"""主智能体 skill 路径自动化测试

验证 /Users/wuming/Downloads/test/skills 下的 skill 能被主智能体正常解析：
1. 离线解析：走 SkillsMiddleware 同款加载链路（ls -> download -> 解析 frontmatter），
   校验 3 个 skill 的 name/description/path，不消耗模型调用
2. 离线读取：通过 backend 读取每个 SKILL.md 全文（即模型 read_file 的底层路径），
   验证虚拟路径 "/skills/..." 正确映射到真实目录
3. 在线端到端：真实调用一次模型，确认 skill 列表已注入系统提示词，
   且模型能通过 read_file 实际读取 skill 文件

运行：.venv/bin/python test_skills.py
"""

import os
import sys

from deepagents.middleware.skills import _list_skills

# 导入 main 即完成主智能体构建（与运行时完全相同的配置）
from main import SKILLS_DIR, agent, skills_backend

EXPECTED_SKILLS = {"deepagents-wiki", "deepseek-harness-wiki", "dify-wiki"}
SKILLS_SOURCE = "/skills"  # 与 main.py 中 create_deep_agent(skills=...) 保持一致


def test_parse_skills() -> None:
    """离线：skill 元数据解析（SkillsMiddleware.before_agent 内部同款链路）。"""
    skills = _list_skills(skills_backend, SKILLS_SOURCE)
    names = {s["name"] for s in skills}
    assert names == EXPECTED_SKILLS, f"期望 {EXPECTED_SKILLS}，实际解析到 {names}"
    for s in skills:
        assert s["description"].strip(), f"{s['name']} 缺少 description"
        assert s["path"] == f"{SKILLS_SOURCE}/{s['name']}/SKILL.md", s["path"]
    print(f"[1/3] 元数据解析通过：{len(skills)} 个 skill")
    for s in skills:
        print(f"  - {s['name']}: {s['description'][:40]}... -> {s['path']}")


def test_read_skill_files() -> None:
    """离线：按解析出的虚拟路径读取 SKILL.md 全文，验证路径映射与内容可读。"""
    skills = _list_skills(skills_backend, SKILLS_SOURCE)
    responses = skills_backend.download_files([s["path"] for s in skills])
    for s, resp in zip(skills, responses, strict=True):
        assert resp.error is None, f"读取 {s['path']} 失败：{resp.error}"
        content = resp.content.decode("utf-8")
        assert content.startswith("---"), f"{s['path']} 缺少 frontmatter"
        assert f"name: {s['name']}" in content, f"{s['path']} frontmatter name 不匹配"
        # 虚拟路径必须映射到真实磁盘文件
        real_path = SKILLS_DIR / s["name"] / "SKILL.md"
        assert real_path.is_file(), f"真实文件不存在：{real_path}"
    print(f"[2/3] 全文读取通过：{len(responses)} 个 SKILL.md 均可按虚拟路径读取")


def test_agent_end_to_end() -> None:
    """在线：真实模型调用，验证 skill 注入系统提示词且模型可读取 skill 文件。"""
    if not os.environ.get("OPENAI_API_KEY"):
        print("[3/3] 跳过：未配置 OPENAI_API_KEY")
        return
    question = (
        "你当前挂载了哪些 skills？请列出全部 skill 的名称和一句话用途；"
        "然后实际读取 deepagents-wiki 的 SKILL.md 文件，"
        "告诉我该文件 frontmatter 中 name 字段的值，证明你可以访问它。"
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content
    if not isinstance(answer, str):  # 兼容内容块列表
        answer = "".join(p.get("text", "") for p in answer if isinstance(p, dict))

    read_paths = [
        tc.get("args", {}).get("file_path", "")
        for m in result["messages"]
        for tc in (getattr(m, "tool_calls", None) or [])
        if tc.get("name") == "read_file"
    ]
    print(f"[3/3] 模型回答：\n{answer}\n")
    if read_paths:
        print(f"  模型实际 read_file 的路径：{read_paths}")

    missing = {name for name in EXPECTED_SKILLS if name not in answer}
    assert not missing, f"模型回答中缺少 skill：{missing}（skill 列表可能未注入系统提示词）"
    assert any("deepagents-wiki" in p for p in read_paths), (
        "模型未实际读取 deepagents-wiki/SKILL.md，read_file 链路未验证"
    )
    print("  端到端验证通过：skill 列表已注入，模型可按路径读取 skill 文件")


if __name__ == "__main__":
    assert SKILLS_DIR.is_dir(), f"skill 目录不存在：{SKILLS_DIR}"
    print(f"skill 目录：{SKILLS_DIR}\n")
    test_parse_skills()
    test_read_skill_files()
    test_agent_end_to_end()
    print("\n全部测试通过")
    sys.exit(0)