# KorgChat

**首个建立在 Korg 认知账本之上的聊天产品。**

[English](README.md) · **简体中文** · [繁體中文](README.zh-TW.md)

KorgChat 的每一轮对话都会作为一条 `AgentToolCall` 事件,写进那本驱动 korgex 智能体循环、MCP 会话浏览器与 korg-tui 中 Ctrl-R 回退的同一个 `.korg/journal.json`。对话通过 `korg_bridge`(PyO3 扩展)**同步、在进程内**写入——无需任何 HTTP 服务。

```
You: 写一句关于账本的俳句
Korg: pages turn forward / each entry signed by the past / time leaves no escape

[已记录:第 1 轮,seq=1(用户)→ seq=2(助手)]
```

对话结束后,这本账本可以:
- 用 `korg-tui` + `Ctrl-R` 重放与回退
- 通过 `mcp_server.py` 以 MCP 暴露,供其他客户端浏览
- 被 `korgex` 当作因果上下文,用于后续的编程任务

## 安装

```bash
# 1. 从 Korg 工作区安装 korg-bridge
cd /path/to/Korg/crates/korg-bridge
maturin develop

# 2. 安装 korgchat
cd /path/to/KorgChat
pip install -e .

# 3.(可选)实时 LLM 模式所需的 Anthropic SDK
pip install -e .[anthropic]
export ANTHROPIC_API_KEY=sk-...
```

> 在中国大陆?用国内 pip 镜像更快——见 korgex 文档的[在中国安装](https://korgex-docs.pages.dev/zh-CN/docs/install-china)。

## 运行

```bash
korgchat --mock            # 确定性的模拟模式——无需密钥、不联网
korgchat                   # 实时模式(Anthropic)
korgchat --journal ./my-conversation.json --mock   # 自定义账本位置
korgchat --mock --no-stream                         # 关闭流式输出
```

## 主要特性

KorgChat 的每个功能都建立在同一本本地、可审计的账本上(完整示例见[英文 README](README.md)):

- **流式输出**:助手回复逐字符实时打印;账本契约不变——每轮仍只产生一条包含完整回复的 `llm_inference` 事件。
- **自动上下文(`--auto-context`)**:每轮新提问都会自动对账本做一次语义 `/recall`,把最相关的历史作为前言注入——像 ChatGPT 的“记忆”,但本地、可见、可审计。账本里记录的仍是**原始**提问,而非增强后的。
- **语义检索(`/recall`)**:装上可选的 `fastembed` 后,按“含义”而非字面匹配——“对 borrowing 感到困惑”也能找到讨论 Rust 借用检查器的那几轮。
- **摘要(`/summarize`)**:把账本的一段切片喂回模型并打印一份摘要;默认临时(不写账本),`--save` 可把摘要也记成一条 `summary` 事件。
- **分支(`/fork` / `/checkout` / `/branches`)**:对话分支是账本里的一个书签(名字 + 分叉处的 seq + 当前 tip),让你并排尝试不同思路而不丢失原始线索;事件本身不变,分支信息存于 `.korg/branches.json`。
- **检索历史(`/recall`)**:对完整事件日志(用户提问、模型回复、工具调用、工具结果)做本地子串检索,开放文件格式,完全本地优先。
- **内置工具**:三个确定性工具 `echo` / `add` / `get_time`;在 `--mock` 模式下可用 `[tool:NAME(arg=value)]` 标记确定性触发。作为库嵌入时可传入你自己的 `ToolRegistry`。

`/help` 会列出所有斜杠命令。

## 因果链

KorgChat 每轮至少写 2 条事件(模型调用工具时更多)。`user_prompt → llm_inference → tool_call*` 的形态与 korgex 一致,所以同一本账本既能承载交互式聊天,也能承载自主智能体的运行,而不丢失因果连贯性。

## 许可证

MIT。
