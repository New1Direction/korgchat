# KorgChat

**首個建立在 Korg 認知帳本之上的聊天產品。**

[English](README.md) · [简体中文](README.zh-CN.md) · **繁體中文**

KorgChat 的每一輪對話都會作為一條 `AgentToolCall` 事件,寫進那本驅動 korgex 代理循環、MCP 工作階段瀏覽器與 korg-tui 中 Ctrl-R 回退的同一個 `.korg/journal.json`。對話透過 `korg_bridge`(PyO3 擴充)**同步、在行程內**寫入——無需任何 HTTP 伺服器。

```
You: 寫一句關於帳本的俳句
Korg: pages turn forward / each entry signed by the past / time leaves no escape

[已記錄:第 1 輪,seq=1(使用者)→ seq=2(助理)]
```

對話結束後,這本帳本可以:
- 用 `korg-tui` + `Ctrl-R` 重放與回退
- 透過 `mcp_server.py` 以 MCP 暴露,供其他用戶端瀏覽
- 被 `korgex` 當作因果上下文,用於後續的編程任務

## 安裝

```bash
# 1. 從 Korg 工作區安裝 korg-bridge
cd /path/to/Korg/crates/korg-bridge
maturin develop

# 2. 安裝 korgchat
cd /path/to/KorgChat
pip install -e .

# 3.(可選)即時 LLM 模式所需的 Anthropic SDK
pip install -e .[anthropic]
export ANTHROPIC_API_KEY=sk-...
```

> 在中國大陸?用國內 pip 鏡像更快——見 korgex 文件的[在中國安裝](https://korgex-docs.pages.dev/zh-TW/docs/install-china)。

## 執行

```bash
korgchat --mock            # 確定性的模擬模式——無需金鑰、不連網
korgchat                   # 即時模式(Anthropic)
korgchat --journal ./my-conversation.json --mock   # 自訂帳本位置
korgchat --mock --no-stream                         # 關閉串流輸出
```

## 主要特性

KorgChat 的每個功能都建立在同一本本機、可稽核的帳本上(完整範例見[英文 README](README.md)):

- **串流輸出**:助理回覆逐字元即時列印;帳本契約不變——每輪仍只產生一條包含完整回覆的 `llm_inference` 事件。
- **自動上下文(`--auto-context`)**:每輪新提問都會自動對帳本做一次語義 `/recall`,把最相關的歷史作為前言注入——像 ChatGPT 的「記憶」,但本機、可見、可稽核。帳本裡記錄的仍是**原始**提問,而非增強後的。
- **語義檢索(`/recall`)**:裝上可選的 `fastembed` 後,按「含義」而非字面比對——「對 borrowing 感到困惑」也能找到討論 Rust 借用檢查器的那幾輪。
- **摘要(`/summarize`)**:把帳本的一段切片餵回模型並列印一份摘要;預設暫時(不寫帳本),`--save` 可把摘要也記成一條 `summary` 事件。
- **分支(`/fork` / `/checkout` / `/branches`)**:對話分支是帳本裡的一個書籤(名字 + 分叉處的 seq + 當前 tip),讓你並排嘗試不同思路而不丟失原始線索;事件本身不變,分支資訊存於 `.korg/branches.json`。
- **檢索歷史(`/recall`)**:對完整事件日誌(使用者提問、模型回覆、工具呼叫、工具結果)做本機子字串檢索,開放檔案格式,完全本機優先。
- **內建工具**:三個確定性工具 `echo` / `add` / `get_time`;在 `--mock` 模式下可用 `[tool:NAME(arg=value)]` 標記確定性觸發。作為函式庫嵌入時可傳入你自己的 `ToolRegistry`。

`/help` 會列出所有斜線指令。

## 因果鏈

KorgChat 每輪至少寫 2 條事件(模型呼叫工具時更多)。`user_prompt → llm_inference → tool_call*` 的形態與 korgex 一致,所以同一本帳本既能承載互動式聊天,也能承載自主代理的執行,而不丟失因果連貫性。

## 授權

MIT。
