<div align="center">
  <img src="minicrab-logo.png" alt="minicrab" width="620" />
  <p>一只帮你聊天、记事、查资料的小螃蟹。</p>
</div>

# minicrab

这是一个持续改进和使用的个人 AI 助手。平时用来聊天、整理资料、查询本地文档，也可以调用工具处理文件和设置提醒。

目前主要在完善记忆、长对话和工具权限这些实际使用中比较容易出问题的部分，不追求做成一个大而全的平台。

## 能做什么

- **聊天**：命令行和本地网页都能用，也保留了飞书、微信等渠道接入。
- **记忆**：区分用户偏好、长期工作记忆和当前会话摘要，支持明确地记住、纠正和忘记。
- **长对话**：压缩旧消息，保留近期内容、文件和技能线索，减少长任务中丢失上下文的情况。
- **文档检索**：上传、索引并查询文档，支持本地嵌入模型和兼容的远程服务。
- **工具与提醒**：文件操作、命令执行、联网查询、定时提醒和 MCP 扩展；主对话中的敏感操作会先请求确认。

## 本地运行

需要 Python 3.11 或以上。先从源码安装，当前版本没有发布到 PyPI。

```bash
git clone https://github.com/1844278557/minicrab.git
cd minicrab
git switch memory-redesign-v1
```

当前改进在 `memory-redesign-v1` 分支，尚未合并到 `main`。

Windows PowerShell：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
minicrab onboard
```

macOS / Linux：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
minicrab onboard
```

如果需要使用 WhatsApp 桥接，还需要 Node.js 20 或以上；命令行和网页聊天不需要。

### 配置模型

初始化会显示配置文件的位置。新环境默认使用 `~/.minicrab/config.json` 和 `~/.minicrab/workspace`。如果已有 `~/.nanobot/config.json` 且尚无新配置，会继续读取旧配置和数据，不需要手动搬动。

在生成的配置里填写自己的模型和 API Key。例如，使用兼容 OpenAI 接口的服务时，可以修改下面这些字段，其他已有配置保留：

```json
{
  "agents": {
    "defaults": {
      "provider": "custom",
      "model": "你的模型名称"
    }
  },
  "providers": {
    "custom": {
      "apiKey": "你的 API Key",
      "apiBase": "https://你的模型服务地址/v1"
    }
  }
}
```

配置好后：

```bash
minicrab agent
minicrab agent -m "你好，介绍一下你能做什么"
minicrab status
```

### 浏览器聊天

在配置中启用 Web 渠道：

```json
{
  "channels": {
    "web": {
      "enabled": true,
      "host": "127.0.0.1",
      "port": 18790,
      "streaming": true
    }
  }
}
```

保留前面的模型配置，然后运行：

```bash
minicrab gateway
```

打开 [本地聊天页面](http://127.0.0.1:18790/web)。网页默认只监听本机，支持聊天记录、Markdown 和操作确认。可以让助手索引并检索本地文档。

如果想单独使用一套配置，可以通过 `--config` 指定文件，例如 `minicrab gateway --config ./config.local.json`。不要把包含 API Key 的配置提交到仓库。

## 其他使用方式

命令行也可以通过 `python -m minicrab` 启动。代码、插件入口和环境变量使用 `minicrab` / `MINICRAB_`，旧入口保留兼容。

Docker 也可以从当前源码构建：`docker build -t minicrab .`。容器运行仍需挂载自己的配置目录，不包含个人配置或聊天数据。

## 项目来源

minicrab 基于 [HKUDS/nanobot](https://github.com/HKUDS/nanobot) 开发，当前版本主要围绕记忆管理、会话压缩、权限确认、Web 聊天和文档检索继续改进。保留原项目的版权声明，许可证见 [LICENSE](LICENSE)。
