<div align="center">
  <img src="minicrab-logo.png" alt="minicrab" width="620" />
  <p>聊天、记事、查资料，一个助手随时接上。</p>
</div>

# minicrab

minicrab 是支持命令行和微信的个人 AI 助手，提供连续对话、长期记忆、文档检索、工具调用和定时提醒。会话摘要与文件线索帮助长对话保持连续，用户偏好和工作信息分别保存，文档内容通过知识库检索参与回答。

## 功能

| 功能 | 用途 |
| --- | --- |
| CLI 对话 | 交互聊天、单次提问，按会话名称保存不同主题。 |
| 微信对话 | 扫码登录，在微信中收发文字、图片和文件。 |
| 长期记忆 | 保存用户偏好和工作信息，支持记住、修正和忘记指定内容。 |
| 文档检索 | 索引 PDF、Word、Markdown 等文档，围绕文档内容提问。 |
| 工具调用 | 读写文件、执行命令、联网查询，通过 MCP 接入其他工具。 |
| 定时提醒 | 创建提醒、查询任务，在对话渠道接收通知。 |
| 权限确认 | 敏感操作执行前提供确认入口，支持单次授权和临时授权。 |

## 安装

需要 Python 3.11 或以上和可访问的模型服务。

```bash
git clone --branch memory-redesign-v1 https://github.com/1844278557/minicrab.git
cd minicrab
```

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[weixin]"
minicrab onboard
```

如果 PowerShell 未启用脚本执行，可直接使用虚拟环境中的程序：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[weixin]"
.\.venv\Scripts\minicrab.exe onboard
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[weixin]"
minicrab onboard
```

`onboard` 创建配置和工作区。再次运行时，输入 `N` 可保留现有设置并补齐配置字段；输入 `y` 会恢复默认设置。

## 配置模型

默认配置文件：

- Windows：`%USERPROFILE%\.minicrab\config.json`
- macOS / Linux：`~/.minicrab/config.json`

Windows 可直接打开：

```powershell
notepad "$env:USERPROFILE\.minicrab\config.json"
```

使用兼容 OpenAI 接口的模型服务时，在现有配置中填写以下字段，其他节点保持不变：

```json
{
  "agents": {
    "defaults": {
      "provider": "custom",
      "model": "填写模型名称",
      "timezone": "Asia/Shanghai"
    }
  },
  "providers": {
    "custom": {
      "apiKey": "填写 API Key",
      "apiBase": "https://api.example.com/v1"
    }
  }
}
```

| 字段 | 填写内容 |
| --- | --- |
| `provider` | 使用兼容接口时填写 `custom`；也可填写已配置的服务名称，例如 `siliconflow`、`deepseek`。 |
| `model` | 模型服务提供的模型标识，需与该服务控制台中的名称一致。 |
| `timezone` | 对话时间和提醒使用的时区，例如 `Asia/Shanghai`。 |
| `apiKey` | 对应模型服务的 API Key。 |
| `apiBase` | 对应服务的接口根地址，替换示例中的 `https://api.example.com/v1`。 |

配置完成后检查状态：

```bash
minicrab status
```

配置、会话、记忆和登录数据统一保存在用户目录下的 `.minicrab` 中，默认工作区为 `.minicrab/workspace`。

## CLI 使用

### 交互聊天

```bash
minicrab agent
```

出现 `You:` 后输入消息，回车发送。可连续提问，例如先说明任务，再补充文件位置或处理要求。输入 `exit` 或 `quit` 结束命令行聊天。

### 单次提问

```bash
minicrab agent -m "列出当前工作区中的文件，并说明各自的用途"
```

### 按主题保存会话

```bash
minicrab agent --session cli:work
minicrab agent --session cli:reading
```

再次使用同一个会话名称时，会接上该会话的历史。两个会话可以分别处理工作任务和阅读资料。

### 常用对话命令

| 输入 | 作用 |
| --- | --- |
| `/new` | 开始新的对话。 |
| `/status` | 查看模型、会话和上下文状态。 |
| `/help` | 查看可用命令。 |
| `exit` 或 `quit` | 退出 CLI。 |

需要查看工具执行过程时：

```bash
minicrab agent --logs
```

使用独立配置时：

```bash
minicrab agent --config ./config.local.json
```

## 微信使用

### 1. 启用微信渠道

安装命令中的 `.[weixin]` 包含二维码展示和微信媒体处理所需依赖。

在配置文件的 `channels` 节点中启用 `weixin`，保留已填写的模型和其他配置：

```json
{
  "channels": {
    "weixin": {
      "enabled": true,
      "allowFrom": ["*"]
    }
  }
}
```

`allowFrom` 是允许交互的用户列表，`["*"]` 表示允许接入用户；填写具体用户 ID 可限定使用范围。

### 2. 扫码登录

```bash
minicrab channels login weixin
```

终端显示二维码或登录链接。使用微信扫码，并按页面提示完成确认。登录成功后，状态保存在 `.minicrab/weixin/account.json` 中。

### 3. 启动消息接收

```bash
minicrab gateway
```

网关运行期间，微信消息会交给助手处理，回复会发回原对话。该终端需要保持运行，按 `Ctrl+C` 可关闭网关。

登录状态已保存时，可直接运行 `minicrab gateway`。

### 4. 在微信中使用功能

可以直接发送：

- “记住：日常回复使用中文，代码说明尽量简洁。”
- “明天下午三点提醒我整理会议记录。”
- 发送文档后：“索引刚才的文件，提取其中的关键结论。”
- “更正刚才的偏好：回复保留必要细节。”

查看或停止当前任务时，可发送 `/status`、`/help` 或 `/stop`。

需要重新扫码时：

```bash
minicrab channels login weixin --force
```

## 许可证

minicrab 采用 [MIT 许可证](LICENSE)。
