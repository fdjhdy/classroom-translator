# 俄语课堂实时翻译系统

手机/笔记本在教室收音，寝室主机跑识别和翻译，字幕实时回传，
每节课自动存档（Markdown 记录 + 完整 WAV 录音），课后一键精翻成笔记。
- **v15 识别引擎换 GigaAM**（Sber 开源的俄语专用模型，ONNX int8，CPU 推理）：
  实测比 Whisper large-v3 幻觉更少（不再把片头音乐听成"Продолжение следует..."），
  远场嘈杂俄语识别明显更准；`--asr-engine whisper` 可切回 faster-whisper
- **v14 内置课堂助理**：每 ~5 分钟自动生成中文课堂纪要、检测到老师在提问时
  弹卡片提醒（附参考回答）、随时向课堂内容提问（"没听懂刚才"/"现在要干嘛"或自由提问）

```
教室: 手机/笔记本浏览器                    寝室: 主机
  麦克风收音 ──Tailscale 隧道──▶  silero-VAD 切句
  字幕/历史记录 ◀── 回传 ────     GigaAM v2 俄语识别（CPU）/ faster-whisper（备用）
                                  LLM 纠错+翻译成中文（带上下文+术语表）
                                  ├── records/*.md   逐句记录
                                  └── records/*.wav  整课录音 → post_refine.py 精翻
```

## 一、主机端部署（只需配一次）

### 1. 环境

Python 3.10~3.12，然后：

```bat
python -m pip install faster-whisper silero-vad onnxruntime websockets openai numpy onnx-asr
python -m pip install nvidia-cublas-cu12 nvidia-cudnn-cu12
```

第二条只有用 `--asr-engine whisper`（显卡推理）才需要。**装完必须重启 server 进程才生效**
（DLL 搜索路径在进程启动时定死）。

### 1.5 下载 GigaAM 模型（约 240MB，首次）

模型文件不入库（见 .gitignore），从 HuggingFace 镜像下载到 `models/gigaam-v2/`：

```bat
mkdir models\gigaam-v2 2>nul & cd models\gigaam-v2
curl -L -o v2_rnnt_encoder.int8.onnx https://hf-mirror.com/istupakov/gigaam-v2-onnx/resolve/main/v2_rnnt_encoder.int8.onnx
curl -L -o v2_rnnt_decoder.int8.onnx https://hf-mirror.com/istupakov/gigaam-v2-onnx/resolve/main/v2_rnnt_decoder.int8.onnx
curl -L -o v2_rnnt_joint.int8.onnx   https://hf-mirror.com/istupakov/gigaam-v2-onnx/resolve/main/v2_rnnt_joint.int8.onnx
curl -L -o v2_vocab.txt              https://hf-mirror.com/istupakov/gigaam-v2-onnx/resolve/main/v2_vocab.txt
cd ..\..
```

### 2. 配置 API Key

翻译 Key 从环境变量 `TRANSLATE_API_KEY` 读取（不要写进任何脚本或提交进仓库）：

```bat
setx TRANSLATE_API_KEY "sk-你的key"
```

（`setx` 后需重开终端/重启生效；临时测试可先在当前窗口 `set TRANSLATE_API_KEY=...`）

### 3. 启动

参考 `start.bat`（记事本保存时编码选 **ANSI**，避免中文乱码）。核心命令：

```bat
python server.py --subject "宏观经济学" --translate-base-url https://api.siliconflow.cn/v1 --translate-model deepseek-ai/DeepSeek-V3 --glossary-file glossary.txt
```

- PATH 里加 cublas/cudnn 的 bin 目录是防止 `cublas64_12.dll not found` 的保险
- 翻译接口可换任何 OpenAI 兼容服务（DeepSeek 官方用 `--translate-model deepseek-chat` 且去掉 base-url）
- 没有术语表就先去掉 `--glossary-file`

启动后首行显示 `[版本]`，看到 `[就绪]` 即完成。首次启动会下载模型
（large-v3 约 3G）；下载卡住就设镜像再重启：`set HF_ENDPOINT=https://hf-mirror.com`
（同参数重启会断点续传；换源则可能重新下载，旧残留可用
`Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub" -Recurse -Filter *.incomplete | Remove-Item` 清理）。

### 4. 隧道（手机/笔记本从外面访问）

主机和终端设备都装 Tailscale，登录同一账号。然后主机上执行一次：

```bat
tailscale serve --bg 8765
tailscale serve --https=8443 --bg http://127.0.0.1:8766
```

第一条是主通道（WebSocket），第二条是 HTTP 兼容通道（WS 被代理吞掉时自动降级用）。
完成后得到 `https://主机名.xxxx.ts.net/` 这个固定网址，发给手机即可。

注意：
- 若提示 `Serve is not enabled`，按它给的链接在管理后台开一下 HTTPS 证书即可
- 笔记本上如果开着代理软件，要给 `*.ts.net` 加 DIRECT 直连规则，否则网页打不开
- 该网址目前无访问控制，拿到链接的人即可查看记录，请勿外传

### 5. 主机日常

- 电源设置里**关闭睡眠**
- `start.bat` 可放开机启动文件夹（Win+R 输入 `shell:startup`）

## 二、上课使用（手机/笔记本）

浏览器打开 `https://主机名.xxxx.ts.net/`（可"添加到主屏幕"当 App 用）：

1. 顶栏「课程」框填课程名（**填一次自动记住**，换课直接改，即时生效）
2. 点「开始」→ 允许麦克风 → 对着老师放桌上
3. 状态栏绿灯 = 已连接；显示"兼容模式" = WebSocket 被网络吞了，自动降级 HTTP，功能不变
4. 顶栏下方绿色音量条说话时跳动（正常说话峰值应在 0.05 以上）
5. 字幕卡片：灰色俄语原文 + 白色中文翻译；底部黄条是正在说的半成品
6. 老师在提问时，顶部弹出蓝色卡片并给出参考回答，可一键"帮我回答"
7. 点顶栏「纪要」看自动生成的课堂纪要（每 ~5 分钟刷新），下方可快速提问：
   「没听懂刚才」让 AI 用大白话解释最近几句，「现在要干嘛」汇总作业/考试安排
8. 下课点「结束」；右上角「记录」可随时翻看历史课程

**收音是准确率的第一杠杆**：坐前排、手机正对老师；强烈建议买个 Type-C 领夹麦。

## 三、课后精翻（质量担当）

实时翻译为了速度牺牲质量，真正的笔记靠课后：

```bat
python post_refine.py "records\20260914_1930_宏观经济学.wav" --subject "宏观经济学" --glossary-file glossary.txt
```

用 large-v3 + beam_size=5 重新识别、带上下文逐句翻译，产出 `*_精翻.md`。
把精翻稿发给 AI 出知识点总结和术语表，术语回填 `glossary.txt`，形成正循环。
Key 同样从环境变量 `TRANSLATE_API_KEY` 读取（也可用 `--key` 显式传入）。

## 四、术语表（越用越准的核心）

`glossary.txt`，每行一组：

```
субсидия=补贴
равновесная цена=均衡价格
```

启动带上 `--glossary-file glossary.txt` 后同时作用于两层：
- **识别**：俄语术语注入 Whisper 的 initial_prompt，专业词汇识别率明显提升
- **翻译**：作为对照表优先采用

可以把课程英文教材发给 AI，让它生成英俄中三语术语表。

## 五、参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--asr-engine` | gigaam | 识别引擎：gigaam（俄语专用，CPU）/ whisper（多语种，显卡） |
| `--gigaam-dir` | ./models/gigaam-v2 | GigaAM 模型文件目录 |
| `--model` | large-v3 | （仅 whisper 引擎）识别模型；large-v3-turbo 更快但嘈杂环境幻觉多 |
| `--beam-size` | 3 | 越大越准越慢 |
| `--deepseek-key` | 环境变量 `TRANSLATE_API_KEY` | 翻译 API Key |
| `--translate-base-url` / `--translate-model` | DeepSeek 官方 | 换硅基流动见启动示例 |
| `--subject` | 空 | 默认课程名（页面上可临时改，优先级更高） |
| `--glossary-file` | 空 | 术语表路径 |
| `--records-dir` | records | 记录/录音目录，设空字符串关闭 |
| `--port` / `--http-port` | 8765 / 8766 | 主通道 / HTTP 兼容通道 |

## 六、排查决策表

终端日志 → 病因 → 解法：

| 日志现象 | 病因 | 解法 |
|---|---|---|
| 没有 `[音频]` 行 | 浏览器没发出声音 | 麦克风权限/选错输入设备（地址栏左侧图标里改） |
| 有 `[音频]` 但峰值 ≤0.003 | 静音轨/音量太小 | 换输入设备；Windows 输入音量拉满；凑近 |
| 有音量无 `[VAD] 语音开始` | VAD 太谨慎 | 音量再放大；还是不行调 threshold |
| `[识别] 结果为空` | 噪音/音量小/幻觉被过滤 | 改善收音；这是过滤在保护你 |
| `[翻译失败: 401]` | key 和 base-url 不配套 | 检查 key 与服务商是否匹配 |
| `[翻译失败: timed out]` ×多 | 到 API 的链路不稳 | 已自动重试 3 次；还频繁就换服务商 |
| `[错误] RuntimeError: cublas...` | CUDA 库没找到 | 见部署第 1、3 步，PATH 保险 + 重启 |
| 网页能开但一直"断线" | WS 被代理吞 | 等 6 秒会自动切"兼容模式"；确认 8443 映射已加 |

## 七、文件清单

- `server.py` — 主机服务端（VAD 切句 + 识别[GigaAM/Whisper 双引擎] + LLM 纠错翻译 + WebSocket 主通道 + HTTP 兼容通道 + 网页分发 + 记录/录音存档 + 课堂助理）
- `phone_client.html` — 网页客户端（手机/笔记本浏览器用，server 自动分发）
- `post_refine.py` — 课后精翻脚本（WAV 重新识别 + 带上下文翻译出笔记）
- `start.bat` — 一键启动示例（Key 从环境变量读取）
- `models/gigaam-v2/` — GigaAM 模型文件（240MB，按部署 1.5 节下载；不入库）
- `glossary.txt` — 术语表（自建，越攒越准；不入库）
