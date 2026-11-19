# Kaggle T4 x2 / ComfyUI MiniMax-H3 INT8 ConvRot 视频生成 Notebook

freedidi 教程整理：在 **Kaggle 免费 GPU（T4 x2）** 上一键部署 ComfyUI + MiniMax-H3（INT8 ConvRot 加速版），通过 Cloudflare 临时隧道访问。

## 包含文件

| 文件 | 说明 |
|---|---|
| `ComfyUI_MiniMax-H3_INT8_Kaggle_T4x2.ipynb` | 完整 Notebook（单元 0 = 主脚本，单元 1/2 为空，单元 3 = 保活守护） |
| `comfyui_minimax_h3_int8_cells.py` | 从 Notebook 提取的纯 Python 代码（便于阅读/Diff，不作为运行入口） |

## 新 Kaggle 账号导入步骤

1. 注册/登录 Kaggle：<https://www.kaggle.com/>
2. **手机号验证**（Kaggle 要求验证后才能启用 GPU 和 Internet）。
3. 获取本仓库的 `ComfyUI_MiniMax-H3_INT8_Kaggle_T4x2.ipynb`（任一方式）：
   - `git clone https://github.com/Cary336/kaggle-comfyui-minimax-h3.git`
   - 或在 GitHub 页面打开该文件 → 右上角 **Raw** → 浏览器另存为 `.ipynb`
4. 进入 <https://www.kaggle.com/code> → **New Notebook** → 顶部菜单 **File → Import Notebook** → 上传上一步的 `.ipynb`。
5. 右侧 **Notebook** 面板 **Settings**：
   - **Accelerator → GPU T4 x2**
   - **Internet → 打开（必须）**（下载模型、克隆 GitHub 仓库、cloudflared 隧道都依赖它）
   - Persistence / 环境：默认即可
6. 运行单元 0（主脚本，约需 20–60 分钟，首次下载约 **37.7 GiB** 模型）：
   - 打印 `ComfyUI 访问网址： https://xxxx.trycloudflare.com` 即成功
   - 浏览器打开该网址 → 左侧 **Workflows** → 优先 `01_reference_4step_sla`（文生视频）或 `04_i2v_fl2v_4step_sla`（图生视频）
   - `PROMPT` 为占位符，请自行替换
7. 单元 0 成功后，**在同一会话内**运行单元 3（保活守护）：
   - 定时检查/自动重启 ComfyUI 与 cloudflared，打印 GPU 心跳，满 `MAX_HOURS = 11.5` 后自动停止（Kaggle 会话上限约 12 小时）
8. 全部停止：在 Notebook 中运行 `stop_services()`。

## 运行前确认

- [ ] Kaggle 账号已完成手机验证
- [ ] Accelerator = **GPU T4 x2**（脚本会检测 GPU，否则直接报错退出）
- [ ] **Internet 已开启**
- [ ] 磁盘空间足够（模型约 37.7 GiB + 3 GiB 余量；Kaggle `/tmp` 随会话结束销毁，重启需重新下载）
- [ ] 不需要任何 API Key / HF Token：模型全部来自 HuggingFace 公开仓库

## 模型清单（自动下载，含 SHA-256 校验和断点续传）

| 文件 | 仓库 | 大小 |
|---|---|---|
| minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors | MATLOWAI/minimax-h3-fused-turbo-int8-convrot | 19.54 GiB |
| qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors | Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4 | 14.61 GiB |
| minimax_h3_video_vae_int8_convrot.safetensors | Kijai/MiniMax-H3-experimental | 2.95 GiB |
| minimax_h3_audio_vae_fp32.safetensors | Comfy-Org/MiniMax-H3 | 0.56 GiB |

> 可选（默认不下载）：`03_derope_dyrope_pb3049_warm375` 工作流需要 LoRA `loras/minimax_h3_temporal_expansion_step375_r16.safetensors`，缺失只影响该工作流。

## 注意事项

- 会话存储（`/tmp/ComfyUI`）随 Kaggle 会话结束而销毁，每次重启需重新下载模型。
- 主脚本只完成“部署并启动服务”，**不自动生成视频**；生成在网页 UI 中手动触发。
- 工作流默认 4 步 SLA；视频目录：`/tmp/ComfyUI/output/video`；诊断日志：`/tmp/ComfyUI/comfyui-h3-int8.log`。
- 隧道网址每次运行都会变化（trycloudflare.com 临时域名），失效会由守护单元自动重建并打印新网址。
- 请遵守 Kaggle、HuggingFace 及各模型上游仓库的使用条款与许可证；生成内容请遵守当地法律法规。

## 来源与致谢

- ComfyUI / ComfyUI-GGUF / ComfyUI-PlagueKind-Nodes-only-sparse / ComfyUI-MAINodes
- 模型：MATLOWAI、Kijai、Momoking（Heretic）、Comfy-Org
- 教程原作者：freedidi（Kaggle 社区教程）
