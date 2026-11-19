# ==================== Cell 0 (code) ====================
"""Kaggle T4 x2 / ComfyUI MiniMax-H3 INT8 ConvRot 加速配置（freedidi 教程）。
直接粘贴到一个 Python 单元运行。启用：Settings > Accelerator > GPU T4 x2、Internet。

替代旧版：FL2VA pruned Q4_K_M GGUF → fused INT8 ConvRot（turbo 已融合，4 步）；
官方 qwen3vl_32b_minimax_h3_nvfp4_awq → Heretic 越狱 NVFP4（qwen3vl_32b_heretic_minimax_h3_nvfp4）；
Kijai int8_convrot 视频 VAE；audio VAE 保留官方 fp32。
自定义节点：ethanfel/ComfyUI-PlagueKind-Nodes-only-sparse（H3SLAAttention）、matlowai/ComfyUI-MAINodes（de-rope 等）。
工作流：MATLOWAI 官方 01/02/03/04/05（N 卡非 API 版），已内嵌ENCODER 文件名替换补丁。
注意：03_derope 工作流需要可选的 motion adapter LoRA（loras/minimax_h3_temporal_expansion_step375_r16.safetensors），
本文未默认下载；缺失时只影响 03，其余工作流不受影响。
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

# ==================== 配置 ====================
ROOT = Path("/tmp/ComfyUI")
PORT = 8188
LOCAL_URL = f"http://127.0.0.1:{PORT}"
START_TUNNEL = True

REVISION_MATLOWAI = "8a8dffaa0cd99c6184833ae0a3b4e9b0089c17b3"
REVISION_KIJAI = "c2e7b89b1800123c49a0f620a1932c01d8952256"
REVISION_HERETIC = "0a9486d17cfb20c1a7abc97118c4d95910b9b1f4"
REVISION_COMFORG = "54472221d27c695fca9621b0c3ed52f06c4af707"

# 新的模型清单（文件名、目录、字节数、SHA-256、下载地址）
MANIFEST = [
    {
        "repo": "MATLOWAI/minimax-h3-fused-turbo-int8-convrot",
        "revision": REVISION_MATLOWAI,
        "file": "diffusion_models/minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors",
        "size": 20980178976,
        "sha256": "4262e4e9963c553fa00016bbe83961407a4fc0a888be95fd836c8d4f2304e48b",
        "folder": "diffusion_models",
    },
    {
        "repo": "Kijai/MiniMax-H3-experimental",
        "revision": REVISION_KIJAI,
        "file": "minimax_h3_video_vae_int8_convrot.safetensors",
        "size": 3171670912,
        "sha256": "9bb2d96f218c76babd85e0611b85ca8fb330a90546c01a0005e8a58a59593410",
        "folder": "vae",
    },
    {
        "repo": "Momoking/Qwen3-VL-32B-Heretic-MiniMax-H3-NVFP4",
        "revision": REVISION_HERETIC,
        "file": "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors",
        "size": 15683129587,
        "sha256": "a166c7bbbe66a22065159e478335fee4a633c4a3e3bb34c8e8ac4cc91bf4996f",
        "folder": "text_encoders",
    },
    {
        "repo": "Comfy-Org/MiniMax-H3",
        "revision": REVISION_COMFORG,
        "file": "vae/minimax_h3_audio_vae_fp32.safetensors",
        "size": 605254808,
        "sha256": "8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48",
        "folder": "vae",
    },
]

WORKFLOW_FILES = [
    "01_reference_4step_sla.json",
    "02_derope_4plus4_sla.json",
    "03_derope_dyrope_pb3049_warm375.json",
    "04_i2v_fl2v_4step_sla.json",
    "05_ref2va_4step_sla.json",
]

# 旧配置文件：移入备份目录（会话内保留，随会话结束销毁；不物理删除）
OLD_FILES = [
    ROOT / "models/diffusion_models/minimax_h3_fl2va_pruned-Q4_K_M.gguf",
    ROOT / "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    ROOT / "models/vae/minimax_h3_video_vae_fp16.safetensors",
    ROOT / "models/loras/minimax_h3_fl2v_turbo_4step_v1.0_768p_comfyui_bf16.safetensors",
]
OLD_WORKFLOWS = [
    ROOT / "user/default/workflows/MiniMax_H3_Kaggle_T2V.json",
    ROOT / "user/default/workflows/MiniMax_H3_Kaggle_I2V.json",
]

ENCODER_AWQ = "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
ENCODER_HERETIC = "qwen3vl_32b_heretic_minimax_h3_nvfp4.safetensors"


def run(*args, cwd=None):
    subprocess.run(list(map(str, args)), cwd=cwd, check=True)


def tail(path):
    path = Path(path)
    if not path.exists():
        return "尚无日志"
    with path.open("rb") as f:
        f.seek(max(0, path.stat().st_size - 12000))
        return f.read().decode("utf-8", "replace")


def request_json(path, timeout=10):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(LOCAL_URL + path, timeout=timeout) as r:
        return json.load(r)


def ready():
    try:
        return "system" in request_json("/system_stats", timeout=2)
    except Exception:
        return False


def stop_owned(name):
    process = globals().get(name)
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def clone_or_update(url, folder):
    folder = Path(folder)
    if not folder.exists():
        run("git", "clone", "--depth", "1", url, folder)
    elif not (folder / ".git").is_dir():
        raise RuntimeError(f"{folder} 已存在但不是 git 仓库；请换 ROOT 或先移开该目录。")


def verify(path, item):
    path = Path(path)
    if not path.is_file() or path.stat().st_size != item["size"]:
        return False
    print("校验 SHA-256：", path.name, flush=True)
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest() == item["sha256"]


def download_model(item):
    target = ROOT / "models" / item["folder"] / Path(item["file"]).name
    target.parent.mkdir(parents=True, exist_ok=True)
    if verify(target, item):
        print("复用：", target.name, flush=True)
        return
    temporary = target.with_name(target.name + ".part")
    url = f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{item['file']}"
    for attempt in range(1, 4):
        try:
            offset = temporary.stat().st_size if temporary.exists() else 0
            if offset >= item["size"]:
                if verify(temporary, item):
                    temporary.replace(target)
                    return
                temporary.unlink()
                offset = 0
            headers = {"User-Agent": "Kaggle-ComfyUI-H3-INT8/1.0"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            print(f"下载 {target.name}，已完成 {offset / 2**30:.2f} GiB，尝试 {attempt}/3", flush=True)
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                append = offset > 0 and r.status == 206
                if append and not r.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                    raise RuntimeError("服务端断点位置不符")
                if r.status not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status}")
                if r.status == 206 and not append:
                    raise RuntimeError("意外的部分文件响应")
                received = offset if append else 0
                last_print = time.monotonic()
                with temporary.open("ab" if append else "wb") as f:
                    while True:
                        chunk = r.read(8 * 1024 * 1024)
                        if not chunk:
                            break
                        received += len(chunk)
                        if received > item["size"]:
                            raise RuntimeError("下载大小超过预期")
                        f.write(chunk)
                        if time.monotonic() - last_print > 20:
                            print(f"  {received / 2**30:.2f} / {item['size'] / 2**30:.2f} GiB", flush=True)
                            last_print = time.monotonic()
            if not verify(temporary, item):
                if temporary.stat().st_size == item["size"]:
                    temporary.unlink()
                raise RuntimeError("文件长度或 SHA-256 不符")
            temporary.replace(target)
            return
        except Exception as e:
            print(f"下载失败：{type(e).__name__}: {e}", flush=True)
            if attempt == 3:
                raise RuntimeError(f"下载 {target.name} 失败；保留断点，下次重跑可续传。") from e
            time.sleep(2 * attempt)


def prepare_python_env(env_dir, base_python):
    """复用旧 venv；跳过 ensurepip，不删除已有依赖。"""
    env_dir = Path(env_dir)
    python = env_dir / "bin/python"
    run(base_python, "-m", "venv", "--without-pip", "--system-site-packages", env_dir)
    probe = subprocess.run([str(python), "-c",
        "import sys; print(sys.prefix); assert sys.prefix != sys.base_prefix"],
        capture_output=True, text=True)
    if probe.returncode or Path(probe.stdout.strip()).resolve() != env_dir.resolve():
        raise RuntimeError("虚拟环境解释器检查失败：" + probe.stderr[-3000:])
    pip_probe = subprocess.run([str(python), "-m", "pip", "--version"],
                               capture_output=True, text=True)
    if pip_probe.returncode:
        manager = subprocess.run([str(base_python), "-m", "pip", "--version"],
                                 capture_output=True, text=True)
        if manager.returncode:
            raise RuntimeError("Notebook 的 Python 也无法调用 pip：" + manager.stderr[-3000:])
        result = subprocess.run([str(base_python), "-m", "pip", "--python", str(python),
                                 "install", "pip"], capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError("用外部 pip 初始化虚拟环境失败：\n" +
                               result.stdout[-2000:] + result.stderr[-4000:])
        run(python, "-m", "pip", "--version")
    else:
        print(pip_probe.stdout.strip(), flush=True)
    return python


print("1/7 检查实际 GPU、内存和旧服务...", flush=True)
probe = subprocess.run([sys.executable, "-c", "import json,torch; print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'gpus':[{'name':torch.cuda.get_device_name(i),'vram':torch.cuda.get_device_properties(i).total_memory,'capability':torch.cuda.get_device_capability(i)} for i in range(torch.cuda.device_count())]}))"], capture_output=True, text=True)
if probe.returncode:
    raise RuntimeError("PyTorch / CUDA 检查失败：" + probe.stderr[-3000:])
hardware = json.loads(probe.stdout.strip().splitlines()[-1])
if not hardware["gpus"]:
    raise RuntimeError("未检测到 CUDA GPU。先在 Kaggle Settings > Accelerator 选择 GPU T4 x2，并重新启动会话。")
for i, gpu in enumerate(hardware["gpus"]):
    print(f"GPU {i}: {gpu['name']} / {gpu['vram']/2**30:.2f} GiB", flush=True)
mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
print(f"系统内存：{int(mem['MemTotal'].split()[0])/2**20:.1f} GiB；当前可用 {int(mem['MemAvailable'].split()[0])/2**20:.1f} GiB", flush=True)
stop_owned("comfy_process")
with socket.socket() as sock:
    if sock.connect_ex(("127.0.0.1", PORT)) == 0:
        raise RuntimeError(f"端口 {PORT} 被其他服务占用。请先停止旧 ComfyUI（例如重启会话），或更改 PORT。")

print("2/7 备份（移出而非删除）旧模型和旧工作流...", flush=True)
backup_dir = ROOT.parent / f"h3_backup_old_{int(time.time())}"
moved = []
for item in OLD_FILES + OLD_WORKFLOWS:
    if item.exists():
        size_bytes = item.stat().st_size
        target = backup_dir / item.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(item), str(target))
        moved.append(f"{item} -> {target}（{size_bytes/2**30:.2f} GiB）" if size_bytes > 2**30 else f"{item} -> {target}")
for line in moved:
    print("  已备份：", line, flush=True)
if not moved:
    print("  没有需要备份的旧文件。", flush=True)

print("3/7 更新 ComfyUI、H3 专用 GGUF 插件与自定义节点（SLA / MAINodes）...", flush=True)
ROOT.parent.mkdir(parents=True, exist_ok=True)
clone_or_update("https://github.com/Comfy-Org/ComfyUI.git", ROOT)
plugin = ROOT / "custom_nodes/ComfyUI-GGUF"
clone_or_update("https://github.com/leejet/ComfyUI-GGUF.git", plugin)
sla_dir = ROOT / "custom_nodes/ComfyUI-PlagueKind-Nodes-only-sparse"
clone_or_update("https://github.com/ethanfel/ComfyUI-PlagueKind-Nodes-only-sparse.git", sla_dir)
mai_dir = ROOT / "custom_nodes/ComfyUI-MAINodes"
clone_or_update("https://github.com/matlowai/ComfyUI-MAINodes.git", mai_dir)

print("4/7 安装依赖，保留 Kaggle 自带 CUDA PyTorch...", flush=True)
venv = ROOT.parent / "h3-python-env"
python = prepare_python_env(venv, sys.executable)
torch_probe = subprocess.run([str(python), "-c",
    "import torch; assert torch.cuda.is_available(); print(torch.__version__)"],
    capture_output=True, text=True)
if torch_probe.returncode:
    raise RuntimeError("虚拟环境无法继承 Kaggle CUDA PyTorch：\n" + torch_probe.stderr[-4000:])
if torch_probe.stdout.strip().splitlines()[-1] != hardware["torch"]:
    raise RuntimeError("虚拟环境 PyTorch 版本与 Notebook 不同，请改用新的 h3-python-env 目录后重跑。")
constraints = ROOT / "h3-torch-constraints.txt"
constraints.write_text("torch==" + hardware["torch"] + "\n")
run(python, "-m", "pip", "install", "-q", "-c", constraints,
    "-r", ROOT / "requirements.txt",
    "-r", plugin / "requirements.txt")
# MAINodes 仅需 cryptography；Triton 随 torch cu 版分发
run(python, "-m", "pip", "install", "-q", "cryptography>=42")
dep_check = subprocess.run([str(python), "-c",
    "import torch; assert torch.cuda.is_available(); import gguf, av, cryptography; import triton; print(torch.__version__, 'triton', triton.__version__)"],
    capture_output=True, text=True)
if dep_check.returncode:
    # Triton 缺失则补装（应与 torch 捆绑版本一致）
    run(python, "-m", "pip", "install", "-q", "triton")
    dep_check = subprocess.run([str(python), "-c",
        "import torch; assert torch.cuda.is_available(); import gguf, av, cryptography, triton; print(torch.__version__)"],
        capture_output=True, text=True)
    if dep_check.returncode:
        raise RuntimeError("依赖 / CUDA 自检失败：" + dep_check.stderr[-4000:])
print("PyTorch：", dep_check.stdout.strip(), flush=True)

print("5/7 下载新 H3 INT8 模型（支持断点续传和 SHA-256 校验）...", flush=True)
needed = 0
for item in MANIFEST:
    target = ROOT / "models" / item["folder"] / Path(item["file"]).name
    if target.exists() and target.stat().st_size == item["size"]:
        continue
    part = target.with_name(target.name + ".part")
    partial = min(part.stat().st_size, item["size"]) if part.exists() else 0
    needed += item["size"] - partial
free = shutil.disk_usage(ROOT).free
print(f"模型合计 {sum(x['size'] for x in MANIFEST)/2**30:.2f} GiB；剩余下载约 {needed/2**30:.2f} GiB；可用磁盘 {free/2**30:.2f} GiB", flush=True)
if free < needed + 3 * 2**30:
    raise RuntimeError("磁盘不足：请为剩余下载和输出视频至少再留 3 GiB；不会自动删除现有模型。")
for item in MANIFEST:
    download_model(item)

print("6/7 保存 MATLOWAI 工作流（ENCODER 替换为 Heretic）并启动服务...", flush=True)
workflow_dir = ROOT / "user/default/workflows"
workflow_dir.mkdir(parents=True, exist_ok=True)
for name in WORKFLOW_FILES:
    url = f"https://huggingface.co/MATLOWAI/minimax-h3-fused-turbo-int8-convrot/resolve/{REVISION_MATLOWAI}/workflows/{name}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Kaggle-ComfyUI-H3-INT8/1.0"}), timeout=120) as r:
            text = r.read().decode("utf-8")
        data = json.loads(text)  # 仅校验 JSON 合法
    except Exception as e:
        raise RuntimeError(f"工作流 {name} 下载或解析失败：{e}")
    patched = text.replace(ENCODER_AWQ, ENCODER_HERETIC)
    if patched == text:
        print(f"  ⚠️ {name} 未直接包含 AWQ 编码器文件名（可能已改名），原样保存。", flush=True)
    path = workflow_dir / name
    old_backup = backup_dir / f"{name}.bak-{int(time.time())}"
    if path.exists():
        path.replace(old_backup)
        print(f"  已备份同名旧工作流：{old_backup}", flush=True)
    path.write_text(patched, encoding="utf-8")
    print("  工作流：", path, flush=True)

server_log = ROOT / "comfyui-h3-int8.log"
command = [str(python), "-u", "main.py", "--listen", "127.0.0.1", "--port", str(PORT),
           "--cuda-device", "all", "--disable-pinned-memory", "--cache-none", "--preview-method", "none"]
with server_log.open("a", encoding="utf-8") as log:
    comfy_process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
deadline = time.monotonic() + 300
while not ready():
    if comfy_process.poll() is not None:
        raise RuntimeError("ComfyUI 启动失败：\n" + tail(server_log))
    if time.monotonic() >= deadline:
        raise TimeoutError("ComfyUI 启动超时：\n" + tail(server_log))
    time.sleep(1)

objects = request_json("/object_info", timeout=60)
required = {"UNETLoader", "VAELoader", "SaveVideo", "CreateVideo",
            "MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo", "MiniMaxH3SigmaShift",
            "ResolutionSelector", "H3SLAAttention", "H3JerkOracle"}
missing = required - objects.keys()
if missing:
    raise RuntimeError("缺少节点：" + ", ".join(sorted(missing)) + "\n" + tail(server_log))
clip_spec = objects["CLIPLoader"]["input"]["required"]["type"]
clip_types = clip_spec[0] if isinstance(clip_spec[0], list) else (
    clip_spec[1].get("options") if len(clip_spec) > 1 and isinstance(clip_spec[1], dict) else None)
if clip_types is not None and "minimax" not in clip_types:
    raise RuntimeError("CLIPLoader 没有 minimax 类型，请更新 ComfyUI。")
print("节点和服务就绪；尚未执行视频生成，启动成功不代表推理自检通过。", flush=True)

print("7/7 创建 / 复用 Cloudflare 临时访问链接...", flush=True)
if START_TUNNEL:
    old_tunnel = globals().get("tunnel_process")
    old_url = globals().get("url", "")
    if not (old_tunnel is not None and old_tunnel.poll() is None and
            globals().get("_h3_tunnel_target") == LOCAL_URL and
            re.fullmatch(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", str(old_url))):
        stop_owned("tunnel_process")
        binary = ROOT.parent / "cloudflared"
        if not binary.exists():
            temporary = binary.with_suffix(".download")
            source = "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
            with urllib.request.urlopen(source, timeout=120) as response, temporary.open("wb") as f:
                shutil.copyfileobj(response, f)
            if temporary.read_bytes()[:4] != b"\x7fELF":
                raise RuntimeError("cloudflared 下载内容不是 Linux ELF 文件")
            temporary.replace(binary)
        binary.chmod(0o755)
        tunnel_log = ROOT / "cloudflared-h3.log"
        with tunnel_log.open("w", encoding="utf-8") as log:
            tunnel_process = subprocess.Popen([str(binary), "tunnel", "--url", LOCAL_URL, "--no-autoupdate"], stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 90
        while True:
            content = tail(tunnel_log)
            match = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", content)
            if match:
                url = match.group(0)
                globals()["url"] = url
                globals()["_h3_tunnel_target"] = LOCAL_URL
                break
            if tunnel_process.poll() is not None or time.monotonic() >= deadline:
                raise RuntimeError("隧道启动失败：\n" + content)
            time.sleep(0.5)
    print("ComfyUI 访问网址：", globals().get("url"), flush=True)
else:
    print("ComfyUI 本地地址：", LOCAL_URL, flush=True)
print("打开 Workflows，优先从 01_reference_4step_sla（文生视频）或 04_i2v_fl2v_4step_sla（图生视频）开始。")
print("03_derope_dyrope_pb3049_warm375 需要可选的 loras/minimax_h3_temporal_expansion_step375_r16.safetensors（未默认下载）。")
print("生成默认为工作流自带参数：4 步 SLA、16:9、分辨率可滚动选择；PROMPT 为占位符，请自行替换。")
print("视频目录：", ROOT / "output/video")
print("诊断日志：", server_log)


def stop_services():
    stop_owned("tunnel_process")
    stop_owned("comfy_process")
    print("本脚本的 ComfyUI 和隧道已停止。")


# ==================== Cell 3 (code) ====================
"""
H3 ComfyUI 保活守护（独立单元，必须在主脚本运行成功后、同一会话内运行）
1. 定时检查 ComfyUI 进程和 /system_stats；连续失败就自动重启
2. 检查 cloudflared 进程和公网地址；失效就重建（新网址会打印出来）
3. 每隔一段时间输出心跳（GPU 显存/利用率、生成队列）
4. 运行满 MAX_HOURS 后自动停止服务，省 GPU 周配额
停止守护：点单元的 ■ Interrupt。只停守护，不停服务；全部停止请运行 stop_services()
"""
import datetime, re, subprocess, threading, time, urllib.request

# ==================== 保活配置 ====================
CHECK_INTERVAL = 60          # 检查间隔（秒）
HEARTBEAT_EVERY = 5          # 每 N 次检查打印一次心跳（60s×5=5 分钟）
FAIL_THRESHOLD = 8           # ComfyUI 连续无响应次数达到后重启
TUNNEL_FAIL_THRESHOLD = 8    # 公网地址连续不可达次数达到后重建隧道
MAX_RESTARTS = 5             # 自动重启总次数上限，防止无限重启
MAX_HOURS = 11.5             # 守护运行满该时长后结束（Kaggle 单次会话上限约 12h）
AUTO_STOP_AT_END = True      # 到时是否同时停止 ComfyUI 和隧道
RUN_IN_BACKGROUND = False    # False=方案A前台（推荐）；True=方案B后台线程

_need = ("ROOT", "LOCAL_URL", "command", "server_log", "ready", "tail",
         "request_json", "stop_owned", "stop_services")
_missing = [n for n in _need if n not in globals()]
if _missing:
    raise RuntimeError("缺少主脚本变量：" + ", ".join(_missing) + "。请先在同一会话运行主脚本。")

_TUNNEL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
_guard_stop = globals().get("_guard_stop") or threading.Event()


def _log(msg):
    print(f"[{datetime.datetime.now():%H:%M:%S}] {msg}", flush=True)


def _gpu_brief():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15).stdout
        rows = [[x.strip() for x in line.split(",")] for line in out.strip().splitlines()]
        return " | ".join(f"GPU{i}:{u}/{t}MiB {g}%" for i, u, t, g in rows)
    except Exception as e:
        return f"GPU 状态不可用({type(e).__name__})"


def _queue_brief():
    try:
        q = request_json("/queue", timeout=10)
        return f"队列 运行{len(q.get('queue_running', []))}/等待{len(q.get('queue_pending', []))}"
    except Exception:
        return "队列 未知"


def _public_ok(u):
    try:
        req = urllib.request.Request(u + "/system_stats", headers={"User-Agent": "h3-keepalive"})
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status == 200
    except Exception:
        return False


def _restart_comfy(reason):
    global comfy_process
    _log(f"⚠️ 重启 ComfyUI：{reason}")
    stop_owned("comfy_process")
    time.sleep(3)
    with server_log.open("a", encoding="utf-8") as log:
        comfy_process = subprocess.Popen(command, cwd=ROOT, stdout=log,
                                         stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 300
    while not ready():
        if comfy_process.poll() is not None:
            raise RuntimeError("ComfyUI 重启后退出：\n" + tail(server_log)[-3000:])
        if time.monotonic() >= deadline:
            raise TimeoutError("ComfyUI 重启超时：\n" + tail(server_log)[-3000:])
        time.sleep(2)
    _log("✅ ComfyUI 已恢复")


def _restart_tunnel(reason):
    global tunnel_process, url, _h3_tunnel_target
    _log(f"⚠️ 重建隧道：{reason}")
    stop_owned("tunnel_process")
    binary = ROOT.parent / "cloudflared"
    if not binary.exists():
        raise RuntimeError("找不到 cloudflared，请先用 START_TUNNEL=True 运行主脚本。")
    tunnel_log = ROOT / "cloudflared-h3.log"
    with tunnel_log.open("w", encoding="utf-8") as log:
        tunnel_process = subprocess.Popen(
            [str(binary), "tunnel", "--url", LOCAL_URL, "--no-autoupdate"],
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        m = _TUNNEL_RE.search(tail(tunnel_log))
        if m:
            url, _h3_tunnel_target = m.group(0), LOCAL_URL
            _log(f"✅ 新访问网址（旧网址已失效）：{url}")
            return
        if tunnel_process.poll() is not None:
            break
        time.sleep(0.5)
    raise RuntimeError("隧道重建失败：\n" + tail(tunnel_log)[-3000:])


def _guard_loop():
    start = time.monotonic()
    restarts = comfy_fail = tunnel_fail = n = 0
    _log(f"保活启动：每 {CHECK_INTERVAL}s 检查；{MAX_HOURS}h 后"
         f"{'停止服务' if AUTO_STOP_AT_END else '仅结束守护'}")
    _log(f"当前访问网址：{globals().get('url') or LOCAL_URL}")
    while not _guard_stop.is_set():
        n += 1
        hours = (time.monotonic() - start) / 3600
        try:
            # 1) ComfyUI
            proc = globals().get("comfy_process")
            if proc is not None and proc.poll() is not None:
                comfy_fail, why = FAIL_THRESHOLD, f"进程已退出 code={proc.returncode}"
            elif ready():
                comfy_fail, why = 0, ""
            else:
                comfy_fail += 1
                why = f"/system_stats 无响应（{comfy_fail}/{FAIL_THRESHOLD}）"
                _log("… " + why)
            if comfy_fail >= FAIL_THRESHOLD:
                if restarts >= MAX_RESTARTS:
                    _log("❌ 已达最大重启次数，守护退出。请查看 server_log。")
                    break
                restarts += 1
                _restart_comfy(why)
                comfy_fail = 0
            # 2) 隧道（仅在 ComfyUI 正常时检查）
            elif globals().get("START_TUNNEL", False):
                tp, u = globals().get("tunnel_process"), globals().get("url", "")
                if tp is None or tp.poll() is not None:
                    tunnel_fail, twhy = TUNNEL_FAIL_THRESHOLD, "隧道进程不存在/已退出"
                elif u and _public_ok(u):
                    tunnel_fail = 0
                else:
                    tunnel_fail += 1
                    twhy = f"公网地址不可达（{tunnel_fail}/{TUNNEL_FAIL_THRESHOLD}）"
                    _log("… " + twhy)
                if tunnel_fail >= TUNNEL_FAIL_THRESHOLD:
                    if restarts >= MAX_RESTARTS:
                        _log("❌ 已达最大重启次数，守护退出。")
                        break
                    restarts += 1
                    _restart_tunnel(twhy)
                    tunnel_fail = 0
        except Exception as e:
            _log(f"❌ 守护异常：{type(e).__name__}: {e}")
        # 3) 心跳
        if n % HEARTBEAT_EVERY == 1:
            _log(f"💓 #{n} 已守护 {hours:.2f}h | 重启 {restarts}/{MAX_RESTARTS} | "
                 f"{_gpu_brief()} | {_queue_brief()}")
        # 4) 到时停止
        if hours >= MAX_HOURS:
            _log(f"⏰ 已达 {MAX_HOURS}h")
            if AUTO_STOP_AT_END:
                stop_services()
            break
        _guard_stop.wait(CHECK_INTERVAL)
    _log("保活守护已结束。")


def stop_guard():
    _guard_stop.set()
    print("已请求停止保活守护（ComfyUI/隧道不受影响）。")


_guard_stop.clear()
if RUN_IN_BACKGROUND:
    _t = globals().get("_guard_thread")
    if _t is not None and _t.is_alive():
        print("后台守护已在运行；如需重启先运行 stop_guard()。")
    else:
        _guard_thread = threading.Thread(target=_guard_loop, daemon=True, name="h3-keepalive")
        _guard_thread.start()
        print("后台守护已启动；停止：stop_guard()")
else:
    try:
        _guard_loop()
    except KeyboardInterrupt:
        _log("⏹ 已手动中断守护；ComfyUI/隧道仍在运行。全部停止请运行 stop_services()")