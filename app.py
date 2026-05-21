import os
import re
import json
import subprocess
import importlib.util
import toml
import glob
import math
from pathlib import Path
import sys
import gradio as gr
import psutil
from PIL import Image
import torch

CSS = """
.gradio-container {
    position: relative !important;
}

#main-header {
    margin-bottom: -10px !important;
}

.attribution {
    position: absolute;
    right: 25px;
    top: 5px;
    z-index: 99999;
    pointer-events: none;
}

.author-text {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    font-size: 16px;
    font-weight: 700;
    text-shadow: 0 0 15px rgba(187, 154, 247, 0.6);
    letter-spacing: 0.5px;
}

#log-container textarea {
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', 'Consolas', monospace !important;
    font-size: 14px !important;
    line-height: 1.3 !important;
    white-space: pre-wrap !important;
    overflow-x: hidden !important;
    height: 385px !important;
    border: 1px solid #2f334d !important;
}

*::-webkit-scrollbar {
    width: 6px !important;
    height: 6px !important;
}
*::-webkit-scrollbar-track {
    background: rgba(0, 0, 0, 0.1) !important;
}
*::-webkit-scrollbar-thumb {
    background: #7aa2f7 !important;
    border-radius: 10px !important;
}
*::-webkit-scrollbar-thumb:hover {
    background: #bb9af7 !important;
}

footer {
    display: none !important;
}

"""

JS_SCROLL = """
function() {
    setTimeout(() => {
        const el = document.querySelector('#log-container textarea');
        if (el) {
            el.scrollTop = el.scrollHeight;
        }
    }, 50);
}
"""


LOG_BLACKLIST = [
    "triton not found",
    "flop counting will not work",
    "Lib\\site-packages\\torch\\utils\\flop_counter.py"
]


LOG_BOX__MAX_LINES = 16
GALLERY_HEIGHT = 440
MAX_LOG_LINES = 500
MAX_PROMPTS = 5


ROOT = Path(__file__).resolve().parent
TRAIN_PYTHON = Path(sys.executable).resolve()

TRAIN_BASE = ROOT / "training"
OUTPUT_BASE = TRAIN_BASE / "output"
SETTINGS_FILE = TRAIN_BASE / "settings.json"

TRAIN_DIR = TRAIN_BASE / "sd-scripts" 
TRAIN_SCRIPT = TRAIN_DIR / "anima_train_network.py"

training_process = None

for d in [TRAIN_BASE, OUTPUT_BASE]:
    d.mkdir(parents=True, exist_ok=True)


DEFAULT_SETTINGS = {
    "trigger_word": "",
    "project_name": "",
    "dataset_path": "",
    "dit_path": str(ROOT / "models" / "anima" / "dit" / "anima-preview.safetensors"),
    "qwen_path": str(ROOT / "models" / "anima" / "text_encoder" / "qwen_3_06b_base.safetensors"),
    "vae_path": str(ROOT / "models" / "anima" / "vae" / "qwen_image_vae.safetensors"),
    "network_rank": 32,
    "learning_rate": "1.0", # Prodigy default
    "optimizer": "Prodigy", # Prodigy default
    "training_steps": 2400,
    "save_steps": 300,
    "sample_steps": 300,
    "pos_prompt": "",
    "pos_prompt_2": "",
    "pos_prompt_3": "",
    "pos_prompt_4": "",
    "pos_prompt_5": "",
    "neg_prompt": "worst quality, low quality, score_1, score_2, score_3, artist name",
    "width": 1024,
    "height": 1024,
    "sample_steps_gen": 30,
    "sample_cfg": 4.0,
    "sample_seed": 42,
    "train_seed": 42,
    "train_batch_size": 1,
    "gradient_accumulation_steps": 1
}
SETTINGS_KEYS = list(DEFAULT_SETTINGS.keys())

def load_settings():
    settings = DEFAULT_SETTINGS.copy()
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings.update(json.load(f))
        except Exception:
            pass
    return settings

def save_settings(settings_dict):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings_dict, f, indent=4)

def auto_save_state(*args):
    current_state = dict(zip(SETTINGS_KEYS, args))
    save_settings(current_state)

def settings_to_values(settings_dict):
    return [settings_dict.get(k, DEFAULT_SETTINGS[k]) for k in SETTINGS_KEYS]

def get_prompt_count(settings_dict):
    count = 1
    for i in range(2, MAX_PROMPTS + 1):
        if str(settings_dict.get(f"pos_prompt_{i}", "")).strip():
            count = i
    return count

def prompt_visibility_updates(count):
    updates = []
    for i in range(2, MAX_PROMPTS + 1):
        updates.append(gr.update(visible=i <= count))
    updates.append(gr.update(interactive=count < MAX_PROMPTS))
    return updates

def add_prompt_row(current_count):
    try:
        count = int(current_count)
    except Exception:
        count = 1
    next_count = min(MAX_PROMPTS, count + 1)
    return [gr.update(value=next_count)] + prompt_visibility_updates(next_count)

def compute_prompt_ui_updates(*vals):
    count = get_prompt_count(dict(zip(SETTINGS_KEYS, vals)))
    return [gr.update(value=count)] + prompt_visibility_updates(count)

def list_output_projects():
    if not OUTPUT_BASE.exists():
        return []
    return sorted([p.name for p in OUTPUT_BASE.iterdir() if p.is_dir()])

def _normalize_project_input(project_input: str) -> Path:
    p = Path((project_input or "").strip())
    if not p:
        return OUTPUT_BASE / ""
    if p.is_absolute():
        if p.name.lower() == "configs":
            return p.parent
        return p
    if p.parts and len(p.parts) >= 2 and p.parts[-1].lower() == "configs":
        return OUTPUT_BASE / p.parts[-2]
    return OUTPUT_BASE / p.name

def _read_prompt_file(prompt_path: Path):
    if not prompt_path.exists():
        return [""], "", 1024, 1024, 30, 4.0, 42
    lines = [ln.strip() for ln in prompt_path.read_text(encoding="utf-8", errors="ignore").splitlines() if ln.strip()]
    if not lines:
        return [""], "", 1024, 1024, 30, 4.0, 42

    pattern = re.compile(r"^(.*?)\s+--n\s+(.*?)\s+--w\s+(\d+)\s+--h\s+(\d+)\s+--l\s+([0-9.]+)\s+--s\s+(\d+)\s+--d\s+(-?\d+)\s*$")
    prompts = []
    neg = ""
    w, h, steps, cfg, seed = 1024, 1024, 30, 4.0, 42
    for idx, line in enumerate(lines):
        m = pattern.match(line)
        if not m:
            prompts.append(line)
            continue
        pos, neg_line, w_line, h_line, cfg_line, steps_line, seed_line = m.groups()
        prompts.append(pos)
        if idx == 0:
            neg = neg_line
            w, h = int(w_line), int(h_line)
            steps, cfg, seed = int(steps_line), float(cfg_line), int(seed_line)
    return prompts, neg, w, h, steps, cfg, seed

def load_project_config(project_input):
    project_dir = _normalize_project_input(project_input)
    configs_dir = project_dir / "configs"

    if not project_dir.exists() or not configs_dir.exists():
        msg = f"⚠️ Project/configs folder not found: {project_input}"
        return settings_to_values(load_settings()) + [msg]

    training_files = sorted(configs_dir.glob("*_training.toml"), key=os.path.getmtime, reverse=True)
    dataset_files = sorted(configs_dir.glob("*_dataset.toml"), key=os.path.getmtime, reverse=True)
    prompt_files = sorted(configs_dir.glob("*_prompts.txt"), key=os.path.getmtime, reverse=True)

    if not training_files:
        msg = f"⚠️ No training config found in: {configs_dir}"
        return settings_to_values(load_settings()) + [msg]

    training_cfg = toml.load(training_files[0])
    dataset_cfg = toml.load(dataset_files[0]) if dataset_files else {}

    loaded = load_settings()
    loaded["project_name"] = project_dir.name
    prefix = (
        dataset_cfg.get("datasets", [{}])[0]
        .get("subsets", [{}])[0]
        .get("caption_prefix")
    )
    loaded["trigger_word"] = (prefix or "").replace(", ", "").strip()
    loaded["dataset_path"] = (
        dataset_cfg.get("datasets", [{}])[0]
        .get("subsets", [{}])[0]
        .get("image_dir", loaded["dataset_path"])
    )
    loaded["dit_path"] = training_cfg.get("pretrained_model_name_or_path", loaded["dit_path"])
    loaded["qwen_path"] = training_cfg.get("qwen3", loaded["qwen_path"])
    loaded["vae_path"] = training_cfg.get("vae", loaded["vae_path"])
    loaded["network_rank"] = int(training_cfg.get("network_dim", loaded["network_rank"]))
    loaded["learning_rate"] = str(training_cfg.get("learning_rate", loaded["learning_rate"]))
    loaded["optimizer"] = training_cfg.get("optimizer_type", loaded["optimizer"])
    loaded["training_steps"] = int(training_cfg.get("max_train_steps", loaded["training_steps"]))
    loaded["save_steps"] = int(training_cfg.get("save_every_n_steps", loaded["save_steps"]))
    loaded["sample_steps"] = int(training_cfg.get("sample_every_n_steps", loaded["sample_steps"]))
    loaded["train_seed"] = int(training_cfg.get("seed", loaded["train_seed"]))
    loaded["train_batch_size"] = int(training_cfg.get("train_batch_size", loaded["train_batch_size"]))
    loaded["gradient_accumulation_steps"] = int(training_cfg.get("gradient_accumulation_steps", loaded["gradient_accumulation_steps"]))

    prompt_path = Path(training_cfg.get("sample_prompts", "")) if training_cfg.get("sample_prompts") else (prompt_files[0] if prompt_files else None)
    if prompt_path:
        prompts, neg, w, h, steps, cfg, seed = _read_prompt_file(prompt_path)
        cleaned_prompts = []
        for pos in prompts[:MAX_PROMPTS]:
            if pos.startswith(loaded["trigger_word"] + ", "):
                pos = pos[len(loaded["trigger_word"]) + 2:]
            cleaned_prompts.append(pos)
        loaded["pos_prompt"] = cleaned_prompts[0] if cleaned_prompts else ""
        for i in range(2, MAX_PROMPTS + 1):
            loaded[f"pos_prompt_{i}"] = cleaned_prompts[i - 1] if len(cleaned_prompts) >= i else ""
        loaded["neg_prompt"] = neg
        loaded["width"] = w
        loaded["height"] = h
        loaded["sample_steps_gen"] = steps
        loaded["sample_cfg"] = cfg
        loaded["sample_seed"] = seed

    save_settings(loaded)
    return settings_to_values(loaded) + [f"✅ Loaded project: {project_dir.name}"]

def refresh_project_choices():
    choices = list_output_projects()
    value = choices[0] if choices else None
    return gr.update(choices=choices, value=value)


HIDDEN_SETTINGS = {
    "lr_scheduler": "cosine",
    "mixed_precision": "bf16",
    "save_precision": "bf16",
    "gradient_checkpointing": True,
    "network_module": "networks.lora_anima",
    "network_train_unet_only": True,
    "timestep_sampling": "logit_normal",
    "discrete_flow_shift": 3.0,
    "cache_latents": True,
    "cache_latents_to_disk": True,
    "cache_text_encoder_outputs_to_disk": True,
    "cache_text_encoder_outputs": True,
    "sdpa": True,
    "weighting_scheme": "logit_normal",
    "max_data_loader_n_workers": 4,
    "persistent_data_loader_workers": True,
    "max_grad_norm": 1.0,
    "vae_batch_size": 1,
    "blocks_to_swap": 0
}


def analyze_dataset_resolution(dataset_path):
    default_base = 512
    default_max_bucket = 1024

    if not dataset_path or not os.path.exists(dataset_path):
        return default_base, default_max_bucket

    valid_exts = {'.png', '.jpg', '.jpeg', '.webp'}
    image_files = [f for f in Path(dataset_path).rglob('*') if f.suffix.lower() in valid_exts]

    if not image_files:
        return default_base, default_max_bucket

    areas = []
    max_side = 0
    for img_path in image_files:
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                areas.append(w * h)
                max_side = max(max_side, w, h)
        except Exception: pass 

    if not areas: return default_base, default_max_bucket

    areas.sort()
    median_area = areas[len(areas) // 2]
    
    equivalent_side = math.sqrt(median_area)
    base_res = int(round(equivalent_side / 64.0) * 64)
    base_res = max(512, min(1024, base_res))

    max_side_rounded = int(math.ceil(max_side / 64.0) * 64)
    max_bucket = max(base_res + 256, max_side_rounded)
    max_bucket = min(1536, max_bucket) 

    return base_res, max_bucket


def create_sample_prompts(project_name, trigger_word, prompts, neg_prompt, width, height, steps_gen, cfg, seed, out_dir):
    prompt_path = out_dir / f"{project_name}_prompts.txt"
    trigger = trigger_word.strip()

    actual_neg = neg_prompt.strip().replace("\n", " ")
    prompt_lines = []
    for prompt in prompts:
        user_prompt = prompt.strip().replace("\n", " ")
        if not user_prompt and not trigger:
            continue
        if trigger and not user_prompt.startswith(trigger):
            actual_pos = f"{trigger}, {user_prompt}" if user_prompt else trigger
        else:
            actual_pos = user_prompt if user_prompt else trigger
        prompt_lines.append(
            f"{actual_pos} --n {actual_neg} --w {int(width)} --h {int(height)} --l {float(cfg)} --s {int(steps_gen)} --d {int(seed)}"
        )
    if not prompt_lines:
        prompt_lines.append(
            f"{trigger} --n {actual_neg} --w {int(width)} --h {int(height)} --l {float(cfg)} --s {int(steps_gen)} --d {int(seed)}"
        )
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(prompt_lines))
    return str(prompt_path)

def create_dataset_toml(project_name, dataset_path, trigger_word, base_res, max_bucket, out_dir):
    config_path = out_dir / f"{project_name}_dataset.toml"
    prefix = f"{trigger_word.strip()}, " if trigger_word.strip() else None
    dataset_config = {
        "general": {"enable_bucket": True, "min_bucket_reso": 256, "max_bucket_reso": max_bucket, "bucket_reso_steps": 64, "bucket_no_upscale": False},
        "datasets": [{
            "resolution": base_res, 
            "subsets": [{"image_dir": Path(dataset_path).resolve().as_posix(), "caption_extension": ".txt", "num_repeats": 1000, "caption_prefix": prefix, "keep_tokens": 1, "caption_dropout_rate": 0.05}]
        }]
    }
    with open(config_path, "w", encoding="utf-8") as f: toml.dump(dataset_config, f)
    return str(config_path)

def create_training_toml(project_name, config_save_dir, actual_output_dir, rank, lr, optimizer, max_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc):
    config_path = config_save_dir / f"{project_name}_training.toml"
    network_alpha = max(1, int(rank) // 2)
    
    opt_args = ["weight_decay=0.01"]
    if optimizer == "Prodigy":
        scheduler = "constant"
        opt_args = ["decouple=True", "weight_decay=0.01", "d_coef=1", "use_bias_correction=True", "safeguard_warmup=True", "betas=0.9,0.99"]
    else:
        scheduler = "cosine"

    training_config = {
        "pretrained_model_name_or_path": Path(models["dit_path"]).resolve().as_posix(),
        "qwen3": Path(models["qwen_path"]).resolve().as_posix(),
        "vae": Path(models["vae_path"]).resolve().as_posix(),
        "network_module": HIDDEN_SETTINGS["network_module"],
        "network_dim": int(rank),
        "network_alpha": network_alpha,
        "network_train_unet_only": HIDDEN_SETTINGS["network_train_unet_only"],
        "gradient_checkpointing": HIDDEN_SETTINGS["gradient_checkpointing"],
        "learning_rate": float(lr),
        "optimizer_type": optimizer,
        "optimizer_args": opt_args,
        "lr_scheduler": scheduler,
        "max_train_steps": int(max_steps),
        "train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_acc),
        "mixed_precision": HIDDEN_SETTINGS["mixed_precision"],
        "output_dir": actual_output_dir.resolve().as_posix(),
        "output_name": project_name,
        "save_every_n_steps": int(save_steps),
        "save_state": True,
        "save_state_on_train_end": True,
        "sample_every_n_steps": int(sample_steps),
        "sample_prompts": Path(prompt_path).resolve().as_posix(),
        "sample_sampler": "euler",
        "timestep_sampling": HIDDEN_SETTINGS["timestep_sampling"],
        "discrete_flow_shift": HIDDEN_SETTINGS["discrete_flow_shift"],
        "weighting_scheme": HIDDEN_SETTINGS["weighting_scheme"],
        "cache_latents": True,
        "cache_latents_to_disk": True,
        "cache_text_encoder_outputs": True,
        "cache_text_encoder_outputs_to_disk": True,
        "attn_mode": "sdpa",
        "save_model_as": "safetensors",
        "save_precision": "bf16",
        "max_data_loader_n_workers": 4,
        "vae_chunk_size": 32,
        "vae_disable_cache": True,
        "seed": int(train_seed),
    }
    with open(config_path, "w", encoding="utf-8") as f: toml.dump(training_config, f)
    return str(config_path)


def get_latest_images(sample_dir):
    if not sample_dir.exists(): return []
    images = glob.glob(str(sample_dir / "*.png")) + glob.glob(str(sample_dir / "*.jpg")) + glob.glob(str(sample_dir / "*.webp"))
    images.sort(key=os.path.getmtime, reverse=True)
    
    return [(img, Path(img).name) for img in images]

def get_latest_checkpoint(project_out_dir, project_name):
    pattern = re.compile(r"(\d+)(?!.*\d)")
    latest_step = -1
    latest_ckpt = None
    for ckpt in project_out_dir.glob(f"{project_name}*.safetensors"):
        m = pattern.search(ckpt.stem)
        if not m:
            continue
        step = int(m.group(1))
        if step > latest_step:
            latest_step = step
            latest_ckpt = ckpt
    return latest_ckpt, max(0, latest_step)

def get_latest_state_dir(project_out_dir, project_name):
    pattern = re.compile(r"step(\d+)-state$")
    latest_step = -1
    latest_state = None
    for state_dir in project_out_dir.glob(f"{project_name}-step*-state"):
        if not state_dir.is_dir():
            continue
        m = pattern.search(state_dir.name)
        if not m:
            continue
        step = int(m.group(1))
        if step > latest_step:
            latest_step = step
            latest_state = state_dir
    return latest_state, max(0, latest_step)

def start_training(trigger_word, project_name, dataset_path, dit_p, qwen_p, vae_p, rank, lr, optimizer, t_steps, save_steps, sample_steps, pos, pos2, pos3, pos4, pos5, neg, w, h, s_steps_gen, s_cfg, s_seed, train_seed, batch_size, grad_acc):
    global training_process

     # --- PATH VALIDATION BLOCK ---
    error_messages = []
    
    # Check DiT file
    if not dit_p or not os.path.isfile(dit_p):
        error_messages.append(f"❌ DiT file not found: {dit_p}")
    
    # Check Qwen file
    if not qwen_p or not os.path.isfile(qwen_p):
        error_messages.append(f"❌ Qwen3 file not found: {qwen_p}")
        
    # Check VAE file
    if not vae_p or not os.path.isfile(vae_p):
        error_messages.append(f"❌ VAE file not found: {vae_p}")

    # Check Dataset directory
    if not dataset_path or not os.path.exists(dataset_path):
        error_messages.append(f"❌ Dataset path not found: {dataset_path}")

    if error_messages:
        full_error = "\n".join(error_messages)
        full_error += "\n\n⚠️ ERROR: Please check and set the correct model paths in the section:\n'🔧 Paths to Models <- Set Once'"
        yield full_error, gr.update()
        return
    # --- END VALIDATION BLOCK ---

    # --- CUDA CHECK BLOCK ---
    if not torch.cuda.is_available():
        cuda_error = (
            "❌ ERROR: NVIDIA GPU not detected or CUDA drivers are not installed!\n\n"
            "Technical details:\n"
            "- PyTorch cannot initialize CUDA.\n"
            "- Training on CPU is extremely slow and is not supported by this script.\n\n"
            "Please update your NVIDIA drivers and restart the app."
        )
        yield cuda_error, gr.update()
        return
    # --- END CUDA CHECK ---
    
    if training_process is not None and training_process.poll() is None:
        yield "⚠️ Training is already running!", gr.update()
        return

    project_name = re.sub(r'[^a-zA-Z0-9]', '_', project_name.strip()).strip('_') or "untitled"

    project_out_dir = OUTPUT_BASE / project_name
    sample_dir = project_out_dir / "sample"
    project_configs_dir = project_out_dir / "configs"

    for d in [project_out_dir, sample_dir, project_configs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    log_lines = [f"🚀 Preparing: {project_name}..."]
    last_image_count = 0
    step_pattern = re.compile(r"(\d+)/(\d+)")
    max_steps = int(t_steps)
    latest_ckpt, ckpt_steps = get_latest_checkpoint(project_out_dir, project_name)
    latest_state_dir, state_steps = get_latest_state_dir(project_out_dir, project_name)
    completed_steps = max(ckpt_steps, state_steps)
    
    yield "\n".join(log_lines), gr.update()

    if completed_steps > 0:
        if completed_steps >= max_steps:
            log_lines.append(f"✅ Already complete: checkpoint at step {completed_steps}, max steps is {max_steps}.")
            yield "\n".join(log_lines), get_latest_images(sample_dir)
            return
        if latest_state_dir is not None and state_steps >= ckpt_steps:
            log_lines.append(f"↪ Resuming from state step {completed_steps}: {latest_state_dir.name}")
        elif latest_ckpt is not None:
            log_lines.append(f"↪ Resuming from checkpoint step {completed_steps}: {latest_ckpt.name}")
    else:
        log_lines.append("ℹ️ No checkpoint/state found. Starting from step 0.")

    log_lines.append(f"🔍 Analyzing dataset images...")
    base_res, max_bucket = analyze_dataset_resolution(dataset_path)
    log_lines.append(f"📐 Auto-Resolution Set: Base {base_res}px, Max Bucket {max_bucket}px")

    models = {"dit_path": dit_p, "qwen_path": qwen_p, "vae_path": vae_p}
    prompt_path = create_sample_prompts(project_name, trigger_word, [pos, pos2, pos3, pos4, pos5], neg, w, h, s_steps_gen, s_cfg, s_seed, project_configs_dir)
    dataset_toml = create_dataset_toml(project_name, dataset_path, trigger_word, base_res, max_bucket, project_configs_dir)
    training_toml = create_training_toml(project_name, project_configs_dir, project_out_dir, rank, lr, optimizer, t_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc)

    launch_errors = []
    if not TRAIN_PYTHON.exists():
        launch_errors.append(f"Python executable not found: {TRAIN_PYTHON}")
    if not TRAIN_SCRIPT.exists():
        launch_errors.append(f"Training script not found: {TRAIN_SCRIPT}")
    if importlib.util.find_spec("accelerate") is None:
        launch_errors.append("Python module not found: accelerate")

    if launch_errors:
        log_lines.append("❌ Launch pre-check failed:")
        for msg in launch_errors:
            log_lines.append(f"- {msg}")
        log_lines.append("Run setup again in this same venv, then relaunch.")
        yield "\n".join(log_lines), gr.update()
        return

    cmd = [
        str(TRAIN_PYTHON), "-m", "accelerate.commands.launch", "--num_processes=1", "--mixed_precision=bf16", "--dynamo_backend=no",
        TRAIN_SCRIPT.resolve().as_posix(), 
        "--config_file", Path(training_toml).resolve().as_posix(), 
        "--dataset_config", Path(dataset_toml).resolve().as_posix(),
        "--sample_at_first",
    ]
    if latest_state_dir is not None and completed_steps > 0 and completed_steps < max_steps:
        cmd += [
            "--resume", latest_state_dir.resolve().as_posix(),
        ]
    elif latest_ckpt is not None and completed_steps > 0 and completed_steps < max_steps:
        cmd += [
            "--network_weights", latest_ckpt.resolve().as_posix(),
            "--initial_step", str(completed_steps),
            "--skip_until_initial_step",
        ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(TRAIN_DIR.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
    env["PYTHONIOENCODING"] = "utf-8"

    env["PYTHONWARNINGS"] = "ignore"
    env["TORCH_CPP_LOG_LEVEL"] = "ERROR"
    env["KMP_WARNINGS"] = "0"

    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["ACCELERATE_USE_CPU"] = "False"

    try:
        training_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, bufsize=1, cwd=str(TRAIN_DIR.resolve()), env=env, encoding="utf-8", errors="ignore")
        
        
        last_progress_idx = -1
        for line in iter(training_process.stdout.readline, ""):
            line_str = line.replace('\r', '').strip()
            if not line_str: continue

            
            if any(skip_word in line_str for skip_word in LOG_BLACKLIST):
                continue 

            
            if "subprocess.CalledProcessError" in line_str and "returned non-zero exit status 15" in line_str:
                log_lines.append("🛑 Interrupted by user.")
                yield "\n".join(log_lines), gr.update()
                break 

           
            is_progress_line = (
                "steps:" in line_str
                or "%|" in line_str
                or "it/s" in line_str
                or "s/it" in line_str
            )
            if is_progress_line:
                match = step_pattern.search(line_str)
                if match:
                    current_step_info = match.group(0)
                    if completed_steps > 0:
                        current_rel = int(match.group(1))
                        total_rel = int(match.group(2))
                        current_abs = min(max_steps, completed_steps + current_rel)
                        total_abs = max(completed_steps + total_rel, max_steps)
                        current_step_info = f"{current_abs}/{total_abs}"
                        line_str = line_str.replace(match.group(0), current_step_info, 1)
                    if (
                        0 <= last_progress_idx < len(log_lines)
                        and current_step_info in log_lines[last_progress_idx]
                    ):
                        log_lines[last_progress_idx] = line_str
                    else:
                        log_lines.append(line_str)
                        last_progress_idx = len(log_lines) - 1
                else:
                    if 0 <= last_progress_idx < len(log_lines):
                        log_lines[last_progress_idx] = line_str
                    else:
                        log_lines.append(line_str)
                        last_progress_idx = len(log_lines) - 1
            else:
                log_lines.append(line_str)
                last_progress_idx = -1
            
            if len(log_lines) > MAX_LOG_LINES: del log_lines[:-MAX_LOG_LINES]

            current_images = get_latest_images(sample_dir)
            if len(current_images) != last_image_count:
                last_image_count = len(current_images)
                yield "\n".join(log_lines), current_images
                continue

            yield "\n".join(log_lines), gr.update()
            
        if training_process is not None:
            training_process.wait()
            
        log_lines.append("✅ Process finished or stopped.")
        yield "\n".join(log_lines), get_latest_images(sample_dir)
        
    except Exception as e:
        log_lines.append(f"❌ Error: {str(e)}")
        yield "\n".join(log_lines), gr.update()
    finally:
        training_process = None

def stop_training():
    global training_process
    if training_process is not None:
        try:
            parent = psutil.Process(training_process.pid)
            for child in parent.children(recursive=True):
                try: child.terminate()
                except psutil.NoSuchProcess: pass
            gone, alive = psutil.wait_procs(parent.children(recursive=True), timeout=3)
            for survival in alive:
                try: survival.kill()
                except psutil.NoSuchProcess: pass
            parent.terminate()
            parent.wait(timeout=3)
            return "🛑 Stopping training... (Clearing VRAM)"
        except psutil.NoSuchProcess:
            return "ℹ️ Process already finished."
        except Exception as e:
            return f"⚠️ Error during stop: {str(e)}"
    return "ℹ️ Not running."

def handle_optimizer_change(opt, current_lr, saved_adam_lr):
    if opt == "Prodigy": return "1.0", current_lr
    return (saved_adam_lr if current_lr == "1.0" else current_lr), saved_adam_lr


cs = load_settings()

with gr.Blocks(title="Anima TrainFlow: Easy LoRA Trainer for Anima 2B") as ui:
    gr.Markdown(
        "# Anima TrainFlow",
        elem_id="main-header"
    )
    
    saved_adam_lr = gr.State(value="0.00005")
    prompt_count_state = gr.Number(value=get_prompt_count(cs), visible=False, precision=0)

    with gr.Group():
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    project_name = gr.Textbox(label="Project Name", value=cs.get("project_name", ""), placeholder="e.g., anima_character_v1", lines=1, max_lines=1)
                    trigger_word = gr.Textbox(label="Trigger Word", value=cs.get("trigger_word", ""), placeholder="e.g., unique_style", lines=1, max_lines=1)
                with gr.Row():
                    dataset_path = gr.Textbox(label="Dataset Path (Images + .txt)", value=cs.get("dataset_path", ""), placeholder="C:/Images/MyDataset", lines=1, max_lines=1)
                    project_picker = gr.Dropdown(
                        label="Project Picker",
                        choices=list_output_projects(),
                        value=cs.get("project_name", None),
                        allow_custom_value=True,
                    )
                with gr.Row():
                    start_btn = gr.Button("🚀 Start", variant="primary")
                    stop_btn = gr.Button("🛑 Stop", variant="stop")
                    load_project_btn = gr.Button("🔄 Refresh Picker", variant="secondary", min_width=180)
            with gr.Column(scale=1):
                with gr.Row():
                    rank_input = gr.Number(label="Network Rank", value=cs.get("network_rank", 16), precision=0)
                    lr_input = gr.Textbox(label="Learning Rate", value=cs.get("learning_rate", "1.0"))
                    optimizer_input = gr.Dropdown(label="Optimizer", choices=["Prodigy", "AdamW8bit", "AdamW"], value=cs.get("optimizer", "Prodigy"))
                    train_seed_val = gr.Number(value=cs.get("train_seed", 42), visible=False)
                    batch_size_input = gr.Number(label="Batch Size", value=cs.get("train_batch_size", 1), precision=0)
                with gr.Row():
                    steps_input = gr.Number(label="Max Training Steps", value=cs.get("training_steps", 2400), precision=0)
                    save_steps_input = gr.Number(label="Save x Steps", value=cs.get("save_steps", 300), precision=0)
                    sample_steps_input = gr.Number(label="Preview x Steps", value=cs.get("sample_steps", 300), precision=0)
                    grad_acc_input = gr.Number(label="Gradient Accum.", value=cs.get("gradient_accumulation_steps", 1), precision=0)
                with gr.Accordion("🔧 Paths to Models <- Set Once", open=False):
                    dit_input = gr.Textbox(label="DiT", value=cs.get("dit_path", ""), lines=1, max_lines=1)
                    qwen_input = gr.Textbox(label="Qwen3", value=cs.get("qwen_path", ""), lines=1, max_lines=1)
                    vae_input = gr.Textbox(label="VAE", value=cs.get("vae_path", ""), lines=1, max_lines=1)

    with gr.Row():
        with gr.Column(scale=1):
            output_log = gr.Textbox(label="Logs", lines=LOG_BOX__MAX_LINES, max_lines=LOG_BOX__MAX_LINES, interactive=False, autoscroll=True, elem_id="log-container")
        with gr.Column(scale=1):
            preview_gallery = gr.Gallery(label="Previews", columns=2, rows=2, height=GALLERY_HEIGHT, object_fit="contain")
            with gr.Group():
                with gr.Row():
                    pos_prompt = gr.Textbox(
                        label="Prompt 1 (Trigger word added automatically)",
                        lines=2,
                        value=cs.get("pos_prompt", ""),
                        scale=20,
                    )
                    add_prompt_btn = gr.Button(
                        "+ Add Prompt",
                        variant="secondary",
                        min_width=130,
                        scale=1,
                        interactive=get_prompt_count(cs) < MAX_PROMPTS,
                    )
                pos_prompt_2 = gr.Textbox(label="Prompt 2", lines=2, value=cs.get("pos_prompt_2", ""), visible=get_prompt_count(cs) >= 2)
                pos_prompt_3 = gr.Textbox(label="Prompt 3", lines=2, value=cs.get("pos_prompt_3", ""), visible=get_prompt_count(cs) >= 3)
                pos_prompt_4 = gr.Textbox(label="Prompt 4", lines=2, value=cs.get("pos_prompt_4", ""), visible=get_prompt_count(cs) >= 4)
                pos_prompt_5 = gr.Textbox(label="Prompt 5", lines=2, value=cs.get("pos_prompt_5", ""), visible=get_prompt_count(cs) >= 5)
                neg_prompt = gr.Textbox(label="Negative Prompt", lines=1, value=cs.get("neg_prompt", ""))
                with gr.Row():
                    width_input = gr.Number(label="Width", value=cs.get("width", 1024), precision=0, min_width=80)
                    height_input = gr.Number(label="Height", value=cs.get("height", 1024), precision=0, min_width=80)
                    sample_steps_gen_input = gr.Number(label="Steps", value=cs.get("sample_steps_gen", 30), precision=0, min_width=80)
                    sample_cfg_input = gr.Number(label="CFG", value=cs.get("sample_cfg", 4.0), min_width=80)
                    sample_seed_input = gr.Number(label="Seed", value=cs.get("sample_seed", 42), precision=0, min_width=80)

    inputs_list = [
        trigger_word, project_name, dataset_path, dit_input, qwen_input, vae_input,
        rank_input, lr_input, optimizer_input, 
        steps_input, save_steps_input, sample_steps_input,
        pos_prompt, pos_prompt_2, pos_prompt_3, pos_prompt_4, pos_prompt_5, neg_prompt, width_input, height_input,
        sample_steps_gen_input, sample_cfg_input, sample_seed_input, train_seed_val, batch_size_input, grad_acc_input
    ]

    def load_state_on_refresh():
        current_settings = load_settings()
        return settings_to_values(current_settings)

    ui.load(fn=load_state_on_refresh, inputs=None, outputs=inputs_list)    

    output_log.change(None, None, None, js=JS_SCROLL)
    optimizer_input.change(fn=handle_optimizer_change, inputs=[optimizer_input, lr_input, saved_adam_lr], outputs=[lr_input, saved_adam_lr])
    for comp in inputs_list: comp.change(fn=auto_save_state, inputs=inputs_list)

    start_btn.click(fn=start_training, inputs=inputs_list, outputs=[output_log, preview_gallery])
    stop_btn.click(fn=stop_training, outputs=output_log)
    load_project_btn.click(fn=refresh_project_choices, inputs=None, outputs=[project_picker])
    project_picker.change(fn=load_project_config, inputs=[project_picker], outputs=inputs_list + [output_log])
    load_project_btn.click(
        fn=compute_prompt_ui_updates,
        inputs=inputs_list,
        outputs=[prompt_count_state, pos_prompt_2, pos_prompt_3, pos_prompt_4, pos_prompt_5, add_prompt_btn],
    )
    project_picker.change(
        fn=compute_prompt_ui_updates,
        inputs=inputs_list,
        outputs=[prompt_count_state, pos_prompt_2, pos_prompt_3, pos_prompt_4, pos_prompt_5, add_prompt_btn],
    )
    add_prompt_btn.click(
        fn=add_prompt_row,
        inputs=[prompt_count_state],
        outputs=[prompt_count_state, pos_prompt_2, pos_prompt_3, pos_prompt_4, pos_prompt_5, add_prompt_btn],
    )

if __name__ == "__main__":
     ui.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)
