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

/* Remove the up/down spinner "ticker" from number inputs. */
input[type=number]::-webkit-inner-spin-button,
input[type=number]::-webkit-outer-spin-button {
    -webkit-appearance: none !important;
    margin: 0 !important;
}
input[type=number] {
    -moz-appearance: textfield !important;
}

#btn-start button, #btn-stop button, #btn-new button, #btn-clone button, #btn-save button, #btn-refresh-picker button {
    padding-left: 0.5rem !important;
    padding-right: 0.5rem !important;
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

# Mouseover tooltips. Keyed by elem_id; edit text here. Use "\n" for line breaks
# (the title attribute renders them). "tt-lr" is set dynamically per optimizer in
# the UI block. Applied to every matching component via make_init_tooltips_js().
TOOLTIPS = {
    # Action buttons
    "btn-start": "Start training with\nthe current settings.",
    "btn-stop": "Stop the current\ntraining process.",
    "btn-refresh-picker": "Refresh the\nproject list.",
    "btn-open": "Open the current project\nfolder in Explorer.",
    "btn-save": "Save current settings.\nRenames the project folder\nif the name changed.",
    # Left card — project
    "tt-project-name": "Name of this project.\nUsed for the output folder\nand file names.",
    "tt-trigger-word": "Trigger word automatically\nprepended to every caption\nand sample prompt.",
    "tt-dataset-path": "Folder containing your\ntraining images and matching\n.txt caption files.",
    "tt-project-picker": "Switch between existing\nprojects, or pick\n'➕ New Project' to create one.",
    # Right card — training hyperparameters
    "tt-rank": "LoRA network rank (dim).\nHigher = more capacity and\nlarger files.\nAlpha is set to rank / 2.",
    "tt-optimizer": "Optimizer algorithm.\nPicking one fills in its\ndefault learning rate, warmup,\nand updates the Auto scheduler.",
    "tt-lr": "Learning rate.",  # overwritten per-optimizer in the UI block
    "tt-batch-size": "Images processed per step.\nHigher needs more VRAM.",
    "tt-scheduler": "Learning-rate schedule.\n'Auto' uses the optimizer's\nrecommended default.\n'cosine_with_restarts' is the\noscillating wave (sine-like)\nschedule.",
    "tt-warmup": "Warmup before the LR scheduler\nramps in, as % of total steps.\n\nProdigy: keep at 0 — warmup\nis handled internally via\nsafeguard_warmup.\nAdamW / CAME: 5% recommended.",
    "tt-grad-acc": "Gradient accumulation steps.\nSimulates a larger batch\nwithout extra VRAM.",
    "tt-steps": "Total training\nsteps to run.",
    "tt-save-steps": "Save a checkpoint\nevery N steps.",
    "tt-sample-steps": "Generate preview sample\nimages every N steps.",
    "tt-dit": "Path to the DiT (anima)\nmodel file.\nSet once; shared\nacross projects.",
    "tt-qwen": "Path to the Qwen3\ntext-encoder file.\nSet once; shared\nacross projects.",
    "tt-vae": "Path to the VAE file.\nSet once; shared\nacross projects.",
    # Sampling (preview generation) settings
    "tt-neg-prompt": "Negative prompt applied\nto all preview samples.",
    "tt-width": "Width of generated\npreview samples.",
    "tt-height": "Height of generated\npreview samples.",
    "tt-gen-steps": "Inference steps used when\ngenerating preview samples.",
    "tt-gen-cfg": "Classifier-free guidance\n(CFG) scale for\npreview samples.",
    "tt-gen-seed": "Seed for preview generation.\nSame seed = reproducible\npreviews.",
}

# Shared JS body that stamps the title attribute on a component's wrapper AND on
# every inner control, so hovering either the label or the value field shows it.
_JS_APPLY_TITLE = """
        const root = document.getElementById(id);
        if (!root) return;
        root.setAttribute("title", tip);
        root.querySelectorAll("button, input, textarea, select, .wrap, label").forEach(el => {
            el.setAttribute("title", tip);
        });
"""

def make_init_tooltips_js():
    """Built at use-time so per-optimizer edits to TOOLTIPS['tt-lr'] are picked up."""
    return ("""
function() {
    const tips = __TIPS__;
    Object.entries(tips).forEach(([id, tip]) => {""" + _JS_APPLY_TITLE + """
    });
}
""").replace("__TIPS__", json.dumps(TOOLTIPS))


LOG_BLACKLIST = [
    "triton not found",
    "flop counting will not work",
    "Lib\\site-packages\\torch\\utils\\flop_counter.py"
]


NEW_PROJECT_SENTINEL = "➕ New Project"

LOG_BOX__MAX_LINES = 16
GALLERY_HEIGHT = 440
MAX_LOG_LINES = 500
MAX_PROMPTS = 5


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZER PROFILES — single source of truth for per-optimizer defaults.
#
# Edit values here to change what "Auto" resolves to. Every change propagates to
# BOTH the generated training TOML (when scheduler/LR are left on Auto) AND the
# "Auto (...)" labels shown in the UI. Nothing else needs to be touched.
#
#   lr             : default learning rate for this optimizer
#   scheduler      : LR scheduler used when the UI scheduler is left on "Auto"
#   optimizer_args : optimizer_args written into the training TOML
# ─────────────────────────────────────────────────────────────────────────────
OPTIMIZER_PROFILES = {
    "Prodigy": {
        "lr": "1.0",
        "scheduler": "constant",
        "warmup": 0,        # safeguard_warmup=True handles this internally
        "optimizer_args": [
            "decouple=True", "weight_decay=0.01", "d_coef=1",
            "use_bias_correction=True", "safeguard_warmup=True", "betas=0.9,0.99",
        ],
    },
    "AdamW8bit": {
        "lr": "0.00005",
        "scheduler": "cosine",
        "warmup": 5,        # 5% of total steps; standard for cosine schedules
        "optimizer_args": ["weight_decay=0.01"],
    },
    "AdamW": {
        "lr": "0.00005",
        "scheduler": "cosine",
        "warmup": 5,
        "optimizer_args": ["weight_decay=0.01"],
    },
    "CAME": {
        "lr": "0.0002",
        "scheduler": "cosine",
        "warmup": 5,
        "optimizer_args": ["weight_decay=0.01", "betas=0.9,0.999,0.9999"],
    },
}
DEFAULT_OPTIMIZER = "Prodigy"
OPTIMIZER_CHOICES = list(OPTIMIZER_PROFILES.keys())

# Sentinel stored when scheduler is left on "Auto" (resolved at TOML-write time).
AUTO_SENTINEL = "auto"

# Schedulers selectable in the UI (besides the Auto entry). These are the names
# sd-scripts' get_scheduler_fix() understands. "cosine_with_restarts" is the
# oscillating / wave ("sine-like") schedule — pair it with warmup as desired.
SCHEDULER_CHOICES = [
    "constant",
    "constant_with_warmup",
    "linear",
    "cosine",
    "cosine_with_restarts",
    "polynomial",
]

# Global default warmup (percentage of total steps, 0 = no warmup).
DEFAULT_LR_WARMUP_STEPS = 0


def optimizer_default_lr(opt):
    return OPTIMIZER_PROFILES.get(opt, {}).get("lr", "1.0")

def optimizer_default_scheduler(opt):
    return OPTIMIZER_PROFILES.get(opt, {}).get("scheduler", "cosine")

def optimizer_default_warmup(opt):
    return OPTIMIZER_PROFILES.get(opt, {}).get("warmup", 5)

def optimizer_args_for(opt):
    return list(OPTIMIZER_PROFILES.get(opt, {}).get("optimizer_args", ["weight_decay=0.01"]))

def resolve_scheduler(scheduler_setting, optimizer):
    """Map the UI scheduler value to a concrete scheduler name.
    An empty value or the Auto sentinel resolves to the optimizer's profile default."""
    if not scheduler_setting or scheduler_setting == AUTO_SENTINEL:
        return optimizer_default_scheduler(optimizer)
    return scheduler_setting

def parse_warmup(value):
    """Warmup is a percentage (0–100). Converts to a ratio for the training TOML."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0
    if f <= 0:
        return 0
    return round(f / 100, 6)

def scheduler_dropdown_choices(opt):
    """(label, value) pairs for the scheduler dropdown, with a live Auto label."""
    return [(f"Auto ({optimizer_default_scheduler(opt)})", AUTO_SENTINEL)] + [(s, s) for s in SCHEDULER_CHOICES]

def lr_tooltip(opt):
    """Per-optimizer learning-rate tooltip (multi-line). The LR is a free numeric
    field (not a dropdown), so its recommended default is surfaced here."""
    return (
        "Learning rate.\n"
        f"Recommended for {opt}: {optimizer_default_lr(opt)}.\n"
        "Switching optimizer fills in\nits recommended value."
    )

# JS that re-stamps the LR field's tooltip when the optimizer changes. Fires on
# .change(), so it also covers programmatic changes from loading a project.
JS_LR_TOOLTIP = ("""
(opt) => {
    const map = __LRMAP__;
    const tip = map[opt];
    if (tip === undefined) return [];
    const root = document.getElementById("tt-lr");
    if (root) {
        root.setAttribute("title", tip);
        root.querySelectorAll("input, textarea, label, .wrap").forEach(el => el.setAttribute("title", tip));
    }
    return [];
}
""").replace("__LRMAP__", json.dumps({o: lr_tooltip(o) for o in OPTIMIZER_CHOICES}))


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
    "lr_scheduler": "auto", # "auto" => use the optimizer profile's default scheduler
    "lr_warmup_steps": 0,   # percentage of total steps (0 = no warmup)
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

def write_project_configs(settings_dict):
    project_name_raw = settings_dict.get("project_name", "").strip()
    if not project_name_raw:
        return
    project_name_clean = re.sub(r'[^a-zA-Z0-9_]', '_', project_name_raw).strip('_') or "untitled"
    project_out_dir = OUTPUT_BASE / project_name_clean
    project_configs_dir = project_out_dir / "configs"
    for d in [project_out_dir, project_out_dir / "sample", project_configs_dir]:
        d.mkdir(parents=True, exist_ok=True)
    pos_prompts = [settings_dict.get(k, "") for k in ["pos_prompt", "pos_prompt_2", "pos_prompt_3", "pos_prompt_4", "pos_prompt_5"]]
    models = {"dit_path": settings_dict.get("dit_path", ""), "qwen_path": settings_dict.get("qwen_path", ""), "vae_path": settings_dict.get("vae_path", "")}
    dataset_path = settings_dict.get("dataset_path", "")
    base_res, max_bucket = analyze_dataset_resolution(dataset_path)
    prompt_path = create_sample_prompts(project_name_clean, settings_dict.get("trigger_word", ""), pos_prompts, settings_dict.get("neg_prompt", ""), settings_dict.get("width", 1024), settings_dict.get("height", 1024), settings_dict.get("sample_steps_gen", 30), settings_dict.get("sample_cfg", 4.0), settings_dict.get("sample_seed", 42), project_configs_dir)
    create_dataset_toml(project_name_clean, dataset_path, settings_dict.get("trigger_word", ""), base_res, max_bucket, project_configs_dir)
    create_training_toml(project_name_clean, project_configs_dir, project_out_dir, settings_dict.get("network_rank", 32), settings_dict.get("learning_rate", "1.0"), settings_dict.get("optimizer", "Prodigy"), settings_dict.get("training_steps", 2400), settings_dict.get("save_steps", 300), settings_dict.get("sample_steps", 300), models, prompt_path, settings_dict.get("train_seed", 42), settings_dict.get("train_batch_size", 1), settings_dict.get("gradient_accumulation_steps", 1), settings_dict.get("lr_scheduler", "auto"), settings_dict.get("lr_warmup_steps", 0))

def _rename_prefixed_files(directory, old_prefix, new_prefix):
    """Rename all files/dirs inside `directory` whose names start with `old_prefix`."""
    if not directory.exists():
        return
    for entry in list(directory.iterdir()):
        if entry.name.startswith(old_prefix):
            new_entry_name = new_prefix + entry.name[len(old_prefix):]
            entry.rename(directory / new_entry_name)

def rename_project_contents(project_dir, old_name, new_name):
    """Rename every file/dir inside a project folder that is prefixed with old_name."""
    _rename_prefixed_files(project_dir / "configs", old_name, new_name)
    _rename_prefixed_files(project_dir / "sample", old_name, new_name)
    _rename_prefixed_files(project_dir, old_name, new_name)

def _detect_old_prefix(project_dir, fallback):
    """Find the filename prefix actually used by a project's config files.
    It can differ from the folder name (e.g. a folder duplicated in Explorer as
    'X - Copy' still holds files named 'X_training.toml')."""
    configs = project_dir / "configs"
    if configs.exists():
        for suffix in ("_training.toml", "_dataset.toml", "_prompts.txt"):
            for f in sorted(configs.glob(f"*{suffix}")):
                return f.name[: -len(suffix)]
    return fallback

def validate_project_name(raw_name):
    """Return (sanitized_name, error_msg). error_msg is None when valid."""
    raw_name = (raw_name or "").strip()
    if not raw_name:
        return None, "⚠️ Project name cannot be empty."
    sanitized = re.sub(r'[^a-zA-Z0-9_]', '_', raw_name).strip('_')
    if not sanitized:
        return None, f"⚠️ '{raw_name}' contains no valid characters. Use letters, numbers, or underscores."
    return sanitized, None

def _clean_name(name):
    return re.sub(r'[^a-zA-Z0-9_]', '_', (name or "").strip()).strip('_')

def save_state(picker_value, folder_tracker, *args):
    settings_dict = dict(zip(SETTINGS_KEYS, args))
    new_name, err = validate_project_name(settings_dict.get("project_name", ""))
    if err:
        return err, gr.update(), gr.update(), gr.update()
    # Persist the sanitized (legal) name, not the raw typed text.
    settings_dict["project_name"] = new_name

    # Determine the project's current on-disk folder name (what to rename FROM).
    # Three independent signals, most reliable first:
    #   1. the project picker's current selection (the loaded project),
    #   2. settings.json (rewritten on every load/new/clone/train),
    #   3. the hidden UI tracker.
    # Match the RAW name exactly as it sits on disk -- real folders may contain
    # spaces/hyphens (e.g. duplicated as "X - Copy"), so we must NOT sanitize
    # before checking existence. Only the NEW name gets legalized.
    if picker_value == NEW_PROJECT_SENTINEL:
        picker_value = ""
    raw_candidates = [picker_value, load_settings().get("project_name", ""), folder_tracker]
    old_name = ""
    for c in raw_candidates:
        c = (c or "").strip()
        if c and c != new_name and (OUTPUT_BASE / c).is_dir():
            old_name = c
            break

    new_dir = OUTPUT_BASE / new_name
    msg = "✅ Settings saved."
    if old_name:
        if new_dir.exists():
            msg = f"⚠️ Cannot rename: '{new_name}' already exists. Settings saved without renaming."
        else:
            (OUTPUT_BASE / old_name).rename(new_dir)
            old_prefix = _detect_old_prefix(new_dir, old_name)
            rename_project_contents(new_dir, old_prefix, new_name)
            msg = f"✅ Renamed '{old_name}' → '{new_name}' and saved."

    save_settings(settings_dict)
    write_project_configs(settings_dict)
    projects = list_output_projects()
    return msg, gr.update(choices=[NEW_PROJECT_SENTINEL] + projects, value=new_name), new_name, gr.update(value=new_name)

def settings_to_values(settings_dict):
    return [settings_dict.get(k, DEFAULT_SETTINGS[k]) for k in SETTINGS_KEYS]

def decorate_auto_displays(values):
    """Patch the scheduler entry in a settings_to_values() list so its Auto label
    reflects the (programmatically) loaded optimizer. The .input() handlers only
    fire on real user interaction, so loads need this."""
    vals = list(values)
    opt = vals[SETTINGS_KEYS.index("optimizer")]
    si = SETTINGS_KEYS.index("lr_scheduler")
    vals[si] = gr.update(choices=scheduler_dropdown_choices(opt), value=vals[si])
    return vals

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
        return settings_to_values(load_settings()) + [[], msg]

    training_files = sorted(configs_dir.glob("*_training.toml"), key=os.path.getmtime, reverse=True)
    dataset_files = sorted(configs_dir.glob("*_dataset.toml"), key=os.path.getmtime, reverse=True)
    prompt_files = sorted(configs_dir.glob("*_prompts.txt"), key=os.path.getmtime, reverse=True)

    if not training_files:
        msg = f"⚠️ No training config found in: {configs_dir}"
        return settings_to_values(load_settings()) + [[], msg]

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
    # Model paths are global/shared (from settings.json), not per-project.
    # Intentionally do NOT load them from the project TOML so switching projects
    # keeps the globally configured model paths.
    loaded["network_rank"] = int(training_cfg.get("network_dim", loaded["network_rank"]))
    loaded["learning_rate"] = str(training_cfg.get("learning_rate", loaded["learning_rate"]))
    # Reverse-map dotted optimizer_type paths (e.g. pytorch_optimizer.CAME) back to friendly dropdown names.
    OPTIMIZER_TYPE_REVERSE_MAP = {"pytorch_optimizer.CAME": "CAME"}
    loaded_opt = training_cfg.get("optimizer_type", loaded["optimizer"])
    loaded["optimizer"] = OPTIMIZER_TYPE_REVERSE_MAP.get(loaded_opt, loaded_opt)
    # Scheduler: if the saved value matches what "Auto" would pick for this optimizer,
    # show it as Auto (so changing the profile default keeps the display in sync).
    loaded_sched = training_cfg.get("lr_scheduler", AUTO_SENTINEL)
    loaded["lr_scheduler"] = AUTO_SENTINEL if loaded_sched == optimizer_default_scheduler(loaded["optimizer"]) else loaded_sched
    raw_warmup = training_cfg.get("lr_warmup_steps", loaded["lr_warmup_steps"])
    if isinstance(raw_warmup, float) and 0 < raw_warmup < 1:
        loaded["lr_warmup_steps"] = int(round(raw_warmup * 100))
    else:
        loaded["lr_warmup_steps"] = int(raw_warmup) if raw_warmup else 0
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
    sample_dir = project_dir / "sample"
    return settings_to_values(loaded) + [get_latest_images(sample_dir), f"✅ Loaded project: {project_dir.name}"]

def refresh_project_choices(current_value):
    projects = list_output_projects()
    choices = [NEW_PROJECT_SENTINEL] + projects
    # Keep the current selection if it still exists; only fall back otherwise.
    value = current_value if current_value in choices else (projects[0] if projects else None)
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

def create_training_toml(project_name, config_save_dir, actual_output_dir, rank, lr, optimizer, max_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc, lr_scheduler="auto", lr_warmup_steps=0):
    config_path = config_save_dir / f"{project_name}_training.toml"
    network_alpha = max(1, int(rank) // 2)

    # Map friendly optimizer names to the actual optimizer_type written to the TOML.
    # sd-scripts imports any dotted "module.Class" path; CAME comes from pytorch_optimizer.
    OPTIMIZER_TYPE_MAP = {"CAME": "pytorch_optimizer.CAME"}
    optimizer_type_value = OPTIMIZER_TYPE_MAP.get(optimizer, optimizer)

    # Scheduler + optimizer args come from the centralized OPTIMIZER_PROFILES.
    # When the UI leaves the scheduler on "Auto", resolve to the profile default.
    opt_args = optimizer_args_for(optimizer)
    scheduler = resolve_scheduler(lr_scheduler, optimizer)
    warmup = parse_warmup(lr_warmup_steps)

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
        "optimizer_type": optimizer_type_value,
        "optimizer_args": opt_args,
        "lr_scheduler": scheduler,
        "lr_warmup_steps": warmup,
        "max_train_steps": int(max_steps),
        "train_batch_size": int(batch_size),
        "gradient_accumulation_steps": int(grad_acc),
        "mixed_precision": HIDDEN_SETTINGS["mixed_precision"],
        "output_dir": actual_output_dir.resolve().as_posix(),
        "output_name": project_name,
        "save_every_n_steps": int(save_steps),
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

def start_training(trigger_word, project_name, dataset_path, dit_p, qwen_p, vae_p, rank, lr, optimizer, lr_scheduler, lr_warmup_steps, t_steps, save_steps, sample_steps, pos, pos2, pos3, pos4, pos5, neg, w, h, s_steps_gen, s_cfg, s_seed, train_seed, batch_size, grad_acc):
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
        full_error += "\n\n⚠️ ERROR: Please check and set the correct model paths in the section:\n'🔧 Paths to Models'"
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

    project_name = re.sub(r'[^a-zA-Z0-9_]', '_', project_name.strip()).strip('_') or "untitled"
    cur = load_settings()
    cur["project_name"] = project_name
    save_settings(cur)

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

    # Show any existing samples immediately so the gallery isn't blank until the
    # first new preview lands.
    yield "\n".join(log_lines), get_latest_images(sample_dir)

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
    training_toml = create_training_toml(project_name, project_configs_dir, project_out_dir, rank, lr, optimizer, t_steps, save_steps, sample_steps, models, prompt_path, train_seed, batch_size, grad_acc, lr_scheduler, lr_warmup_steps)

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

def remember_lr(opt, current_lr, lr_memory):
    """Remember the current LR under the currently selected optimizer (user typing)."""
    lr_memory = dict(lr_memory or {})
    if opt:
        lr_memory[opt] = current_lr
    return lr_memory

def apply_optimizer_lr(opt, current_lr, lr_memory):
    """When the user switches optimizer, restore its remembered LR or fall back to its
    profile default (see OPTIMIZER_PROFILES)."""
    lr_memory = lr_memory or {}
    return lr_memory.get(opt) or optimizer_default_lr(opt)

def apply_optimizer_warmup(opt):
    """When the user switches optimizer, update warmup to the profile's recommended default."""
    return optimizer_default_warmup(opt)


def _unique_project_name(base_name):
    base_clean = re.sub(r'[^a-zA-Z0-9_]', '_', (base_name or "").strip()).strip('_') or "untitled"
    existing = {p.lower() for p in list_output_projects()}
    if base_clean.lower() not in existing:
        return base_clean
    i = 2
    while f"{base_clean}_{i}".lower() in existing:
        i += 1
    return f"{base_clean}_{i}"


def open_project_folder(project_name):
    project_name = (project_name or "").strip()
    if not project_name or project_name == NEW_PROJECT_SENTINEL:
        return "⚠️ No project selected."
    project_dir = OUTPUT_BASE / re.sub(r'[^a-zA-Z0-9_]', '_', project_name).strip('_')
    if not project_dir.exists():
        return f"⚠️ Project folder not found: {project_dir}"
    os.startfile(str(project_dir))
    return f"📂 Opened: {project_dir.name}"

def new_project_action():
    new_name_clean = _unique_project_name("untitled")
    new_settings = DEFAULT_SETTINGS.copy()
    new_settings["project_name"] = new_name_clean
    save_settings(new_settings)
    project_out_dir = OUTPUT_BASE / new_name_clean
    for d in [project_out_dir, project_out_dir / "configs", project_out_dir / "sample"]:
        d.mkdir(parents=True, exist_ok=True)
    choices = [NEW_PROJECT_SENTINEL] + list_output_projects()
    new_vals = settings_to_values(new_settings)
    count = get_prompt_count(new_settings)
    return (
        new_vals
        + [gr.update(choices=choices, value=new_name_clean)]
        + [f"✅ New project created: {new_name_clean}"]
        + [gr.update(value=count)]
        + prompt_visibility_updates(count)
    )

def clone_project_action(*current_input_values):
    current_settings = dict(zip(SETTINGS_KEYS, current_input_values))
    base_name = f"{current_settings.get('project_name', '')}_clone"
    new_name_clean = _unique_project_name(base_name)
    current_settings["project_name"] = new_name_clean
    save_settings(current_settings)
    project_out_dir = OUTPUT_BASE / new_name_clean
    for d in [project_out_dir, project_out_dir / "configs", project_out_dir / "sample"]:
        d.mkdir(parents=True, exist_ok=True)
    choices = [NEW_PROJECT_SENTINEL] + list_output_projects()
    cloned_vals = settings_to_values(current_settings)
    count = get_prompt_count(current_settings)
    return (
        cloned_vals
        + [gr.update(choices=choices, value=new_name_clean)]
        + [f"✅ Cloned to: {new_name_clean}"]
        + [gr.update(value=count)]
        + prompt_visibility_updates(count)
    )



cs = load_settings()

with gr.Blocks(title="Anima TrainFlow") as ui:
    gr.Markdown(
        '# <span title="Original repo creator: ThetaCursed">Anima TrainFlow</span> <a href="https://github.com/MNeMoNiCuZ/Anima-TrainFlow" title="This fork maintained by MNeMoNiCuZ" target="_blank">🔗</a>',
        elem_id="main-header"
    )
    
    # Per-optimizer LR memory, seeded with defaults plus the currently loaded optimizer's saved LR.
    lr_memory_state = gr.State(value={**{o: optimizer_default_lr(o) for o in OPTIMIZER_CHOICES}, cs.get("optimizer", DEFAULT_OPTIMIZER): cs.get("learning_rate", "1.0")})
    prompt_count_state = gr.Number(value=get_prompt_count(cs), visible=False, precision=0)
    # Tracks the actual on-disk folder name of the currently loaded project.
    # Updated by every "project loaded" event. Never touched by the user typing in project_name.
    _folder_tracker = gr.Textbox(visible=False, value=cs.get("project_name", ""))

    cur_opt = cs.get("optimizer", DEFAULT_OPTIMIZER)
    TOOLTIPS["tt-lr"] = lr_tooltip(cur_opt)  # initial LR tooltip for the loaded optimizer
    with gr.Row():
        # ── LEFT: project / actions / logs ────────────────────────────────────
        with gr.Column(scale=1):
            with gr.Group():
                with gr.Row():
                    project_name = gr.Textbox(label="Project Name", value=cs.get("project_name", ""), placeholder="e.g., anima_character_v1", lines=1, max_lines=1, elem_id="tt-project-name")
                    trigger_word = gr.Textbox(label="Trigger Word", value=cs.get("trigger_word", ""), placeholder="e.g., unique_style", lines=1, max_lines=1, elem_id="tt-trigger-word")
                with gr.Row():
                    dataset_path = gr.Textbox(label="Dataset Path (Images + .txt)", value=cs.get("dataset_path", ""), placeholder="C:/Images/MyDataset", lines=1, max_lines=1, elem_id="tt-dataset-path")
                    project_picker = gr.Dropdown(
                        label="Project Picker",
                        choices=[NEW_PROJECT_SENTINEL] + list_output_projects(),
                        value=cs.get("project_name", None),
                        allow_custom_value=True,
                        elem_id="tt-project-picker",
                    )
            with gr.Accordion("🔧 Paths to Models", open=False):
                dit_input = gr.Textbox(label="DiT", value=cs.get("dit_path", ""), lines=1, max_lines=1, elem_id="tt-dit")
                qwen_input = gr.Textbox(label="Qwen3", value=cs.get("qwen_path", ""), lines=1, max_lines=1, elem_id="tt-qwen")
                vae_input = gr.Textbox(label="VAE", value=cs.get("vae_path", ""), lines=1, max_lines=1, elem_id="tt-vae")
            with gr.Group():
                with gr.Row():
                    start_btn = gr.Button("🚀 Start", variant="primary", scale=1, min_width=0, elem_id="btn-start")
                    stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1, min_width=0, elem_id="btn-stop")
                    open_folder_btn = gr.Button("📁 Open", variant="secondary", scale=1, min_width=0, elem_id="btn-open")
                    load_project_btn = gr.Button("🔄 Refresh", variant="secondary", scale=1, min_width=0, elem_id="btn-refresh-picker")
                    save_btn = gr.Button("💾 Save", variant="secondary", scale=1, min_width=0, elem_id="btn-save")
                with gr.Group(visible=False) as new_project_modal:
                    gr.Markdown("**Create Project**")
                    with gr.Row():
                        modal_name_input = gr.Textbox(label="Project Name", placeholder="Leave empty for auto-generated name", scale=3)
                        modal_type_radio = gr.Radio(["New Project", "Clone Current"], value="New Project", label="Type", scale=1)
                    with gr.Row():
                        modal_confirm_btn = gr.Button("✅ Create", variant="primary", scale=1, min_width=0)
                        modal_cancel_btn = gr.Button("❌ Cancel", scale=1, min_width=0)
            output_log = gr.Textbox(label="Logs", lines=LOG_BOX__MAX_LINES, max_lines=LOG_BOX__MAX_LINES, interactive=False, autoscroll=True, elem_id="log-container")
        # ── RIGHT: training hyperparameters + sample gallery ──────────────────
        with gr.Column(scale=1):
            with gr.Group():
                with gr.Row():
                    rank_input = gr.Number(label="Network Rank", value=cs.get("network_rank", 16), precision=0, elem_id="tt-rank")
                    optimizer_input = gr.Dropdown(label="Optimizer", choices=OPTIMIZER_CHOICES, value=cur_opt, elem_id="tt-optimizer")
                    lr_input = gr.Textbox(label="Learning Rate", value=cs.get("learning_rate", "1.0"), elem_id="tt-lr")
                    batch_size_input = gr.Number(label="Batch Size", value=cs.get("train_batch_size", 1), precision=0, elem_id="tt-batch-size")
                    train_seed_val = gr.Number(value=cs.get("train_seed", 42), visible=False)
                with gr.Row():
                    scheduler_input = gr.Dropdown(label="LR Scheduler", choices=scheduler_dropdown_choices(cur_opt), value=cs.get("lr_scheduler", AUTO_SENTINEL), elem_id="tt-scheduler")
                    warmup_input = gr.Number(label="Warmup Steps %", value=cs.get("lr_warmup_steps", optimizer_default_warmup(cur_opt)), precision=0, elem_id="tt-warmup")
                    grad_acc_input = gr.Number(label="Gradient Accum.", value=cs.get("gradient_accumulation_steps", 1), precision=0, elem_id="tt-grad-acc")
                with gr.Row():
                    steps_input = gr.Number(label="Max Training Steps", value=cs.get("training_steps", 2400), precision=0, elem_id="tt-steps")
                    save_steps_input = gr.Number(label="Save x Steps", value=cs.get("save_steps", 300), precision=0, elem_id="tt-save-steps")
                    sample_steps_input = gr.Number(label="Preview x Steps", value=cs.get("sample_steps", 300), precision=0, elem_id="tt-sample-steps")
            with gr.Accordion("🖼️ Sample Gallery", open=True):
                preview_gallery = gr.Gallery(label="Previews", columns=2, rows=2, height=GALLERY_HEIGHT, object_fit="contain", show_label=False)
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
                neg_prompt = gr.Textbox(label="Negative Prompt", lines=1, value=cs.get("neg_prompt", ""), elem_id="tt-neg-prompt")
                with gr.Row():
                    width_input = gr.Number(label="Width", value=cs.get("width", 1024), precision=0, min_width=80, elem_id="tt-width")
                    height_input = gr.Number(label="Height", value=cs.get("height", 1024), precision=0, min_width=80, elem_id="tt-height")
                    sample_steps_gen_input = gr.Number(label="Steps", value=cs.get("sample_steps_gen", 30), precision=0, min_width=80, elem_id="tt-gen-steps")
                    sample_cfg_input = gr.Number(label="CFG", value=cs.get("sample_cfg", 4.0), min_width=80, elem_id="tt-gen-cfg")
                    sample_seed_input = gr.Number(label="Seed", value=cs.get("sample_seed", 42), precision=0, min_width=80, elem_id="tt-gen-seed")

    inputs_list = [
        trigger_word, project_name, dataset_path, dit_input, qwen_input, vae_input,
        rank_input, lr_input, optimizer_input, scheduler_input, warmup_input,
        steps_input, save_steps_input, sample_steps_input,
        pos_prompt, pos_prompt_2, pos_prompt_3, pos_prompt_4, pos_prompt_5, neg_prompt, width_input, height_input,
        sample_steps_gen_input, sample_cfg_input, sample_seed_input, train_seed_val, batch_size_input, grad_acc_input
    ]

    def load_state_on_refresh():
        current_settings = load_settings()
        project_name_val = current_settings.get("project_name", "")
        sample_dir = OUTPUT_BASE / project_name_val / "sample" if project_name_val else Path("nonexistent")
        images = get_latest_images(sample_dir) if sample_dir.exists() else []
        return decorate_auto_displays(settings_to_values(current_settings)) + [images, project_name_val]

    ui.load(fn=load_state_on_refresh, inputs=None, outputs=inputs_list + [preview_gallery, _folder_tracker], js=make_init_tooltips_js())    

    output_log.change(None, None, None, js=JS_SCROLL)
    # .input() fires only on real user interaction (not on programmatic project loads),
    # so loading a project never clobbers its saved LR.
    lr_input.input(fn=remember_lr, inputs=[optimizer_input, lr_input, lr_memory_state], outputs=[lr_memory_state])
    optimizer_input.input(fn=apply_optimizer_lr, inputs=[optimizer_input, lr_input, lr_memory_state], outputs=[lr_input])
    optimizer_input.input(fn=apply_optimizer_warmup, inputs=[optimizer_input], outputs=[warmup_input])

    def relabel_scheduler_auto(opt, current_value):
        choices = scheduler_dropdown_choices(opt)
        valid = {c[1] for c in choices}
        val = current_value if current_value in valid else AUTO_SENTINEL
        return gr.update(choices=choices, value=val)

    # Keep the "Auto (...)" scheduler label in sync with the selected optimizer,
    # reading live from OPTIMIZER_PROFILES.
    optimizer_input.input(fn=relabel_scheduler_auto, inputs=[optimizer_input, scheduler_input], outputs=[scheduler_input])
    # Re-stamp the LR field's tooltip for the new optimizer (fires on user picks
    # and on programmatic project loads).
    optimizer_input.change(fn=None, inputs=[optimizer_input], outputs=None, js=JS_LR_TOOLTIP)

    def maybe_load_project(val):
        if val == NEW_PROJECT_SENTINEL:
            return [gr.update()] * (len(inputs_list) + 3)  # +2 gallery+log, +1 tracker
        result = load_project_config(val)
        # result = settings_to_values + [images, log_msg]
        # Extract the sanitized folder name from the loaded project_name value
        folder_name = result[SETTINGS_KEYS.index("project_name")]
        n = len(SETTINGS_KEYS)
        result = decorate_auto_displays(result[:n]) + result[n:]
        return result + [folder_name]

    def cancel_modal():
        projects = list_output_projects()
        val = projects[0] if projects else None
        return gr.update(visible=False), gr.update(choices=[NEW_PROJECT_SENTINEL] + projects, value=val)

    def modal_create_action(modal_name, modal_type, *current_inputs):
        current_settings = dict(zip(SETTINGS_KEYS, current_inputs))
        if modal_type == "Clone Current":
            base_name = modal_name.strip() or f"{current_settings.get('project_name', '')}_clone"
            new_settings = current_settings.copy()
        else:
            base_name = modal_name.strip() or "untitled"
            new_settings = DEFAULT_SETTINGS.copy()
        new_name_clean = _unique_project_name(base_name)
        new_settings["project_name"] = new_name_clean
        save_settings(new_settings)
        write_project_configs(new_settings)
        project_out_dir = OUTPUT_BASE / new_name_clean
        for d in [project_out_dir, project_out_dir / "configs", project_out_dir / "sample"]:
            d.mkdir(parents=True, exist_ok=True)
        choices = [NEW_PROJECT_SENTINEL] + list_output_projects()
        new_vals = decorate_auto_displays(settings_to_values(new_settings))
        count = get_prompt_count(new_settings)
        msg = f"✅ {'Cloned to' if modal_type == 'Clone Current' else 'New project created'}: {new_name_clean}"
        return (
            new_vals
            + [gr.update(choices=choices, value=new_name_clean)]
            + [msg]
            + [gr.update(value=count)]
            + prompt_visibility_updates(count)
            + [gr.update(visible=False)]
            + [new_name_clean]
        )

    save_btn.click(fn=save_state, inputs=[project_picker, _folder_tracker] + inputs_list, outputs=[output_log, project_picker, _folder_tracker, project_name])
    start_btn.click(fn=start_training, inputs=inputs_list, outputs=[output_log, preview_gallery])
    stop_btn.click(fn=stop_training, outputs=output_log)
    load_project_btn.click(fn=refresh_project_choices, inputs=[project_picker], outputs=[project_picker])
    project_picker.change(fn=maybe_load_project, inputs=[project_picker], outputs=inputs_list + [preview_gallery, output_log, _folder_tracker])
    project_picker.change(
        fn=lambda v: gr.update(visible=(v == NEW_PROJECT_SENTINEL)),
        inputs=[project_picker],
        outputs=[new_project_modal],
    )
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

    modal_cancel_btn.click(fn=cancel_modal, outputs=[new_project_modal, project_picker])
    _modal_outputs = inputs_list + [project_picker, output_log, prompt_count_state, pos_prompt_2, pos_prompt_3, pos_prompt_4, pos_prompt_5, add_prompt_btn, new_project_modal, _folder_tracker]
    modal_confirm_btn.click(fn=modal_create_action, inputs=[modal_name_input, modal_type_radio] + inputs_list, outputs=_modal_outputs)

    open_folder_btn.click(fn=open_project_folder, inputs=[project_name], outputs=output_log)

if __name__ == "__main__":
     ui.launch(inbrowser=True, theme=gr.themes.Soft(), css=CSS)


