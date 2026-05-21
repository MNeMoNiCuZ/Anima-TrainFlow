# Anima TrainFlow

## Original Repo
https://github.com/ThetaCursed/Anima-TrainFlow

## Changes

- Added a root `requirements.txt` for app dependencies.
- Added `setup.py` for non-embedded environment setup.
- Setup now auto-downloads `sd-scripts` if missing.
- Setup now installs dependencies with `uv pip`.
- Setup now installs and links `sd-scripts` automatically.
- Added `launch.py` for interpreter-based app launch.
- Added `launch.bat` for venv-based app launch.
- Updated `Install_Requirements.bat` to run `py setup.py`.
- Split `Project Name` and `Trigger Word` into separate GUI inputs.
- `Project Name` now controls output/config naming.
- Added `Project Picker` dropdown listing projects in `training/output`.
- Added `Load Project` button to load project configs into the GUI.
- Project loading supports selecting either a project folder or its `configs` folder.
- Removed `Checkpoint Folder` button.
- Removed `Created by ThetaCursed` header text.
- Renamed `Training Steps` to `Max Training Steps`.
- Added multi-prompt support with a `+` button in the prompt section.
- Added prompt slots `Prompt 1` to `Prompt 5`.
- Multiple prompts are now saved/loaded with settings.
- Prompt file generation now supports multiple prompt lines.
- Training subprocess now uses current interpreter instead of embedded Python path.
- Added clearer launch pre-check errors for missing files/modules.
- Improved log behavior to update tqdm/progress lines in place (less spam).


## Original Readme
Anima TrainFlow is a streamlined, one-page GUI for training LoRA on the **Anima 2B** model. Optimized to run on hardware with as little as **6GB of VRAM**, it eliminates technical overhead by focusing on the essential settings that impact training results the most.

![Anima TrainFlow Interface Preview](preview.png)

## Quick Start (Portable)
1. [**Download Portable Version (1.7GB)**](https://github.com/ThetaCursed/Anima-TrainFlow/releases/download/v1.0.1/Anima-TrainFlow-v1.0.1-Portable.7z)
2. **Extract the archive using [7-Zip](https://7-zip.org/) or WinRAR**.
3. Run `start_trainer.bat`.
4. Open the `🔧 Paths to Models <- Set Once` accordion and specify the paths to your model files.
5. Specify your **Dataset Path** (images + .txt files) and **Trigger Word**, then click **Start**.

## Manual Installation
If you prefer to set up the environment manually instead of using the portable version, follow these steps:
1. **Clone the repository:**
   ```bash
   git clone https://github.com/ThetaCursed/Anima-TrainFlow
   cd Anima-TrainFlow
   ```
2. **Install dependencies:** `Install_Requirements.bat`
3. **Launch the Trainer:** `start_trainer.bat`

## Key Features
* **Zero-Tab Interface:** All critical parameters (Trigger Word, Rank, LR, Steps) are accessible on a single screen.
* **Live Training Previews** Watch your LoRA improve in real-time. The built-in gallery automatically updates whenever a new sample is generated.
* **Smart Dataset Analyzer:** Automatically calculates optimal base resolution and bucket sizes.
* **Portable Edition:** Includes an embedded Python environment to avoid installation or complex setup.
* **Low VRAM Friendly:** Specifically tuned for 6GB+ NVIDIA GPUs.
* **Optimized Defaults:** Pre-configured for BF16 precision and latent caching to ensure maximum performance and stability.
* **Prodigy Native:** Intelligent Learning Rate handling and optimized defaults for the Prodigy optimizer.

## Dataset Preparation
Place all your training images (.png, .jpg, .webp) in a single folder. Every image must have a matching text file with the same name containing its tags/captions (e.g., `image1.png` and `image1.txt`).

## System Requirements
* **OS:** Windows 10/11.
* **GPU:** NVIDIA GPU (6GB+ VRAM recommended for Anima 2B training).
* **Storage:** ~5GB of free space (SSD recommended).

## Technical Details
* **Core:** Based on a modified version of `sd-scripts` for Anima 2B architecture.
* **UI:** Built with Gradio featuring a customized dark theme.
* **Backend:** Utilizes `accelerate launch` for optimized execution.
* **Auto-Save:** All paths and configurations are automatically saved to `settings.json`.
