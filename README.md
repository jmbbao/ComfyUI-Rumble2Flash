# ComfyUI-Rumble2Flash

Custom nodes for running the **[Reson8/Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash)** speech recognition model (`Qwen3-ASR` architecture, ~0.5B parameters, 44 languages) 100% locally within **ComfyUI**, using PyTorch and Hugging Face `transformers`.


## NODES

1. **Rumble-2-Flash Model Loader** — Loads the model from `models/rumble_asr` or automatically downloads files from `https://huggingface.co/Reson8/Rumble-2-Flash/tree/main` if it is not present yet. Includes selectors for precision (`bfloat16`, `float16`, `float32`, `auto`), attention mechanism (`sdpa`, `flash_attention_2`, `eager`), and VRAM management policy.

2. **Rumble-2-Flash Audio / URL Loader** — Loads local files or from HTTP/HTTPS URLs and generates a standard ComfyUI `AUDIO` tensor (16 kHz, mono). You can use any other Load Audio ComfyUI node.

3. **Rumble-2-Flash Transcribe (with Chunking)** — Splits long audio files based on RMS energy-based pause detection (with configurable overlap), removes duplicate words at segment boundaries, and outputs `RUMBLE_SEGMENTS` containing timestamps.

4. **Rumble-2-Flash Subtitle Formatter (SRT/VTT)** — Formats segments into SubRip (`.srt`) or WebVTT (`.vtt`) files, with options for character-per-line limits and automatic saving to the `ComfyUI/output/` directory.

5. **Rumble-2-Flash Batch Folder** — Transcribes entire folders (including recursive search), updates the ComfyUI progress bar, automatically chunks long files, and saves individual outputs along with a `_batch_report.json` file. 


## INSTALLATION

**1)** Click on the green "Code" button on the top of this page and choose "Download zip". Unzip the content into `ComfyUI/custom_nodes/`, and rename the folder removing the "-main" at the end of the name, you will have now: `ComfyUI/custom_nodes/ComfyUI-Rumble2Flash`.

Or you can open a terminal in /ComfyUI/custom_nodes and use:
```bash
git clone https://github.com/jmbbao/ComfyUI-Rumble2Flash.git
```
You will probably have already all the dependencies so you can go to step 4 for the workflows and just try it in ComfyUI.
If it gives error then you need to install the dependencies:


**2)** Install the dependencies:

Close ComfyUI, open a terminal inside of /ComfyUI folder. Now enter the environment. For example in Linux with venv: (ask ChatGPT or any other AI to give you the exact steps for your system, give to it this whole text)
```bash
cd ComfyUI
source venv/bin/activate
```
Now the terminal changed showing you are inside the environment, so the pip order will place the packs in the environment folder instead of in the system of the computer. So we enter the node folder and install the dependencies:
```bash
cd custom_nodes/ComfyUI-Rumble2Flash
pip install -r requirements.txt
```
(ask ChatGPT or any other AI to give you the exact steps for windows or any other variation different of Linux with venv)


**3)** Restart ComfyUI and drag any of the workflows from the `examples/` folder into ComfyUI.


**4)** Included workflows

There are 2 workflows in /custom_nodes/ComfyUI-Rumble2Flash/examples/  you can just drag them to ComfyUI window


