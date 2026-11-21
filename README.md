# ComfyUI-Rumble2Flash

Custom nodes for running the **[Reson8/Rumble-2-Flash](https://huggingface.co/Reson8/Rumble-2-Flash)** speech recognition model (`Qwen3-ASR` architecture, ~0.5B parameters, 44 languages) 100% locally within **ComfyUI**, using PyTorch and Hugging Face `transformers`.


## NODES

1. **Rumble-2-Flash Model Loader** — Loads the model from `models/rumble_asr` or automatically downloads files from `https://huggingface.co/Reson8/Rumble-2-Flash/tree/main` if it is not present yet. Includes selectors for precision (`bfloat16`, `float16`, `float32`, `auto`), attention mechanism (`sdpa`, `flash_attention_2`, `eager`), and VRAM management policy.

2. **Rumble-2-Flash Audio / URL Loader** — Loads local files or from HTTP/HTTPS URLs and generates a standard ComfyUI `AUDIO` tensor (16 kHz, mono). You can use any other Load Audio ComfyUI node.

3. **Rumble-2-Flash Transcribe (with Chunking)** — Splits long audio files based on RMS energy-based pause detection (with configurable overlap), removes duplicate words at segment boundaries, and outputs `RUMBLE_SEGMENTS` containing timestamps.

4. **Rumble-2-Flash Subtitle Formatter (SRT/VTT)** — Formats segments into SubRip (`.srt`) or WebVTT (`.vtt`) files, with options for character-per-line limits and automatic saving to the `ComfyUI/output/` directory.

5. **Rumble-2-Flash Batch Folder** — Transcribes entire folders (including recursive search), updates the ComfyUI progress bar, automatically chunks long files, and saves individual outputs along with a `_batch_report.json` file. 


## INSTALLATION

**1)** Unzip into `ComfyUI/custom_nodes/`, you will have `ComfyUI/custom_nodes/ComfyUI-Rumble2Flash`.


I think you probably will have already all the dependencies so you can skip this and open ComfyUI and see if it works, go to step 4 for the workflows.

If it gives error then you need to install the dependencies. Close ComfyUI and do this:


**2)** Install the dependencies:

(Ask ChatGPT or other AI if you don't understand how to do this, say it to guide you one step at the time so it waits for you to give next step)

Open a terminal in /ComfyUI folder, enter the environment, change folder to /custom_nodes/ComfyUI-Rumble2Flash and install the dependencies with:

Linux:
```bash
pip install -r requirements.txt
```
Windows portable:
```bash
python_embeded\python.exe -m pip install -r requirements.txt
```

**3)** Restart ComfyUI and drag any of the workflows from the `examples/` folder into ComfyUI.


**4)** Included workflows

There are 2 workflows in /custom_nodes/ComfyUI-Rumble2Flash/examples/  you can just drag them to ComfyUI window


