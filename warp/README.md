install python dependencies and run cloth sim creation

```bash
cd warp
uv venv && source .venv/bin/activate
uv pip install warp-lang[extras]
# for arm agx orin cuda11
uv pip install https://github.com/NVIDIA/warp/releases/download/v1.7.0/warp_lang-1.7.0+cu11-py3-none-manylinux2014_aarch64.whl
uv run python warp_cloth.py
uv run python warp_ik.py
```

use usd viewer to view the cloth sim
download binaries from https://developer.nvidia.com/usd?sortBy=developer_learning_library%2Fsort%2Ffeatured_in.usd_resources%3Adesc%2Ctitle%3Aasc#section-getting-started

```bash
sudo apt-get install libxkbcommon-x11-0 libxcb-xinerama0 libxcb-image0 libxcb-shape0 libxcb-render-util0 libxcb-icccm4 libxcb-keysyms1
unzip ~/Downloads/usd.py310.linux-x86_64.usdview.release-0.25.02-ba8aaf1f.zip -d ~/dev/usd
/home/oop/dev/usd/scripts/usdview_gui.sh /home/oop/dev/cu/warp/example_cloth.usd
/home/oop/dev/usd/scripts/usdview_gui.sh /home/oop/dev/cu/warp/example_jacobian_ik.usd
```

Scroll in with mousewheel
Use <space> to play and pause
Use <alt>+<left click> to rotate the camera
Use <ctrl>+<6> for geometry view