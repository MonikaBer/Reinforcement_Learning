#!/bin/bash

# uwaga, nie należy wykonywać pip install <pakiet> --upgrade, ponieważ nowsze wersje mogą okazać się niekompatybilne.

sudo apt install cuda-toolkit-11-5 libcudnn8
pip install dm-env dm-reverb tensorboard
pip install gym[atari]
pip install tf-agents[reverb]
pip install pyglet
pip install pyvirtualdisplay
pip install imageio
pip install autorom[accept-rom-license]
pip install imageio_ffmpeg
AutoROM