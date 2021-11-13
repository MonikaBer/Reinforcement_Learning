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

# odkomentuj dla instalacji acme (example1.py). W mojej wersji powoduje to popsucie tensorflow przez segmentation fault. 
# używać na własną odpowiedzialność
#pip install dm-acme dm-acme[jax] dm-acme[tensorflow]
#pip install dm-acme[launchpad]
#pip install dm-acme[envs]