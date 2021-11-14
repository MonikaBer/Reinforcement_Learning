# Reinforcement_Learning

## Documentation
[v1](https://onedrive.live.com/edit.aspx?action=editnew&resid=531E553CA6EE03C2!10911&ithint=file%2cdocx&action=editnew&wdNewAndOpenCt=1635605862040&wdPreviousSession=b34f15e1-7138-4bcd-958e-461fe0fbe4e7&wdOrigin=OFFICECOM-WEB.START.NEW)

[v2](https://onedrive.live.com/edit.aspx?action=editnew&resid=531E553CA6EE03C2!10914&ithint=file%2cdocx&action=editnew&wdNewAndOpenCt=1635607261619&wdPreviousSession=a5ffaf89-1c8c-42b6-a68d-c34ed3e85b84&wdOrigin=OFFICECOM-WEB.START.NEW)

acme: https://github.com/deepmind/acme

aktor - krytyk: https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic

sudo apt install cuda-toolkit-11-5
pip install autorom[accept-rom-license]
pip install dm-env
pip install dm-reverb tensorboard
AutoROM
pip install gym[atari]
pip install tf-agents[reverb]
pip install pyglet
pip install pyvirtualdisplay
pip install imageio

Źródła
https://stackoverflow.com/questions/69442971/error-in-importing-environment-openai-gym
https://superuser.com/questions/247620/how-to-globally-modify-the-default-pythonpath-sys-path

Inne materiały, które mogą pomóc:

dodać plik zawierający ścieżki, w któych python ma szukać skryptów
/usr/local/lib/python2.6/dist-packages/site-packages.pth

W razie problemów z mujoco lub innymi paczkami:
pip list > tmp.txt
pip freeze > tmp.txt
wyszukać tensorflow pyglet tf-agents pyvirtualdisplay dm_* imageio gym dm-env pillow oraz je usunąć pip uninstall
zainstalować na nowo potrzebne paczki

należy pobrać
python:
    imageio_ffmpeg

linux:
    libcudnn8