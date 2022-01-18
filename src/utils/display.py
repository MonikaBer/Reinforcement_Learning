import pyvirtualdisplay
import imageio
import base64
import IPython
import numpy as np
import pandas as pd


def createDisplay(x = 160, y = 210):
	return pyvirtualdisplay.Display(visible = 0, size = (x, y)).start()


def render(env):
    return env.environment.render(mode = 'rgb_array')


def collectFrames(env, actor, fname, steps = 1000, saveCsv = False):
    frames = []
    timestep = env.reset()

    dframe = pd.DataFrame(columns = ['akcja', 'nagroda'])

    for i in range(steps):
        frames.append(render(env))
        action = actor.select_action(timestep.observation)

        newdframe = pd.DataFrame({
            'akcja': str(action),
            'nagroda': str(timestep.reward)
        }, index = [1])
        dframe = dframe.append(newdframe)

        timestep = env.step(action)
        if(timestep.observation.reward is None):
            timestep = env.reset()

    if saveCsv:
        dframe.to_csv(fname, sep = ';')
    return frames


def saveVideo(frames, filename = 'temp.mp4'):
    if (not isinstance(frames, np.ndarray)):
        frames = np.array(frames)
    """Save and display video."""
    # Write video
    with imageio.get_writer(filename, fps = 60) as video:
        for frame in frames:
            video.append_data(frame)
    # Read video and display the video
    video = open(filename, 'rb').read()
    b64_video = base64.b64encode(video)
    video_tag = ('<video  width="160" height="224" controls alt="test" '
                'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
    return IPython.display.HTML(video_tag)
