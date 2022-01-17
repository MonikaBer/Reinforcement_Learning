import pyvirtualdisplay
import imageio
import base64
import IPython
import numpy as np


def createDisplay(x = 160, y = 210):
	return pyvirtualdisplay.Display(visible = 0, size = (x, y)).start()


def render(env):
    return env.environment.render(mode = 'rgb_array')


def collectFrames(env, actor, steps = 1000):
    frames = []
    timestep = env.reset()

    for _ in range(steps):
        frames.append(render(env))
        action = actor.select_action(timestep.observation)
        timestep = env.step(action)
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
    video_tag = ('<video  width="160" height="210" controls alt="test" '
                'src="data:video/mp4;base64,{0}">').format(b64_video.decode())
    return IPython.display.HTML(video_tag)
