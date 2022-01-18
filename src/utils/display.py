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


def collectFramesDQN(env, actor, fname, steps = 1000, saveCsv = False):
    frames = []
    timestep = env.reset()

    dframe = pd.DataFrame(columns = ['akcja', 'nagroda', 'reset'])

    for i in range(numSteps):
        frames.append(env.environment.render(mode = 'rgb_array'))
        action = agent.select_action(timestep.observation)

        timestep = env.step(action)
        print(timestep.observation)
        if(timestep.reward is None):
            dframe = dfapend(dframe, action=action, timestep=timestep, idx=i, envReset=True)
            timestep = env.reset()
        else:
            dframe = dfapend(dframe, action=action, timestep=timestep, idx=i, envReset=False)

    if saveCsv:
        dframe.to_csv(fname, sep = ';', index=False)
    return np.array(frames)


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
