from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from gym.wrappers.monitoring import video_recorder

env = gym_super_mario_bros.make('SuperMarioBros-v0')
vid = video_recorder.VideoRecorder(env, path="./recordings/vid.mp4")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

done = True
for step in range(5000):
    if done:
        state = env.reset()
    state, reward, done, info = env.step(env.action_space.sample())
    env.render()
    vid.capture_frame()

env.close()
vid.close()