#DOESN'T WORK even though passes env checker

from math import gamma
from imports import *
import params
import os

from stable_baselines3 import PPO
#from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.policies import ActorCriticCnnPolicy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed


#Preprocess, skip frames, grayscale, resize etc
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0, high=255, shape=self.observation_space.shape[:2], dtype=np.uint8)

    def observation(self, observation):
        transform = transforms.Grayscale()
        return transform(torch.tensor(np.transpose(observation, (2, 0, 1)).copy(), dtype=torch.float))


class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        self.shape = (shape, shape)
        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transformations = transforms.Compose([transforms.Resize(self.shape), transforms.Normalize(0, 255)])
        #output = (input - mean)/std , so (input-0)/255
        return transformations(observation).squeeze(0).numpy()


from stable_baselines3.common.env_checker import check_env
env = gym.make('BreakoutNoFrameskip-v4')

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
#env = FrameStack(env, num_stack=4)
check_env(env)

def env_layers(env_id):
    env = gym.make(env_id)

    # Apply Wrappers to environment
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
#    env = FrameStack(env, num_stack=4)
    return env

from typing import Callable
def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """
    def _init() -> gym.Env:
        env = env_layers(env_id)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

env_id = 'BreakoutNoFrameskip-v4'
num_cpu = 4  # Number of processes to use
# Create the vectorized environment
env = DummyVecEnv([make_env(env_id, i) for i in range(num_cpu)])

env = VecFrameStack(env, n_stack=4)

#Set seeds
env.seed(params.seed)
env.action_space.seed(params.seed)
torch.manual_seed(params.seed)
torch.random.manual_seed(params.seed)
random.seed(params.seed)
np.random.seed(params.seed)



#Create logging folders if required
models_dir = "models"
logdir = "sb3_tensorboard"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
else:
    for file in os.listdir(models_dir):
        os.remove(os.path.join(models_dir, file))

if not os.path.exists(logdir):
    os.makedirs(logdir)



#Not confirmed, but this should turn off the default normalization in the CNN policy (we are already doing)
class CustomCnnPolicy(ActorCriticCnnPolicy):
    def __init__(self):
        super().__init__(self)
        self.normalize_images = False


model = PPO(CustomCnnPolicy, env, 
            learning_rate = params.learning_rate,
            batch_size = params.batch_size,
            gamma = params.gamma,
            verbose = 1,
            tensorboard_log = logdir)


sub_timesteps = 850000*2
for i in range(1,6): #10000 episodes worked out to ~8.5 mill steps
    model.learn(total_timesteps = sub_timesteps, reset_num_timesteps = False,
                log_interval=10, tb_log_name='PPO')
    
    model.save(f"{models_dir}/{sub_timesteps*i}")