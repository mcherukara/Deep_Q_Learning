from math import gamma
from imports import *
import params
import os

from stable_baselines3 import DQN

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
        return transformations(observation).squeeze(0)


env = gym.make('BreakoutNoFrameskip-v4')

# Apply Wrappers to environment
env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env)
env = ResizeObservation(env, shape=84)
env = FrameStack(env, num_stack=4)

env.seed(params.seed)
env.action_space.seed(params.seed)
torch.manual_seed(params.seed)
torch.random.manual_seed(params.seed)
random.seed(params.seed)
np.random.seed(params.seed)


models_dir = "models"
logdir = "sb3_tensorboard"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
else:
    for file in os.listdir(models_dir):
        os.remove(os.path.join(models_dir, file))

if not os.path.exists(logdir):
    os.makedirs(logdir)

from stable_baselines3.dqn import CnnPolicy

#Not confirmed, but this should turn off the default normalization in the CNN policy (we are already doing)
class CustomCnnPolicy(CnnPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_images = False

model = DQN(CustomCnnPolicy, env, 
            learning_rate = params.learning_rate,
            buffer_size = params.memory_size,
            learning_starts = params.batch_size*10,
            gamma = params.gamma,
            train_freq = params.learn_every,
            target_update_interval = 2500*params.learn_every,
            exploration_initial_eps = params.exploration_rate_start,
            exploration_final_eps = params.exploration_rate_min,
            # Note SB3 does a linear decay, our code does an exponential decay
            verbose = 1,
            exploration_fraction = 0.5, #How much of training run to decay exploration over
            #This actually works if need > 1 because of how we are breaking up the run
            tensorboard_log = logdir)


sub_timesteps = 850000*2
for i in range(1,6): #10000 episodes worked out to ~8.5 mill steps
    model.learn(total_timesteps = sub_timesteps, reset_num_timesteps = False,
                log_interval=10, tb_log_name='DQN')
    
    model.save(f"{models_dir}/{sub_timesteps*i}")