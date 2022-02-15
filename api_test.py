from imports import *
import params
import os,sys
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from torchsummary import summary
from agent import DDQNAgent

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

#Load checkpoint
episode = params.episode
max_episodes = params.max_episodes


#Log rewards and network weights
rewards_file = params.rewards_file
save_directory = params.save_directory
if not os.path.exists (os.getcwd() + '/' + save_directory):
    os.mkdir(os.getcwd() + '/' + save_directory)


#Initialize agent
agent = DDQNAgent(action_dim=env.action_space.n, obs_dim = env.observation_space.shape,
                  save_directory=save_directory, rewards_file=rewards_file)

if params.load_checkpoint is not None: #Start from checkpoint?
    agent.load_checkpoint(save_directory + "/" + params.load_checkpoint) #Load weights
    memory_file = save_directory + "/memory_%d.pkl" %episode #Load experience replay deque
    agent.memory = pickle.load( open( memory_file, "rb" ) )
#    agent.current_step = episode * env.Nvacs #Load number of steps


