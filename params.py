#Checkpoint params
checkpoint_period = 1000 #How often to save model
log_period = 10 #How often to output model performance
episode = 0
max_episodes = 10000
if episode:
    load_checkpoint = "checkpoint_%d.pth" %episode
else:
    load_checkpoint = None

#Log rewards and network weights
rewards_file = 'rewards2.txt'
save_directory = "Double_DQN/v2"
save_memory = False #Save memory on checkpoint yes/no

#DQNAgent params
exploration_rate_decay = 0.999995
exploration_rate_min = 0.01
learn_every = 4 #How many env steps to train

#Misc
seed = 3 #Random seed for reproduce

#Which model
model_type = "Double_DQN" #Duel_Double_DQN or Double_DQN

#MD specific
#Nvacs = 40 #How many vacancies in a step
#Baseline_E = -5.029576 #Average E of random arrangements of 40 vacs
#E_conversion = 4.3363*10**-2 #kcal/mol to eV for ReaxFF runs
#cores = 32 #How many cores to use?
