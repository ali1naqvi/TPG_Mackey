import numpy as np
import pandas as pd
from tpg.trainer import Trainer, loadTrainer
from tpg.agent import Agent
import multiprocessing as mp
import time
import ast, pickle
import zlib
from pathlib import Path
from tpg.utils import getLearners, getTeams, learnerInstructionStats, actionInstructionStats, pathDepths
#import ppmd

#values we can modify
MAX_STEPS_G = 1000 #max values we want for training starts with 1 (so subtract one)
GENERATIONS = 972
EXTRA_TIME_STEPS  = 300 #number of wanted generated values 
STARTING_STEP = 0 #starting step
DATA_DIVISION = 0.5


PRIME_STEPS = 50
TRAINING_STEPS = 50
VALIDATION_STEPS = 100


# Read the text file into a pandas DataFrame
data = pd.read_csv("mackey_glass_1100_samples.txt", header=None)
data = data.squeeze()

# full_song_bytes = training_data.to_numpy().ravel().tobytes()
# c_full_song = len(zlib.compress(full_song_bytes))

#reward function, using Normalized Compression Distance


def compress_ppmz(data, level=6, mem_size=16):

    mem_size_bytes = mem_size * 1024 * 1024
    with ppmd.PpmdBufferEncoder(level, mem_size_bytes) as encoder:
        result = encoder.encode(data)
        result += encoder.flush()
    return result

def ncd(sample, target):
    # Normalize the sample
    sample = np.clip(sample, 0, 1)
    
    # Convert sample and target to bytes
    sample_bytes = sample.tobytes()
    target_bytes = target.tobytes()
    
    # Compress individual sequences and concatenated sequence
    #c_sample = len(compress_ppmz(sample_bytes))
    #c_target = len(compress_ppmz(target_bytes))
    c_sample = len(zlib.compress(sample_bytes))
    c_target = len(zlib.compress(target_bytes))
    
    #c_concatenated = len(compress_ppmz(sample_bytes + target_bytes))
    c_concatenated = len(zlib.compress(sample_bytes + target_bytes))
    
    # Calculate NCD value
    ncd_value = (c_concatenated - min(c_sample, c_target)) / max(c_sample, c_target)
    
    return ncd_value

def mse(sample, target):
    sample = np.clip(sample, 0, 1)
    sum_squared_error = 0
    for a, p in zip(target, sample):
        sum_squared_error += (a - p) ** 2
    mse = (sum_squared_error / len(sample))

#environment
class TimeSeriesEnvironment:
    def __init__(self, max_steps = MAX_STEPS_G, current_step_g=STARTING_STEP):

        self.max_generated_steps = max_steps
        self.current_step = current_step_g
        self.total_states = []
        self.total_true_states = []
        
        self.last_state = self._get_state()

    def reset(self, episodenum, window_size):
        # Resets to the starting step we want
        self.total_states = []
        self.total_true_states = []
        self.current_step = episodenum * window_size 

        self.max_generated_steps = (self.current_step + window_size)
        self.last_state = self._get_state() 

        return self.last_state

    def step(self, action_type, reward_func):

        #reset values 
        done = False
        reward = 0
        action_id, action_value = action_type
        
        if self.current_step+1 <= self.max_generated_steps:
            #max steps arent reached so we keep adding each state to a string
            self.total_true_states.extend(self._get_state()) 
            self.total_states.extend(action_value)  # Append the new state as a tuple
        else:
            #end of window for episode
            #calculate at the end of the episode with all values
            self.total_states = np.array(self.total_states, dtype=np.float64)
            self.total_true_states = np.array(self.total_true_states, dtype=np.float64)
            if reward_func == 'ncd':
                reward = -ncd(self.total_states.ravel(), self.total_true_states.ravel()) * 100
            elif reward_func == 'mse':
                reward = -mse(self.total_states.ravel(), self.total_true_states.ravel())
            #print("states: ", self.total_states, "reward: ", reward)
            done = True

        self.current_step +=1
        return self.last_state, action_value, reward, done
    
    def step_simulation(self, action_type):
        # stuff print(action_type)
        _, action_value = action_type
        self.current_step +=1
        return action_value
    # Access the row corresponding to the current step
    
    def _get_state(self):
        return [data[self.current_step]]
        #state = data[self.current_step]
        #return state


def runAgent(args):
    agent, scoreList = args

    #funcion for handling parallelism
    agent.configFunctionsSelf()
    env = TimeSeriesEnvironment()
    scoreTotal = 0
    episode_length = PRIME_STEPS + TRAINING_STEPS # 50+50=100

    numEpisodes = int(MAX_STEPS_G / episode_length) # 800/100=8
    reward = 0

    for ep in range(numEpisodes):
        isDone = False
        action_state = env.reset(ep, episode_length) #resets at next 100 window (based on episode)
        
        predicted_state = action_state  #recursion will only occur for an episode with the correct one starting
        scoreEp = 0
        while True:
            #prime first half of the episode
            #if current step is less than the second half (0-49)
            if  env.current_step < ((ep * episode_length) + PRIME_STEPS):
                action_value = (agent.act(action_state))
                env.current_step += 1 #increase step in environment
                action_state = env._get_state()
                predicted_state = action_state #updating predicted
            else:
            #second half of episode
                action_value = (agent.act(predicted_state))
                #stuff  print("step now: ", env.current_step, "with step: ", predicted_state)
                action_state, predicted_state, reward, isDone = env.step(action_value, reward_func='mse') #now fix step 
                scoreEp += reward    
                
            if isDone:
                # stuff print("we finished")
                break
        scoreTotal += scoreEp
        #print("ScoreTotal:", scoreTotal)

    scoreTotal /= numEpisodes
    agent.reward(scoreTotal, task='main')

    scoreList.append((agent.team.id, agent.team.outcomes))
    return agent, scoreList

def RunValidationAgents(args):
    agent, validscoreList = args

    #funcion for handling parallelism
    agent.configFunctionsSelf()

    env = TimeSeriesEnvironment()
    scoreTotal = 0
    episode_length = PRIME_STEPS + VALIDATION_STEPS #50 + 100 = 150
    numEpisodes = int((MAX_STEPS_G - PRIME_STEPS) / episode_length) # 750 / 150 = 5
    reward = 0

    for ep in range(numEpisodes):
        #memory array is returned as this is the action state
        action_state = env.reset(ep, episode_length) #resets at next 25 window (based on episode)
        #predicted_state = action_state  #recursion will only occur for an episode with the correct one starting
        scoreEp = 0
        isDone = False
        while True:
            #action state is current value, memory based on this will return from get_action_mem (includes it)
            # change the action_state returning. Either use prediction (recursion) or direct approach 
            #prime first half of the episode
            #intuition: multiplication of episode and length gives us the starting step + priming steps gives us where the forecasting should start
            #0 * 150 = 0
            #... 
            #4 * 150 = 600 + 50 = 650 
            if ((ep * episode_length) + PRIME_STEPS) > env.current_step:
                action_value = (agent.act(action_state)) #can ignore output since its priming
                env.current_step +=1 #increase step in environment
                action_state = env._get_state() #recursively updating observation state
            #second half of episode: 100 steps forecasting
            else:
                predicted_state = (agent.act(action_state))
                action_state, predicted_state, reward, isDone = env.step(predicted_state,reward_func='ncd') #now fix step
                scoreEp += reward

            if isDone:
                break
        scoreTotal += scoreEp

    scoreTotal /= numEpisodes
    agent.reward(scoreTotal, task='validation')

    validscoreList.append((agent.team.id, agent.team.outcomes))
    return agent, validscoreList

def RunBestAgent(args):
    agent, scoreList = args

    simulation_results = []
    env = TimeSeriesEnvironment()
    
    # stuff print("Priming")
    # 0 to 799 (inclusive)
    for x in range(MAX_STEPS_G):
        last_state = [data[x]]
        action_value = (agent.act(last_state))
        print("step "+str(x)+" for state: ", last_state)
    
    
    print("Priming complete")
    for x in range(EXTRA_TIME_STEPS-1): 
        print("FORECASTING: step "+str(x)+" for state: ", last_state)
        action_value = (agent.act(last_state))
        #print("action value:", action_value)
        action_state = env.step_simulation(action_value)
        simulation_results.append(action_state)
        last_state = action_state
    print("Simulation complete..")

    simulated_data = pd.DataFrame(simulation_results)

    simulated_data = pd.concat([data[:MAX_STEPS_G+1], simulated_data], ignore_index=True)
    simulated_data.to_csv('Simulation_2.txt', index=False)

if __name__ == '__main__':
    tStart = time.time()
    trainer_checkpoint_path = Path("trainer_savepoint_2.pkl")
    gen_checkpoint_path = Path("gen_savepoint_2.txt")

    if trainer_checkpoint_path.exists():
        trainer = loadTrainer(trainer_checkpoint_path)
        print("LOADED TRAINER")
    else:
        trainer = Trainer(actions=[1], teamPopSize=150, initMaxTeamSize=10, initMaxProgSize=100, pActAtom=0.95, memType="default", operationSet="def")
        gen_start = 0
    
    if gen_checkpoint_path.exists():
        with open(gen_checkpoint_path, 'r') as file:
            gen_start = int(file.read().strip())  # Read the number and convert it to an integer
        print("LOADED GEN NUMBER: ", gen_start)
    else:
        gen_start = 0

    with open('results_2.txt', 'a' if gen_start > 0 else 'w') as file:
        file.write(f"Trainer started: {trainer}\n")
        processes = mp.cpu_count()

        man = mp.Manager() 
        pool = mp.Pool(processes=processes)

        allScores = []

        try:
            for gen in range(gen_start, GENERATIONS): 
                scoreList = man.list()
                
                agents = trainer.getAgents()
                pool.map(runAgent, [(agent, scoreList) for agent in agents])
                
                teams = trainer.applyScores(scoreList)  
                
                champ = trainer.getEliteAgent(task='main')
                champ.saveToFile("best_agent_2")

                trainer.evolve(tasks=['main'])
                
                validation_champion_path = Path("validation_champion_2")
                if gen % 10 == 0 and gen != 0:  # Validation phase every 10 generations
                    prevbestscore = float('-inf')  # Starting value of negative infinity
                    looper = True
                    start_validation_time = time.time()
                    print("Values")
                    while looper:
                        validationScores = man.list()
                        agents = trainer.getAgents()
                        pool.map(RunValidationAgents, [(agent, validationScores) for agent in agents])
                        teams1 = trainer.applyScores(validationScores)
                        
                        current_best_validation = trainer.getEliteAgent(task='validation')
                        print("Validation Generation Score: ", current_best_validation.team.outcomes['validation'])
                        
                        if current_best_validation.team.outcomes['validation'] >= prevbestscore:
                            prevbestscore = current_best_validation.team.outcomes['validation']
                            validationChamp = current_best_validation
                            validationChamp.saveToFile("validation_champion_2")
                        else: 
                            if validation_champion_path.exists():
                                validationChamp = pickle.load(open(validation_champion_path, 'rb'))
                                validationChamp.configFunctionsSelf()
                            else: 
                                validationChamp = trainer.getEliteAgent(task='validation')
                            print("Validation champion: ", validationChamp.team.outcomes)
                            print(f"Validation champ with the best test score: {validationChamp.team.outcomes['validation']} on test data.")
                            with open("final_validation_scores_2.txt", 'w') as f:
                                f.write(str(validationChamp.team.outcomes['validation']))
                            looper = False

                        if time.time() - start_validation_time > (3600 * 4):  # Check if 4 hours have passed
                            print("Time limit for finding a better validation champ exceeded.")
                            looper = False

                        if looper:
                            trainer.evolve(tasks=['validation'])

                scoreStats = trainer.fitnessStats
                allScores.append((scoreStats['min'], scoreStats['max'], scoreStats['average']))
                print(f"Gen: {gen}, Best Score: {scoreStats['max']}, Avg Score: {scoreStats['average']}, Time: {str((time.time() - tStart)/3600)}")
                file.write(f"Gen: {gen}, Best Score: {scoreStats['max']}, Avg Score: {scoreStats['average']}, Time: {str((time.time() - tStart)/3600)}\n")
                file.flush()  # Ensure data is written to disk
                
                trainer.saveToFile("trainer_savepoint_2.pkl")
                with open("gen_savepoint_2.txt", 'w') as gen_file:
                    gen_file.write(str(gen))
                
            file.write(f'Time Taken (Hours): {(time.time() - tStart)/3600}\n')
            file.write('Final Results:\nMin, Max, Avg\n')
            for score in allScores:
                file.write(f"{score}\n")
            file.flush()
        except Exception as e:
            file.write(f"Error occurred: {str(e)}\n")
            file.flush()

        champ = pickle.load(open("best_agent_2", 'rb'))
        champ.configFunctionsSelf()
        print(champ.team)
        print(champ.team.fitness)
        print(champ.team.learners)
        print(champ.team.outcomes)
        print("---------------")
        #champ.configFunctions()

        # Assuming RunBestAgent is a function you have defined earlier
        #empty array is: scorelist
        RunBestAgent((champ, []))
