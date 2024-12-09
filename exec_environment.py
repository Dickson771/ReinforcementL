"""
    This code communicates with the coppeliaSim software and simulates shaking a container to mix objects of different color 

    Install dependencies:
    https://www.coppeliarobotics.com/helpFiles/en/zmqRemoteApiOverview.htm
    
    MacOS: coppeliaSim.app/Contents/MacOS/coppeliaSim -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
    Ubuntu: ./coppeliaSim.sh -GzmqRemoteApi.rpcPort=23004 ~/path/to/file/mix_Intro_to_AI.ttt
"""

import numpy as np
from zmqremoteapi import RemoteAPIClient
import time
import random
import os
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import numpy as np

class Simulation:
    def __init__(self, sim_port=23004):
        self.sim_port = sim_port
        self.directions = ['Up', 'Down', 'Left', 'Right']
        self.initializeSim()

    def initializeSim(self):
        self.client = RemoteAPIClient('localhost', port=self.sim_port)
        self.client.setStepping(True)
        self.sim = self.client.getObject('sim')
        self.defaultIdleFps = self.sim.getInt32Param(self.sim.intparam_idle_fps)
        self.sim.setInt32Param(self.sim.intparam_idle_fps, 0)
        self.getObjectHandles()
        self.sim.startSimulation()
        self.dropObjects()
        self.getObjectsInBoxHandles()

    def getObjectHandles(self):
        self.tableHandle = self.sim.getObject('/Table')
        self.boxHandle = self.sim.getObject('/Table/Box')

    def dropObjects(self):
        self.blocks = 18
        frictionCube = 0.06
        frictionCup = 0.8
        blockLength = 0.016
        massOfBlock = 14.375e-03

        self.scriptHandle = self.sim.getScript(self.sim.scripttype_childscript, self.tableHandle)
        self.client.step()
        self.sim.callScriptFunction('setNumberOfBlocks', self.scriptHandle,
                                    [self.blocks], [massOfBlock, blockLength, frictionCube, frictionCup], ['cylinder'])
        print('Wait until blocks finish dropping')
        while True:
            self.client.step()
            signalValue = self.sim.getFloatSignal('toPython')
            if signalValue == 99:
                loop = 20
                while loop > 0:
                    self.client.step()
                    loop -= 1
                break

    def getObjectsInBoxHandles(self):
        self.object_shapes_handles = []
        self.obj_type = "Cylinder"
        for obj_idx in range(self.blocks):
            obj_handle = self.sim.getObjectHandle(f'{self.obj_type}{obj_idx}')
            self.object_shapes_handles.append(obj_handle)

    def getObjectsPositions(self):
        pos_step = []
        box_position = self.sim.getObjectPosition(self.boxHandle, self.sim.handle_world)
        for obj_handle in self.object_shapes_handles:
            obj_position = self.sim.getObjectPosition(obj_handle, self.sim.handle_world)
            obj_position = np.array(obj_position) - np.array(box_position)
            pos_step.append(list(obj_position[:2]))
        return pos_step

    def getState(self):
        object_positions = self.getObjectsPositions()
        state_x = np.var([pos[0] for pos in object_positions])
        state_y = np.var([pos[1] for pos in object_positions])
        return (state_x, state_y)

    def mixedState(self, state):
        state_x, state_y = state
        mixing_threshold = 0.01
        return state_x < mixing_threshold and state_y < mixing_threshold

    def action(self, direction=None):
        if direction not in self.directions:
            print(f'Direction: {direction} invalid, please choose one from {self.directions}')
            return

        print(f'Action: {direction}')

        box_position = self.sim.getObjectPosition(self.boxHandle, self.sim.handle_world)
        _box_position = box_position
        span = 0.02
        steps = 5
        if direction == 'Up':
            idx = 1
            dirs = [1, -1]
        elif direction == 'Down':
            idx = 1
            dirs = [-1, 1]
        elif direction == 'Right':
            idx = 0
            dirs = [1, -1]
        elif direction == 'Left':
            idx = 0
            dirs = [-1, 1]

        for _dir in dirs:
            for _ in range(steps):
                _box_position[idx] += _dir * span / steps
                self.sim.setObjectPosition(self.boxHandle, self.sim.handle_world, _box_position)
                self.stepSim()

    def stepSim(self):
        self.client.step()

    def stopSim(self):
        self.sim.stopSimulation()

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Sets up our neural network for Q-learning - nothing fancy, just a straightforward
        feedforward network that'll help us make decisions.
        """
        super(QNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),  # First layer - let's make it beefy
            nn.ReLU(),  # ReLU works well enough, no need to overthink it
            nn.Linear(128, 64),  # Narrow it down a bit
            nn.ReLU(),
            nn.Linear(64, output_size)  # Final layer for our action choices
        )

    def predict(self, x):  # renamed from forward
        """
        Runs our input through the network. Simple as that.
        """
        return self.network(x)

# Memory bank for our agent's experiences
class ExperienceReplayBuffer:
    def __init__(self, capacity=50000):
        """
        Sets up our memory buffer - we'll store 50k experiences by default,
        which should be plenty for this task.
        """
        self.buffer = deque(maxlen=capacity)

    def remember(self, state, action, reward, next_state, done):  # renamed from store_experience
        """
        Stores a memory in our buffer. Just like your brain remembering what happened
        after making a decision!
        """
        self.buffer.append((state, action, reward, next_state, done))

    def recall(self, batch_size):  # renamed from sample_batch
        """
        Grabs a random batch of memories to learn from. This is crucial for stable learning -
        otherwise we'd be too fixated on recent experiences.
        """
        states, actions, rewards, next_states, dones = zip(*random.sample(self.buffer, batch_size))
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        """
        Just tells us how many memories we've stored.
        """
        return len(self.buffer)

# Our main learning agent
class DQLAgent:
    def __init__(self, env, learning_rate=0.0005, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.997, batch_size=64):
        """
        Initializes learning agent with all the parameters it needs to start exploring
        and learning from the environment.
        """
        self.env = env
        self.state_size = 2
        self.action_size = len(env.directions)
        self.memory = ExperienceReplayBuffer()
        self.gamma = gamma  
        self.epsilon = epsilon 
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay 
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        # Main network for making decisions and target network for stable learning
        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.td_errors = []  # Keeps track of the learning progress

    def decide_action(self, state):
        """
        Picks an action using our epsilon-greedy strategy - sometimes random, sometimes smart.
        As we learn, we'll do fewer random actions and more smart ones.
        """
        if random.random() < self.epsilon:
            return random.randrange(self.action_size) 
            
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_values = self.q_network.predict(state)
        return torch.argmax(action_values).item()  # Pick the best action

    def learn_from_experience(self, episodes=100, target_update_interval=5):
        """
        The main learning loop - we'll try things out, remember what happened,
        and gradually get better at our task.
        """
        rewards_history = []
        best_reward = float('-inf')

        for episode in range(episodes):
            state = self.env.getState()
            total_reward = 0
            done = False
            steps = 0
            max_steps = 10 

            while not done and steps < max_steps:
                action = self.decide_action(state)
                self.env.action(self.env.directions[action])
                next_state = self.env.getState()

                reward = self.calculate_reward(state, next_state) 
                done = self.env.mixedState(next_state)

                if done:
                    reward += 2.0 

                self.memory.remember(state, action, reward, next_state, done)

                if len(self.memory) > self.batch_size:
                    td_error = self.improve_policy() 
                    self.td_errors.append(td_error)

                state = next_state
                total_reward += reward
                steps += 1

            # Decrease randomness over time
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # Update target network occasionally for stability
            if episode % target_update_interval == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            rewards_history.append(total_reward)
            print(f"Episode {episode + 1}: Steps = {steps}, Total Reward = {total_reward:.2f}, Epsilon = {self.epsilon:.3f}")

            # Saves progress
            if total_reward > best_reward:
                best_reward = total_reward
                self.save_brain("best_model.pth")  

            if episode % 50 == 0:
                self.save_brain()
            self.save_progress(rewards_history)
            self.env.dropObjects()

        return rewards_history

    def calculate_reward(self, state, next_state):
        """
        calculates rewards
        """
        current_variance = sum(state)
        next_variance = sum(next_state)
        reward = (current_variance - next_variance) * 10 

        if next_variance > 0.1:
            reward -= 0.5 

        return reward

    def improve_policy(self): 
        states, actions, rewards, next_states, dones = self.memory.recall(self.batch_size)

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        current_q_values = self.q_network.predict(states).gather(1, actions.unsqueeze(1))

        with torch.no_grad():
            next_q_values = self.target_network.predict(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()

        # Keep our gradients in check
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def save_brain(self, filename="DQL_model.pth"):
        """
        Saves model to a file to  use it later.
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filename)

    def load_brain(self, filename="DQL_model.pth"): 
        """
        Loads a previously saved brain (model) from a file.
        """
        checkpoint = torch.load(filename)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint.get('epsilon', self.epsilon_min)

    def save_progress(self, rewards): 
        """
        Keeps track of  saving rewards and errors.
        """
        np.savetxt("rewards.txt", rewards)
        np.savetxt("td_errors.txt", self.td_errors)

def DQLTest(env, agent, test_episodes=100):
    success_count = 0
    total_time = 0
    
    agent.epsilon = 0.05  
    
    for episode in range(test_episodes):
        env.dropObjects()
        start_time = time.time()
        state = env.getState()
        steps = 0
        max_steps = 15
        
        while not env.mixedState(state) and steps < max_steps:
            action = agent.decide_action(state)
            env.action(env.directions[action])
            state = env.getState()
            steps += 1
        
        episode_time = time.time() - start_time
        if steps < max_steps:
            success_count += 1
            total_time += episode_time
        
        print(f"Test Episode {episode + 1}: {'Success' if steps < max_steps else 'Failure'} in {steps} steps")
    
    return success_count, total_time

def start_training(): 
    env = Simulation() 
    agent = DQLAgent(env)
    rewards = agent.learn_from_experience(episodes=100)
    success_count, time_taken = DQLTest(env, agent, test_episodes=100)
    
    with open("results.txt", "w") as f:
        f.write(f"Deep Q-Learning Results:\n")
        f.write(f"Successful mixes: {success_count}/100\n")
        f.write(f"Average time per successful mix: {time_taken/max(1, success_count):.2f} seconds\n")
        f.write(f"Final training reward: {rewards[-1]:.2f}\n")
        f.write(f"Best training reward: {max(rewards):.2f}\n")

    env.stopSim()
if __name__ == "__main__":
    start_training()