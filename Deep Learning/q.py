from environment import MountainCar
import numpy as np
class LinearModel:
    def __init__(self, state_size: int, action_size: int, lr: float, indices: bool):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        self.indices = indices

        self.w = np.array([[0]*state_size]*action_size)
        self.b = 0

    """indices is True if indices are used as input for one-hot features.
    Otherwise, use the sparse representation of state as features
    """
    def predict(self, state):
        state_vector = np.zeros(self.state_size)
        for key,value in state.items():
            state_vector[key] = value
        Q_list = []
        for i in range(self.action_size):
            Q_i = np.dot(self.w[i],state_vector)+self.b
            Q_list.append(Q_i)
        return Q_list
    """
    Given state, makes predictions.
    """
    
    def update(self, state, action: int, target: int):
        state_vector = np.zeros(self.state_size)
        for key,value in state.items():
            state_vector[key] = value
        temp = np.dot(state_vector,self.w[action]) + self.b - target
        grad_w = self.lr*temp*state_vector
        self.w = self.w - grad_w
        grad_b = self.lr*temp
        self.b = self.b - grad_b
        return 
    """
    Given state, action, and target, update weights.
    """

class QLearningAgent:
    def __init__(self, env: MountainCar, mode: str, gamma: float, lr: float, epsilon: float):
        self.state = env.state
        self.mode = mode
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon

    def get_action(self, state):
        if np.random.random() >= self.epsilon:
            action = np.argmax(lm.predict(state))
        else:
            action = np.random.randint(0,3)
        return action
    """epsilon-greedy strategy.
    Given state, returns action.
    """
    def train(self, episodes: int, max_iterations: int):
        reward_list = []
        for i in range(episodes):
            e_reward = 0.0
            self.state = env.reset()
            for j in range(max_iterations):
                action = self.get_action(self.state)
                next_state, reward, done= env.step(action)
                target = reward + self.gamma*max(lm.predict(self.state))
                e_reward += reward
                lm.update(self.state,action,target)
                self.state = next_state
                if done:
                    break
            reward_list.append(e_reward)
        return reward_list
    """training function.
    Train for ’episodes’ iterations, where at most ’max_iterations‘ iterations
    should be run for each episode. Returns a list of returns.
    """

    def output(self,outfile):
        outfile = open(outfile, "w", encoding="utf8")
        outfile.write('{}\n'.format())
        return 

# if __name__ == "main":
# run parser and get parameters values
mode = "tile" #
episodes = 25 #
iterations = 200 #
epsilon = 0.0 #
gamma = 0.99 #
lr = 0.005 #
indices = 1 if mode=="tile" else 0

env = MountainCar(mode=mode,fixed=1)
lm = LinearModel(state_size=env.state_space,action_size=3,lr=lr,indices=indices)
agent = QLearningAgent(env, mode, gamma, epsilon, lr)
returns = agent.train(episodes, iterations)
print(returns)