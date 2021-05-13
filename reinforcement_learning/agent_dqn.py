"""DQN QL agent

    Simple Policy-learning algorithm
    In this project, we address the task of learning control policies for text-based games using reinforcement learning.
    In these games, all interactions between players and the virtual world are through text.
    The current world state is described by elaborate text, and the underlying state is not directly observable.
    Players read descriptions of the state and respond with natural language commands to take actions.

    For this project you will conduct experiments on a small Home World, which mimic the environment of a typical house.The world consists of a few rooms, and each room contains a representative object that the player can interact with.
    For instance, the kitchen has an apple that the player can eat. The goal of the player is to finish some quest. An example of a quest given to the player in text is You are hungry now .
    To complete this quest, the player has to navigate through the house to reach the kitchen and eat the apple.
    In this game, the room is hidden from the player, who only receives a description of the underlying room.
    At each step, the player read the text describing the current room and the quest, and respond with some command (e.g., eat apple ).
    The player then receives some reward that depends on the state and his/her command.

    In order to design an autonomous game player, we will employ a reinforcement learning framework to learn command policies using game rewards as feedback.
    Since the state observable to the player is described in text, we have to choose a mechanism that maps text descriptions into vector representations.

    A naive approach is to create a map that assigns a unique index for each text description.  -- agent_tabular_ql.py

    However, such approach becomes difficult to implement when the number of textual state descriptions are huge.
    An alternative method is to use a bag-of-words representation derived from the text description. -- agent_linear.py

    As observed in agent_linear.py, a linear model is not able to correctly approximate the Q-function for our simple task.
    Deep-learning approach -- agent_dqn.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = False

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 300
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)

model = None
optimizer = None


def epsilon_greedy(state_vector,current_room_index, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy

    Args:
        state_vector (torch.FloatTensor): extracted vector representation
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    (action_index, object_index) = (None, None)
    choice = np.random.choice([0,1], p=[epsilon, 1-epsilon])

    valid_actions = framework.command_is_valid[current_room_index,...]
    valid_actions = np.argwhere(valid_actions == 1)
    random_actions = []
    for (i,j) in valid_actions:
        random_actions.append((i,j))
    if choice == 1 :  # choose the best one
        action_values, object_values = model(state_vector)
        action_index = action_values.argmax()
        object_index = object_values.argmax()
        if (action_index, object_index) in random_actions:
            pass
        else:
            choice = 0

    if choice == 0: # random choose from valid actions
        choice = np.random.choice(len(random_actions))
        (action_index, object_index) = random_actions[choice]

    return (action_index, object_index)

class DQN(nn.Module):
    """A simple deep Q network implementation.
    Computes Q values for each (action, object) tuple given an input state vector
    """

    def __init__(self, state_dim, action_dim, object_dim, hidden_size=100):
        super(DQN, self).__init__()
        self.state_encoder = nn.Linear(state_dim, hidden_size)
        self.state2action = nn.Linear(hidden_size, action_dim)
        self.state2object = nn.Linear(hidden_size, object_dim)

    def forward(self, x):
        state = F.relu(self.state_encoder(x))
        return self.state2action(state), self.state2object(state)


# pragma: coderesponse template
def deep_q_learning(current_state_vector, action_index, object_index, reward,
                    next_state_vector, terminal):
    """Updates the weights of the DQN for a given transition

    Args:
        current_state_vector (torch.FloatTensor): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (torch.FloatTensor): vector representation of next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """
    with torch.no_grad():
        q_values_action_next, q_values_object_next = model(next_state_vector)

    # calculate the max Q-value of (next state, *) -- i.e. the V-value (next state)
    maxq_next = 1 / 2 * (q_values_action_next.max()
                         + q_values_object_next.max())
    # calculate the new Q-value of (state, action, object)
    new_q_value = reward + GAMMA * maxq_next

    # find the current state Q-values (state, *)
    q_values_cur_state_actions, q_values_cur_state_objects = model(current_state_vector)
    # calculate the current Q-value of (state, action, object)
    cur_q_value = 1 /2 * (q_values_cur_state_actions[action_index] + q_values_cur_state_objects[object_index])

    # Current State value  = max{Q(state, *) for all actions, objects}

    # loss function: square distance
    loss = 1 / 2 * (cur_q_value - new_q_value)**2

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
# pragma: coderesponse end


def run_episode(for_training):
    """
        Runs one episode
        If for training, update Q function
        If for testing, computes and return cumulative discounted reward
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP
    epi_reward = None

    # initialize for each episode
    # TODO Your code here

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    while not terminal:
        # Choose next action and execute
        current_state = current_room_desc + current_quest_desc
        current_state_vector = torch.FloatTensor(
            utils.extract_bow_feature_vector(current_state, dictionary))

        current_room_index = framework.rooms_desc_map[current_room_desc]
        action_index, object_index = epsilon_greedy(current_state_vector, current_room_index, epsilon)
        (next_room_desc, next_quest_desc, reward, terminal) = framework.step_game(current_room_desc, current_quest_desc,
                                                                                  action_index, object_index)
        next_state = next_room_desc + next_quest_desc
        next_state_vector = torch.FloatTensor(utils.extract_bow_feature_vector(next_state, dictionary))

        if for_training:
            # update Q-function.
            deep_q_learning(current_state_vector, action_index, object_index, reward, next_state_vector, terminal)


        if not for_training:
            # update reward
            if epi_reward == None:
                epi_reward = reward
            else:
                epi_reward += reward

        # prepare next step
        current_room_desc = next_room_desc
        current_quest_desc = next_quest_desc

    if not for_training:
        return epi_reward


def run_epoch():
    """Runs one epoch and returns reward averaged over test episodes"""
    rewards = []

    for _ in range(NUM_EPIS_TRAIN):
        run_episode(for_training=True)

    for _ in range(NUM_EPIS_TEST):
        rewards.append(run_episode(for_training=False))

    return np.mean(np.array(rewards))


def run():
    """Returns array of test reward per epoch for one run"""
    global model
    global optimizer
    model = DQN(state_dim, NUM_ACTIONS, NUM_OBJECTS)
    optimizer = optim.SGD(model.parameters(), lr=ALPHA)

    single_run_epoch_rewards_test = []
    pbar = tqdm(range(NUM_EPOCHS), ncols=80)
    for _ in pbar:
        single_run_epoch_rewards_test.append(run_epoch())
        pbar.set_description(
            "Avg reward: {:0.6f} | Ewma reward: {:0.6f}".format(
                np.mean(single_run_epoch_rewards_test),
                utils.ewma(single_run_epoch_rewards_test)))
    return single_run_epoch_rewards_test


if __name__ == '__main__':
    state_texts = utils.load_data('game.tsv')
    dictionary = utils.bag_of_words(state_texts)
    state_dim = len(dictionary)

    # set up the game
    framework.load_game_data()

    epoch_rewards_test = []  # shape NUM_RUNS * NUM_EPOCHS

    for _ in range(NUM_RUNS):
        epoch_rewards_test.append(run())

    epoch_rewards_test = np.array(epoch_rewards_test)

    x = np.arange(NUM_EPOCHS)
    fig, axis = plt.subplots()
    axis.plot(x, np.mean(epoch_rewards_test,
                         axis=0))  # plot reward per epoch averaged per run
    axis.set_xlabel('Epochs')
    axis.set_ylabel('reward')
    axis.set_title(('Linear: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
