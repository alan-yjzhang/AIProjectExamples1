"""Tabular QL agent
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

    Deep-learning approach -- agent_dqn.py

"""
# https://learning.edx.org/course/course-v1:MITx+6.86x+1T2021/block-v1:MITx+6.86x+1T2021+type@sequential+block@P5_rl/block-v1:MITx+6.86x+1T2021+type@vertical+block@P5_rl-tab3

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import framework
import utils

DEBUG = True

GAMMA = 0.5  # discounted factor
TRAINING_EP = 0.5  # epsilon-greedy parameter for training
TESTING_EP = 0.05  # epsilon-greedy parameter for testing
NUM_RUNS = 10
NUM_EPOCHS = 200
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.1  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


# pragma: coderesponse template
def epsilon_greedy(state_1, state_2, q_func, epsilon):
    """Returns an action selected by an epsilon-Greedy exploration policy

    Note that the Q-learning algorithm does not specify how we should interact in the world so as to learn quickly.
        It merely updates the values based on the experience collected. If we explore randomly, i.e., always select actions at random, we would most likely not get anywhere.
        A better option is to exploit what we have already learned, as summarized by current Q-values.
        a typical exploration strategy is to follow a so-called epsilon-greedy policy: with probability epsilon take a random action out of 𝐶, and with probability 1−epsilon follow the best policy.
        The value of 𝜀 here balances exploration vs exploitation. A large value of 𝜀 means exploring more (randomly), not using much of what we have learned.
        A small 𝜀, on the other hand, will generate experience consistent with the current estimates of Q-values.

    Args:
        state_1, state_2 (int, int): two indices describing the current state
        q_func (np.ndarray): current Q-function
        epsilon (float): the probability of choosing a random command
                with probabily (1 - epsilon) following that command.
    Returns:
        (int, int): the indices describing the action/object to take
    """
    current_room_index = state_1
    best_action_value = 0
    (action_index, object_index) = (None, None)
    choice = np.random.choice([0,1], p=[epsilon, 1-epsilon])

    valid_actions = framework.command_is_valid[current_room_index,...]
    valid_actions = np.argwhere(valid_actions == 1)
    random_actions = []
    for (i,j) in valid_actions:
        random_actions.append((i,j))
        qValue = q_func[state_1, state_2, i, j]
        if qValue > best_action_value :
            best_action = (i, j)
            best_action_value = qValue

    if best_action_value > 0 and choice == 1 :  # choose the best one
        (action_index, object_index) = best_action
    else: # random choose from valid actions
        choice = np.random.choice(len(random_actions))
        (action_index, object_index) = random_actions[choice]

    #
    # valid_actions = []
    # best_action = (None, None)
    # for i in range(NUM_ACTIONS):
    #     for j in range(NUM_OBJECTS):
    #         if (framework.command_is_valid[current_room_index, i, j] == 1):
    #             valid_actions.append((i,j))
    #             qValue = q_func[state_1, state_2, i, j]
    #             if qValue > best_action_value :
    #                 best_action = (i, j)
    #                 best_action_value = qValue
    # if best_action_value > 0 and choice == 1 :  # choose the best one
    #     (action_index, object_index) = best_action
    # else: # random choose from valid actions
    #     choice = np.random.choice(len(valid_actions))
    #     (action_index, object_index) = valid_actions[choice]

    return (action_index, object_index)


def state_value(q_func, state_1, state_2):
    """
    V(s) = max Q(s,c) for all c, where c represents an action of a on object b: (a, b)
    """
    q_v = q_func[state_1, state_2, ...]
    return np.amax(q_v)


def tabular_q_learning(q_func, current_state_1, current_state_2, action_index,
                       object_index, reward, next_state_1, next_state_2,
                       terminal):
    """Update q_func for a given transition

        Qnew(s,c) = (1-alpha) * Qold(s,c) + alpha * (Reward(s,c,s') + gamma * V(s'))
        V(s) = max Q(s,c) for all c, where c represents an action of a on object b: (a, b)

    Args:
        q_func (np.ndarray): current Q-function
        current_state_1, current_state_2 (int, int): two indices describing the current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_1, next_state_2 (int, int): two indices describing the next state
        terminal (bool): True if this episode is over

    Returns:
        None
    """
    stateValue = state_value(q_func, current_state_1, current_state_2)

    currentQ = q_func[current_state_1, current_state_2, action_index,
           object_index]

    q_func[current_state_1, current_state_2, action_index,
           object_index] = (1 - ALPHA) * currentQ + ALPHA * (reward + GAMMA * stateValue)

    return None  # This function shouldn't return anything


# In this section, you will evaluate your learning algorithm for the Home World game.
# The metric we use to measure an agent's performance is the cumulative discounted reward obtained per episode averaged over the episodes.
# The evaluation procedure is as follows. Each experiment (or run) consists of multiple epochs (the number of epochs is NUM_EPOCHS).
# In each epoch:
#  1. You first train the agent on NUM_EPIS_TRAIN episodes, following an 𝜀-greedy policy with 𝜀=TRAINING_EP and updating the 𝑄 values.
#  2. Then, you have a testing phase of running NUM_EPIS_TEST episodes of the game, following an 𝜀-greedy policy with 𝜀=TESTING_EP,
#     which makes the agent choose the best action according to its current Q-values 95% of the time.
#     At the testing phase of each epoch, you will compute the cumulative discounted reward for each episode and then obtain the average reward over the NUM_EPIS_TEST episodes.
# Finally, at the end of the experiment, you will get a sequence of data (of size NUM_EPOCHS) that represents the testing performance at each epoch.
#
# Note that there is randomness in both the training and testing phase. You will run the experiment NUM_RUNS times and then compute the averaged reward performance over NUM_RUNS experiments.
#

def run_episode(for_training):
    """ Runs one episode
    If for training, update Q function
    If for testing, computes and return cumulative discounted reward

    Args:
        for_training (bool): True if for training

    Returns:
        None
    """
    epsilon = TRAINING_EP if for_training else TESTING_EP

    epi_reward = None
    # initialize for each episode
    # TODO Your code here

    (current_room_desc, current_quest_desc, terminal) = framework.newGame()
    while not terminal:
        # Choose next action and execute

        current_room_index = framework.rooms_desc_map[current_room_desc]
        quest_index = framework.quests_map[current_quest_desc]

        (action, object) = epsilon_greedy(current_room_index, quest_index, q_func, epsilon)
        (next_room_desc, next_quest_desc, reward, terminal) = framework.step_game(current_room_desc, current_quest_desc, action, object)

        if for_training:
            # update Q-function.
            next_room_index = framework.rooms_desc_map[next_room_desc]
            next_quest_index = framework.quests_map[next_quest_desc]
            tabular_q_learning(q_func, current_room_index, quest_index, action, object, reward, next_room_index, next_quest_index, terminal)


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


# pragma: coderesponse end


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
    global q_func
    q_func = np.zeros((NUM_ROOM_DESC, NUM_QUESTS, NUM_ACTIONS, NUM_OBJECTS))

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
    # Data loading and build the dictionaries that use unique index for each state
    (dict_room_desc, dict_quest_desc) = framework.make_all_states_index()
    NUM_ROOM_DESC = len(dict_room_desc)
    NUM_QUESTS = len(dict_quest_desc)

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
    axis.set_title(('Tablular: nRuns=%d, Epilon=%.2f, Epi=%d, alpha=%.4f' %
                    (NUM_RUNS, TRAINING_EP, NUM_EPIS_TRAIN, ALPHA)))
    plt.show()
