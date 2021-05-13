"""Linear QL agent
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
NUM_EPOCHS = 600
NUM_EPIS_TRAIN = 25  # number of episodes for training at each epoch
NUM_EPIS_TEST = 50  # number of episodes for testing
ALPHA = 0.001  # learning rate for training

ACTIONS = framework.get_actions()
OBJECTS = framework.get_objects()
NUM_ACTIONS = len(ACTIONS)
NUM_OBJECTS = len(OBJECTS)


def tuple2index(action_index, object_index):
    """Converts a tuple (a,b) to an index c"""
    return action_index * NUM_OBJECTS + object_index


def index2tuple(index):
    """Converts an index c to a tuple (a,b)"""
    return index // NUM_OBJECTS, index % NUM_OBJECTS


# pragma: coderesponse template name="linear_epsilon_greedy"
def epsilon_greedy(state_vector, theta, epsilon):
    """Returns an action selected by an epsilon-greedy exploration policy
        Note that the Q-learning algorithm does not specify how we should interact in the world so as to learn quickly.
        It merely updates the values based on the experience collected. If we explore randomly, i.e., always select actions at random, we would most likely not get anywhere.
        A better option is to exploit what we have already learned, as summarized by current Q-values.
        a typical exploration strategy is to follow a so-called epsilon-greedy policy: with probability epsilon take a random action out of 𝐶 with probability 1−epsilon follow that policy.
        The value of 𝜀 here balances exploration vs exploitation. A large value of 𝜀 means exploring more (randomly), not using much of what we have learned.
        A small 𝜀, on the other hand, will generate experience consistent with the current estimates of Q-values.

    Args:
        state_vector (np.ndarray): extracted vector representation
        theta (np.ndarray): current weight matrix
        epsilon (float): the probability of choosing a random command

    Returns:
        (int, int): the indices describing the action/object to take
    """
    # TODO Your code here
    action_index, object_index = None, None
    return (action_index, object_index)
# pragma: coderesponse end


# pragma: coderesponse template
def linear_q_learning(theta, current_state_vector, action_index, object_index,
                      reward, next_state_vector, terminal):
    """Update theta for a given transition

    Note: Q(s,c,theta) can be accessed through q_value = (theta @ state_vector)[tuple2index(action_index, object_index)]

    Args:
        theta (np.ndarray): current weight matrix
        current_state_vector (np.ndarray): vector representation of current state
        action_index (int): index of the current action
        object_index (int): index of the current object
        reward (float): the immediate reward the agent recieves from playing current command
        next_state_vector (np.ndarray): vector representation of next state
        terminal (bool): True if this epsiode is over

    Returns:
        None
    """
    # TODO Your code here
    theta = None # TODO Your update here
# pragma: coderesponse end


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
        current_state = current_room_desc + current_quest_desc
        current_state_vector = utils.extract_bow_feature_vector(
            current_state, dictionary)
        # TODO Your code here

        if for_training:
            # update Q-function.
            # TODO Your code here
            pass

        if not for_training:
            # update reward
            # TODO Your code here
            pass

        # prepare next step
        # TODO Your code here

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
    global theta
    theta = np.zeros([action_dim, state_dim])

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
    action_dim = NUM_ACTIONS * NUM_OBJECTS

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

