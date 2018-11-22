import gym
import random 
import numpy as np 
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter
#Learning from random moves 

LR = 1e-3   #Learning rate
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500  #frames that we want the pole to be balanced for 
score_requirement = 50  #Learning from all the games that have score of >50
initial_games = 10000   #Amount of times repeated(???)

def some_random_games_first():
    for episode in range(10):
        env.reset()
        for t in range(goal_steps):
            env.render()    #shows whats happening
            action = env.action_space.sample()  #generates a random action 
            #action = tuple(action)
            observation, reward, done, info = env.step(action) #observation is array of data(cart position etc), reward is 0 or 1 was it balanced or not 
            if done:
                break
#some_random_games_first()

 #creating random sample data from the moves
def initial_population():
    training_data=[]    #actual data, observation, but random. Only append if above score requirement 
    scores =[]
    accepted_scores=[]
    for _ in range (initial_games):
        score= 0
        game_memory = []    #store all movements
        prev_observation = []   

        #game happening 
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action) #take action, the data stored in the variables

            if len(prev_observation) > 0:   #if game success, save
                game_memory.append([prev_observation, action])  #could do just observation , action is binary 

            prev_observation = observation
            score+= reward
            if done:
                break

        #analyize 
            if score >= score_requirement:
                accepted_scores.append(score)
                for data in game_memory:    
                    if data[1] ==1:     #the data is prev_observation,action so we're grabbing action which is binary 
                        output = [0,1]
                    elif data[1] == 0:
                        output = [1,0]
                
                    training_data.append([data[0], output])
        
        #each itteration restart the game
        env.reset()
        scores.append(score) #keep track of the scores

    training_data_save = np.array(training_data) #turn the tranining data to numpy array
    np.save('saved.npy',training_data_save)

    print('Average accepted score:',mean(accepted_scores)) 
    print('Median accepeted score:',median(accepted_scores))
    print(Counter(accepted_scores))
    return training_data

initial_population()

def neural_netwok_model(input_size):
    #layers
    network = input_data(shape=[None,input_size,1], name='input')

    network = fully_connected(network, 128, activation = 'relu')
    network=dropout(network,0.8)

    network = fully_connected(network, 256, activation = 'relu')
    network=dropout(network,0.8)

    network = fully_connected(network, 512, activation = 'relu')
    network=dropout(network,0.8)

    network = fully_connected(network, 256, activation = 'relu')
    network=dropout(network,0.8)

    network = fully_connected(network, 128, activation = 'relu')
    network=dropout(network,0.8)

    #output layer
    network = fully_connected(network, 2, activation= 'softmax')
    network = regression(network, optimizer= 'adam',learning_rate=LR,loss='categorical_crossentropy',name='targets')
    
    model = tflearn.DNN(network,tensorboard_dir='log')

    return model    #untrained model

def train_model(training_data, model=False):
    x= np.array([i[0] for i in training_data]).reshape(-1, len(training_data[0][0]), 1) #observation data
    y = [i[1] for i in training_data]

    if not model:
        model = neural_netwok_model(input_size = len(x[0]))
    
    model.fit({'input':x}, {'targets':y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openaistuff')

    return model

training_data = initial_population()
model = train_model(training_data)

scores = []
choices = []

for each_game in range(10):
    score = 0
    game_memory = []
    prev_obs = []
    env.reset
    for _ in range(goal_steps):
        env.render()
        if len(prev_obs) == 0:
            action = random.randrange(0,2)
        else:
            action = np.argmax(model.predict(prev_obs.reshape(-1 , len(prev_obs),1) [0]))
        choices.append(action)

        new_observation, reward, done, info = env.step(action)
        prev_obs = new_observation
        game_memory.append([new_observation, action])
        score += reward
        if done:
            break 
    scores.append(score)
print('average score', sum(scores)/len(scores))
print('Choices 1: {} , Choice 2 {}', format(choices.count(1)/len(choices)), choices.count(0)/len(choices))

