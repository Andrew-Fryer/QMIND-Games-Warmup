import random
import gym
import tflearn
import numpy as np
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import mean, median
from collections import Counter

LR = 1e-5
env = gym.make('CartPole-v0')
env.reset()
goal_steps = 500
score_req = 70
initial_games = 10000

def randgame():
    for episode in range(initial_games):
        env.reset()
        for t in range(goal_steps):
            env.render()
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                break

def initpop():
    traindata = []
    scores = []
    goodscores = []
    for _ in range(initial_games):
        score = 0
        gamemem = []
        prevob = []
        for _ in range(goal_steps):
            action = random.randrange(0,2)
            observation, reward, done, info = env.step(action)

            if len(prevob)>0:
                gamemem.append([prevob,action])

            prevob=observation
            score += reward
            if done:
                break
        if score >= score_req:
            goodscores.append(score)
            for data in gamemem:
                if data[1] ==1:
                    output = [0, 1]
                elif data[1] == 0:
                    output = [1, 0]
                traindata.append([data[0], output])

        env.reset()
        scores.append(score)

    traindatasave = np.array(traindata)
    np.save('saved.npy',traindatasave)

    print('Ave good score', mean(goodscores))
    print('Med good score', median(goodscores))
    print(Counter(goodscores))

    return traindata

def neuralnet(input_size):
    network = input_data(shape = [None, input_size,1], name = 'input')

    network = fully_connected(network,128,activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network,256,activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network,512,activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network,256,activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network,128,activation='relu')
    network = dropout(network, 0.8)

    network = fully_connected(network, 2, activation='softmax')
    network = regression(network, optimizer='adam',learning_rate=LR, loss='categorical_crossentropy',name='targets')

    model = tflearn.DNN(network, tensorboard_dir='log')

    return model

def trainmodel(traindata,model=False):
    X = np.array([i[0] for i in traindata]).reshape(-1, len(traindata[0][0]),1)
    y = [i[1] for i in traindata]

    if not model:
        model = neuralnet(input_size=len(X[0]))

    model.fit({'input':X},{'targets':y}, n_epoch=15,snapshot_step=500, show_metric=True, run_id='openaistuff')

    return model

traindata = initpop()
model = trainmodel(traindata)

scores = []
choices = []

for eachgame in range(10):
    score = 0
    gammem = []
    prevob = []
    env.reset()
    for _ in range(goal_steps):
        env.render()
        if len(prevob)==0:
            action = random.randrange(0,1)
        else:
            action = np.argmax(model.predict(prevob.reshape(-1, len(prevob),1))[0])
        choices.append(action)
        newob, reward, done, info = env.step(action)
        prevob = newob
        gammem.append([newob,action])
        score += reward
        if done:
           break
    scores.append(score)

print('avg', sum(scores)/len(scores))
print('c1: {}, C2: {}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
