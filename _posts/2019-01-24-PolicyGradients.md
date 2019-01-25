---
layout: post
title:  "Policy Gradients"
date:   2019-01-24 11:42:04 +0530
categories: Reinforcement Learning
author: Madhuparna Bhowmik
---

In this article, we will learn about Policy Gradients and implement it in Pytorch.

## Introduction

Policy Gradients method is a model-free Reinforcement Learning algorithm that directly approximates the Policy. 

In the previous blog posts, we saw Q-learning based algorithms like DQN and DRQNs where given a state we were finding the Q-values of the possible actions where the Q-values are the expected reward we can get from that state if that action is selected. And we were using an epsilon-greedy strategy to select the action. However, in Policy Gradients method instead of approximating the action value function, we are directly learning the optimal policy. 

## Advantages of Policy Gradient Method

1.Better Convergence properties.

2.Continuous Action Space - We cannot use Q-learning based methods for environments having Continuous action space. However, policy gradient methods like DDPG ( Deep deterministic Policy Gradients ) can be used for such cases.

3.Policy Gradients can learn Stochastic policies.
As we will see in the Implementation details section that we choose the action stochastically and hence we need not use something like an epsilon-greedy strategy that we used in Q-learning, the policy will itself be stochastic.

## Policy Gradients 

In Policy Gradients Method we are trying to learn the optimal policy. Here, we can take the example of a function
 y = f(x)
i.e., for some input x, the output is f(x) or y. 
Similarly, we can say that the policy is a function such that when different states are given as inputs, it will output the probabilities corresponding to the different actions. 
So, our goal is to approximate this function so that we can get the maximum possible reward.

When we use a neural network, we can say that the weights of the neural network are the parameters of the policy. Therefore we need to update these parameters in such a way that the expected reward is maximized. 

So, in general, classification problems, we have a loss function, and our objective is to minimize the loss. Here we have the reward function, and our objective is to maximize the reward.

$$J(θ)= \underset{t \sim \pi _\theta}E[R(\tau) ]$$

Here, the policy is parametrized by $$\theta$$, and $$J(\theta)$$ is the reward function.
Now we need to find the Gradient of this function with respect to the parameters of the policy *theta* so that we can perform the parameter update as follows:-

$$ \theta = \theta + \alpha \times \Delta J(\theta)$$

However, directly calculating the gradient is tricky as it depends upon both the policy and the state distribution. That is, since the environment is unknown we do not know the effect of policy update on state distribution. Therefore, we need to use the policy gradient theorem which provides a reformation such that the gradient does not depend upon the state.

![PGequations](/assets/PGequations.png){:height="80%" width="100%"}

( [Image Source](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html) )

For the proof of this theorem, you can refer to the book Sutton & Barto, 2017; Sec. 13.1.

## Policy Gradient Algorithm

Pseudo Code:


1.Initialize parameters $$\theta$$

2.for i in number of episodes:

3.Generate trajectory $$ s_0, a_0, r_0, s_1, a_1, r_1....... s_{t-1}, a_{t-1}, r_{t-1}, s_t $$ using the policy.

4.for t in timestep:

5.Estimate the return value

6.Evaluate the policy and update the parameters.


So we Generate a trajectory using the current policy, i.e., by taking actions according to the current policy. This trajectory can be one episode of an episodic task or a fixed number of time steps for a non-episodic task. Then we evaluate the current policy using the reward function and calculate the gradients and update the parameters.

## Implementational Details

In this section, we will go through the code to implement this algorithm in Pytorch. We will use OpenAI's Cartpole environment.

1.Importing necessary libraries

{% highlight python %}
import random
import math
import gym
import numpy as np
import PIL
from PIL import Image
import matplotlib
import matplotlib.cm as cm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical
{% endhighlight %}

2.Defining the model

{% highlight python %}
class model(nn.Module):

    def __init__(self):
        super(model,self).__init__()
        self.fc1 = nn.Linear(4,16)
        self.fc2 = nn.Linear(16,32)
        self.fc3 = nn.Linear(32, 2)
        
    def forward(self,x):
        x=x.reshape(x.shape[0],1,4)
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        return F.sigmoid(self.fc3(x.contiguous().view(x.size(0), -1)))
{% endhighlight %}


 3.Setting up the environment and declaring optimizer

{% highlight python %}
policy=model()
optimizer = optim.RMSprop(policy.parameters())
gamma = 0.99
env = gym.make('CartPole-v0').unwrapped
{% endhighlight %}

4.Policy Gradient Algorithm

So the way we select the actions is by sampling from the probability distribution, i.e., the policy outputs probability of taking each action and from that distribution, we sample an action. For this, we can use the Categorical distribution from Pytorch which samples a number based on the probability corresponding to each index, i.e., if the input is 
0.5, 0.3, 0.2, then it will sample with a probability of 0.5 for 0, 0.3 for 1 and 0.2 for 2.


{% highlight python %}

def PolicyGradient(numepisodes):
    rew =0 
    max_reward = 0
    for i in range(numepisodes):
        states = []     #List to store the states
        actions = []    #List to store the actions
        rewards = []    #List to store the rewards
        done = False
        state = env.reset()
        steps = 0
        states.append(state)
        rr=0
        while done != True and steps<1500:
            steps+=1
            inp = torch.from_numpy(state).unsqueeze(0)
            inp= inp.type('torch.FloatTensor')
            act = policy(inp)
            m = Categorical(act)
            action = m.sample()
            state,reward,done,_=env.step(int(action.item()))
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            rew+=reward
            rr+=reward
        max_reward = max(max_reward,rr)
        optimizer.zero_grad()
        loss_sum =0
        rewards[-1] = -1
        cum_reward =0
        
        # Calculate the expected rewards
        for j in range(steps-1,0,-1):
            if rewards[j]!=0:
                cum_reward = cum_reward*gamma + rewards[j]
                rewards[j]=cum_reward
                
        #Normalize the rewards     
        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards)
        for j in range(steps):
            rewards[j] = (rewards[j] - rewards_mean) / rewards_std
        for j in range(steps):
            inp = torch.from_numpy(states[j]).unsqueeze(0)
            inp = inp.type('torch.FloatTensor')
            acts = policy(inp)
            m = Categorical(acts)
            loss = -m.log_prob(actions[j]) * rewards[j]
            loss.backward()
            loss_sum+=loss
        optimizer.step()
        if i % 50 == 0 and i!=0:            
            print("Episode ",i,rew/100)
            rew =0

{% endhighlight %}

5.Run the code - 

{% highlight python %}

PolicyGradient(500)

{% endhighlight %}

6.Output - 

{% highlight python %}
Episode  50 14.53
Episode  100 23.37
Episode  150 46.66
Episode  200 352.57
Episode  250 147.85
Episode  300 720.52
Episode  350 740.87
Episode  400 463.9
Episode  450 74.81

{% endhighlight %}

The code is available on [Github](https://gist.github.com/Madhuparna04/f846030cfaa48dcfd0ceeb8a435e8d06).


## Results

Even though in the above example the policy converges quite fast in just 200 episodes we observe an average reward of 300+ while it goes down and up thereafter. However, if we run the same code many times we will observe that sometimes the average reward does not even increase beyond 10. The reason behing this is Policy Gradients algorithms are prone to getting stuck in some local optimum and not doing well at all thereafter. But Q-learning based algorithms are less prone to getting stuck in local optimum. This is one of the disadvantages of Policy Gradients.





