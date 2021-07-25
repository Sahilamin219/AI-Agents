# Taxi-ai
AI for playing taxi-v3 (in gym env.)

# Taxi-Agent
What is Q-Learning?
Q learning is an off-policy based Algorithim which uses valued based mrthod for finding its optimal policy and uses TD-approach for training its action-value function.
It is used for traing Q-function (an action value function) which bascially complete its Q-Table.


Minesweeper logic
    power = +1
    mine = -100
    end = +100
    
    NewQ(Start, right) = Q(start,  right) + aplha*(some delta value)
    aplha=learning rate =0.9
    some delta value = Reward at that state + (maxQ`(actions)) - Q(start, right);
    actions = left, right,up, down
    maxQ` = gamma*(max(actions)) 
    gamma = discount rate;

    
![Screenshot from 2021-07-08 18-54-27](https://user-images.githubusercontent.com/48405411/126893486-68208bab-d6d6-4b3c-bf16-3b7905f6f0b4.png)
![Screenshot from 2021-07-08 18-54-24](https://user-images.githubusercontent.com/48405411/126893490-51d69f3a-128c-4ff9-90ac-b76be2f4b9ec.png)
