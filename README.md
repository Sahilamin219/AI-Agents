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

    
