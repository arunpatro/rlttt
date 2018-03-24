import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import pickle

input_size = 9
hidden_size = 9
num_classes = 9
num_epochs = 5
# batch_size = 100
learning_rate = 0.001
reward = 1.0
gamma = 0.6
winning_states = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]


# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)  
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
# q_net = Net(input_size, hidden_size, num_classes)
q_net = torch.load('arun_qnet.tz') 

# Loss and Optimizer
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)  

def getNextStates(state, player=1):
    out = []
    for i in range(9):
        temp = state.clone() 
        if not temp[i].data.numpy():
            temp[i] = player
        out.append(temp)
    return out


def think_move(state, player):
    state = Variable(torch.from_numpy(state))
    qsa = q_net(state)
    return np.argmax(gsa.data.numpy())


def update_q(state, player = 1):
    print 'q player', player
    state = Variable(torch.from_numpy(state))
    optimizer.zero_grad()  # zero the gradient buffer
    qsa = q_net(state)

    next_states = getNextStates(state, player)
    # print 'next_states', next_states

    q_s_a_ = []
    for state in next_states:
        q = q_net(state)
        q_s_a_.append(np.max(q.data.numpy()))

    # print 'q_s_a_', q_s_a_
    max_a_idx = np.argmax(q_s_a_)
    # print max_a_idx, type(max_a_idx)
    max_q_s_a_ = np.max(q_s_a_)
    qTarget = qsa.clone()
    qTarget[max_a_idx] = reward(state, max_a_idx, player) + gamma * max_q_s_a_

    loss = torch.sum((qsa - qTarget) * (qsa - qTarget)) / qsa.data.nelement()
    print 'loss', loss
    loss.backward()
    optimizer.step()

    return max_a_idx


def reward(state, max_a_idx, player = -1):
    temp_state = state.clone()
    cond = temp_state[max_a_idx].data.numpy()

    if not cond:
        return -20
    else:
        # print temp_state[max_a_idx], type(temp_state[max_a_idx]), player, type(player)
        temp_state[max_a_idx] += player
        for i in winning_states:
            if temp_state[i[0]] == temp_state[i[1]] == temp_state[i[2]] == player:
                return 1
        return 0.1

class Board(object):
    """docstring for Board"""
    def __init__(self, state=np.zeros(9, dtype='f'), player = 1):
        super(Board, self).__init__()
        self.state = state
        self.dic = {1:'x', -1:'o', 0:'.'}
        self.player = player

    def show(self,):
        i = self.state
        print '---------'
        print '|', self.dic[i[0]], self.dic[i[1]], self.dic[i[2]], '|'
        print '|', self.dic[i[3]], self.dic[i[4]], self.dic[i[5]], '|'
        print '|', self.dic[i[6]], self.dic[i[7]], self.dic[i[8]], '|'
        print '---------'

    def clear(self,):
        self.state = np.zeros(9, dtype='f')

    def play(self, action, player):
        if self.state[action] == 0:
            self.state[action] += player

        if self.win():
            print 'Done'
            self.show()
            exit()
        pass

    def win(self,):
        for i in winning_states:
            if self.state[i[0]] == self.state[i[1]] == self.state[i[2]] == self.player:
                return True
            elif self.state[i[0]] == self.state[i[1]] == self.state[i[2]] == -1 * self.player:
                return True


def main():
    print 'Enter [x] or [o]:'
    choice = raw_input()

    if choice == 'x':
        b = Board()
    else:
        b = Board(player = -1)

    b.show()

    for i in range(5):
        if i % 2 == 0:
            print 'Enter position (0-9) for [' + choice + ']:'
            move = int(raw_input())
            b.play(move, b.player)
            b.show()
        else:
            move = update_q(b.state, -1 * b.player)
            print 'Computer playing at ', move
            b.play(move, -1 * b.player)
            b.show()
    pass

main()

print 'saving net'
torch.save(q_net, 'arun_qnet.tz')

