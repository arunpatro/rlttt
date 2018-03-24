import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable

board = [[0, 1, 2],
         [3, 4, 5],
         [6, 7, 8]]

x = -1 
o = +1

# b = np.zeros([8,9])
# b[0][:2] = 1
# for (idx, row) in enumerate(b):
#     row[idx] = 1
#     print idx, row
# s_win = [row[j] = 1 for row in enumerate(b)]

input_size = 9
hidden_size = 9
num_classes = 9
num_epochs = 5
# batch_size = 100
learning_rate = 0.001
reward = 1.0
gamma = 0.6


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
    
q_net = Net(input_size, hidden_size, num_classes)
 
# Loss and Optimizer
criterion = nn.MSELoss()  
optimizer = torch.optim.Adam(q_net.parameters(), lr=learning_rate)  

def getNextStates(state, player=1):
    out = np.zeros([2,9], dtype='f')
    out[0][2:4] = player
    out[1][5:7] = player
    return out


def update_q():
    state = np.zeros((9), dtype='f')
    state[5] = 1
    state = Variable(torch.from_numpy(state))
    optimizer.zero_grad()  # zero the gradient buffer
    qsa = q_net(state)

    next_states = getNextStates(state, player=x)

    q_s_a_ = []
    for state in next_states:
        state = Variable(torch.from_numpy(state))
        q = q_net(state)
        q_s_a_.append(np.max(q.data.numpy()))

    max_a_idx = np.argmax(q_s_a_)
    max_q_s_a_ = np.max(q_s_a_)
    qTarget = qsa.clone()
    qTarget[max_a_idx] = reward(state, max_a_idx, player = x) + gamma * max_q_s_a_

    loss = torch.sum((qsa - qTarget) * (qsa - qTarget)) / qsa.data.nelement()
    loss.backward()
    optimizer.step()

    print 'done'


def reward(state, max_a_idx, player):
    winning_states = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
    if not state[max_a_idx] is 0:
        return -20
    else:
        state[max_a_idx] += player
        for i in winning_states:
            if state[i[0]] == state[i[1]] == state[i[2]] == player:
                return 1
        return 0

class Board(object):
    """docstring for Board"""
    def __init__(self, state=np.zeros(9)):
        super(Board, self).__init__()
        self.state = state
        self.dic = {1:'x', -1:'o', 0:'.'}

    def show(self,):
        i = self.state
        print self.dic[i[0]], self.dic[i[1]], self.dic[i[2]]
        print self.dic[i[3]], self.dic[i[4]], self.dic[i[5]]
        print self.dic[i[6]], self.dic[i[7]], self.dic[i[8]]



def main():
    print 'Enter [x] or [o]:'
    choice = raw_input()

    if choice == 'x':
        player = 1
    else:
        player = 0

    b = Board()
    b.show()


    pass

main()