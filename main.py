import torch
import visdom
from model import FineDustModel
from model import train
from model import predict
from model import cal_loss
from dataload import DustTimeDataset


trainset = DustTimeDataset('./data/train', 6)
testset = DustTimeDataset('./data/test', 6)
train_x, train_y = trainset[:]
test_x, test_y = testset[:]
trainloader = DataLoader(trainset, batch_size=32, shuffle=True)

model= FineDustModel(0.3)
if torch.cuda.is_available():
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_func = nn.L1Loss()

vis = visdom.Visdom()


for epoch in range(1000):
    model.train()
    for i, (x_val, y_val) in enumerate(trainloader, 19):
        train(model, loss_func, optimizer, x_val, y_val)
    train_loss = cal_loss(model, train_x, train_y)
    test_loss = cal_loss(model, test_x, test_y)
    print(epoch,'-->' , train_loss, '----', test_loss)
    
    if epoch == 1:
        plot = vis.line(
            Y=np.column_stack([train_loss, test_loss]), 
            X=np.column_stack([np.array([epoch]), np.array([epoch])]),
            opts={'title': 'PM25 Loss', 'legend': ['Train', 'Test'], 'showlegend': True}
        )
    elif epoch > 1:
        vis.line(
            Y=np.column_stack([train_loss, test_loss]), 
            X=np.column_stack([np.array([epoch]), np.array([epoch])]),
            win=plot, update='append',
            opts={'title': 'PM25 Loss', 'legend': ['Train', 'Test'], 'showlegend': True}
        )