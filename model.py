import torch
from torch import nn
from torch.autograd import Variable

class FineDustModel(nn.Module):
    
    def __init__(self, dropout_p):
        super(MiseModel, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=10, out_channels=16, kernel_size=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        self.conv2 = self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.conv_fc = nn.Sequential(
            nn.Linear(400, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )
        self.encoder = nn.LSTM(input_size=128, hidden_size=128, num_layers=4, batch_first=True, dropout=dropout_p)
        self.decoder = nn.LSTM(input_size=128, hidden_size=128, num_layers=4, batch_first=True, dropout=dropout_p)
        self.decoder_fc =  nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 25)
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        time_step = x.size(1)
        
        x = x.view(batch_size * time_step, 10, 6, 6)
        brach3 = self.conv3(x)
        brach1 = self.conv1(x)
        x = torch.cat([brach3, brach1], dim=1)
        x = self.conv2(x)
        x = x.view(batch_size * time_step, -1)
        x = self.conv_fc(x)
        x = x.view(batch_size, time_step, -1)
        en_outputs, en_hidden  = self.encoder(x)
        
        outputs = []
        dec_input = en_outputs[:, -1:, :]
        dec_hidden = en_hidden
        for _ in range(3):
            dec_input, dec_hidden = self.decoder(dec_input, dec_hidden)
            #print(dec_input[:, 0, :].shape)
            output = self.decoder_fc(dec_input[:, 0, :])
            outputs.append(output.view(batch_size, 1, 25))
            
        return torch.cat(outputs, dim=1)


def train(model, loss_func, optimizer, x_val, y_val):
    model.train()
    x = Variable(x_val, requires_grad=False)
    y = Variable(y_val, requires_grad=False)
    if torch.cuda.is_available():
        x = x.cuda()
        y = y.cuda()
    optimizer.zero_grad()
    output = model(x)
    output = loss_func(output, y)
    output.backward()
    optimizer.step()

    
def predict(model, x_val):
    model.eval()
    x_val = Variable(x_val, requires_grad=False)
    if torch.cuda.is_available():
        x_val = x_val.cuda()
    output = model(x_val)
    return output


def cal_loss(model, x_val, y_val):
    y_val = Variable(y_val, requires_grad=False)
    if torch.cuda.is_available():
        y_val = y_val.cuda()
    loss_func = nn.L1Loss()
    pred = predict(model, x_val)
    loss = loss_func(pred, y_val)
    
    return loss.cpu().data.numpy()