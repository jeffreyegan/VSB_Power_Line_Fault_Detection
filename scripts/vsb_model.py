import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class VSBNet(nn.Module):
    
    def __init__(self):
        super(VSBNet, self).__init__()
        
        self.conv0a = nn.Conv1d(in_channels=36, out_channels=64, kernel_size=4)
        self.conv0b = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4)
        #self.conv0c = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=4)

        self.mp0 = nn.MaxPool1d(2, padding=1)

        self.conv1a = nn.Conv1d(in_channels=64, out_channels=20, kernel_size=4)
        self.conv1b = nn.Conv1d(in_channels=20, out_channels=20, kernel_size=4)

        self.gap1 = nn.AvgPool1d(192)
        self.do1 = nn.Dropout(0.2)
        
        self.fc0 = nn.Linear(20,32)
        self.do2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(32,8)
        self.do3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(8,1)
        
        def init_weights(m):
            # Pytorch default for Conv1d defaults to kaiming 
            if type(m) == nn.Linear:
                # default Dense to glorot_uniform (xavier uniform)
                # Whereas Pytorch default for Linear is kaiming
                torch.nn.init.xavier_uniform_(m.weight)
                
        self.apply(init_weights)
        
        
    def forward(self,x):
        x = F.relu(self.conv0a(x))
        x = F.relu(self.conv0b(x))
        #x = F.relu(self.conv0c(x))

        x = self.mp0(x)

        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))

        x = self.gap1(x)
        x = self.do1(x)
        x = x.view(x.shape[0],-1)
        
        x = torch.tanh(self.fc0(x))
        x = self.do2(x)
        
        x = torch.tanh(self.fc1(x))
        x = self.do3(x)
        
        x = self.fc2(x) # Leave sigmoid for the loss function
            
        return x


    def train_model(self, data, epochs=30, lr=1e-3):

        self.criterion = torch.nn.BCEWithLogitsLoss() #pos_weight=torch.Tensor([1.0,1.2])) - pos_weight seems to work differently on latest pytorch
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.train()
        for t in tqdm(range(epochs)):
            #running_loss = 0.0
            for i, (X_batch, y_target_batch) in tqdm(enumerate(data.loader)):
                    
                # Forward pass: Compute predicted y by passing x to the model
                y_pred = self(X_batch.float())
                # y_pred = net(X.view(X.shape[0], 1, X.shape[1]))

                # Compute and print loss
                loss = self.criterion(y_pred, y_target_batch.float())

                # Zero gradients, perform a backward pass, and update the weights.
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.eval()
        y_pred = self(torch.Tensor(data.X_train).float()).detach()
        loss = self.criterion(y_pred, torch.Tensor(data.y_train).float()).detach()
        print(loss)
        # Number of target==0, no of target==1
        print((y_pred <= 0).sum().item(), (y_pred > 0).sum().item())

        return y_pred
        

    def evaluate_model(self, data):

        self.eval()
        y_pred_train = self(torch.Tensor(data.X_train).float()).detach()
        loss = self.criterion(y_pred_train, torch.Tensor(data.y_train).float()).detach()
        print(loss)
        # Number of target==0, no of target==1
        print((y_pred_train <= 0).sum().item(), (y_pred_train > 0).sum().item())

        self.eval()
        y_val_preds = self(torch.Tensor(data.X_val).float()).detach()
        loss_val = self.criterion(y_val_preds, torch.Tensor(data.y_val).float()).detach()
        print(loss_val)
        print((y_val_preds <= 0).sum().item(), (y_val_preds > 0).sum().item())

        return y_pred_train, y_val_preds
    

'''
VSBNet(
  (conv0a): Conv1d(36, 64, kernel_size=(4,), stride=(1,))
  (conv0b): Conv1d(64, 64, kernel_size=(4,), stride=(1,))
  (mp0): MaxPool1d(kernel_size=2, stride=2, padding=1, dilation=1, ceil_mode=False)
  (conv1a): Conv1d(64, 20, kernel_size=(4,), stride=(1,))
  (conv1b): Conv1d(20, 20, kernel_size=(4,), stride=(1,))
  (gap1): AvgPool1d(kernel_size=(192,), stride=(192,), padding=(0,))
  (do1): Dropout(p=0.2, inplace=False)
  (fc0): Linear(in_features=20, out_features=32, bias=True)
  (do2): Dropout(p=0.3, inplace=False)
  (fc1): Linear(in_features=32, out_features=8, bias=True)
  (do3): Dropout(p=0.3, inplace=False)
  (fc2): Linear(in_features=8, out_features=1, bias=True)
)
'''
