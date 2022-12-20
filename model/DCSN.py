import torch
import pandas as pd
from data.data_access import Data
from torch.utils.data import Dataset, DataLoader
import numpy as np
from torch import nn
from sklearn.metrics import roc_auc_score
from sklearn import metrics
import math
ep_number = 1
epeach=1000
dropouts=[0.95]
momentums=[0.99]
weight_decays=[5e-3]
batchs=[20]
lrs=[1e-5]
first_channels=[2000]
second_channels=[1900]
return_number=5
return_spilit=60


def seed_everything(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    import os
    import random
    import numpy as np

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True


seed_everything(3407)




data_pa = {'id': 'ALL', 'type': 'prostate_paper',
           'params': {'data_type': ['mut_important', 'cnv_del', 'cnv_amp'], 'drop_AR': False, 'cnv_levels': 3,
                      'mut_binary': True, 'balanced_data': False, 'combine_type': 'union',
                      'use_coding_genes_only': True,
                      'selected_genes': 'tcga_prostate_expressed_genes_and_cancer_genes.csv', 'training_split': 0}}
data = Data(**data_pa)
x_train, x_validate_, x_test_, y_train, y_validate_, y_test_, info_train, info_validate_, info_test_, cols = data.get_train_validate_test()

x_t = np.concatenate((x_test_, x_validate_))
y_t = np.concatenate((y_test_, y_validate_))




def caculateAUC(AUC_outs, AUC_labels):
    ROC = 0
    outs = []
    labels = []
    for (index, AUC_out) in enumerate(AUC_outs):
        softmax = nn.Softmax(dim=1)
        out = softmax(AUC_out).numpy()
        out = out[:, 1]
        for out_one in out.tolist():
            outs.append(out_one)
        for AUC_one in AUC_labels[index].tolist():
            labels.append(AUC_one)

    outs = np.array(outs)

    labels = np.array(labels)

    fpr, tpr, thresholds = metrics.roc_curve(labels, outs, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    aupr = metrics.average_precision_score(labels, outs)

    return auc, aupr

print(x_train.shape)
print(x_t.shape)

class Mydata(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        gene = torch.from_numpy(self.x[index]).float()
        gene = gene.view(-1, 1)
        # gene=gene.float()

        return gene, int(self.y[index][0])

    def __len__(self):
        return len(self.x)


train_mydata = Mydata(x_train, y_train)
validate_mydata=Mydata(x_validate_,y_validate_)
test_mydata = Mydata(x_test_, y_test_)

for batch_size in batchs:
 for momentum in momentums:
  for weight_decay in weight_decays:
      for lr in lrs:
          for dropout in dropouts:
              for first_channel in first_channels:
                  for second_channel in second_channels:
                      class DCSN(nn.Module):

                          def __init__(self):
                              super(DCSN, self).__init__()
                              self.conv1 = nn.Conv1d(27687, first_channel, kernel_size=(1,), stride=1)
                              nn.init.xavier_normal_(self.conv1.weight, gain=1)

                              self.conv2 = nn.Conv1d(27687, second_channel, kernel_size=(1,), stride=1)
                              nn.init.xavier_normal_(self.conv2.weight, gain=1)

                              self.Flatten = nn.Flatten()

                              self.relu = nn.ReLU()
                              self.linear1 = nn.Linear(first_channel + second_channel, 2)
                              nn.init.xavier_normal_(self.linear1.weight, gain=1)

                              # self.linear3=nn.Linear(100,2)
                              self.BatchNorm1 = nn.BatchNorm1d(num_features=first_channel)

                              self.BatchNorm2 = nn.BatchNorm1d(num_features=second_channel)
                              self.dropout = nn.Dropout(dropout)


                          def forward(self, input):
                              # food_conv1 = self.maxpool1(self.dropout(self.BatchNorm(self.relu(self.food_conv1(input)))).squeeze(3))
                              conv1 = self.conv1(input)
                              conv1 = self.relu(conv1)
                              conv1 = self.BatchNorm1(conv1)
                              conv1 = self.dropout(conv1)

                              #

                              conv2 = self.conv2(input)
                              conv2 = self.relu(conv2)
                              conv2 = self.BatchNorm2(conv2)
                              conv2 = self.dropout(conv2)



                              all = torch.cat([conv1, conv2], 1).squeeze(2)



                              all = self.linear1(all)

                              return all


                      train_loader = DataLoader(dataset=train_mydata, batch_size=batch_size, shuffle=True)
                      validate_loader = DataLoader(dataset=validate_mydata, batch_size=batch_size, shuffle=True)
                      test_loader = DataLoader(dataset=test_mydata, batch_size=batch_size, shuffle=True)
                      average_accuracy=0
                      average_auc=0
                      average_auprc=0
                      for ep in range(ep_number):
                          model = DCSN()
                          pro_model = model.state_dict()
                          torch.save(model.state_dict(), 'fist_model' + str(ep) + '.pkl')
                          model = model.cuda()
                          best_accuracy = 0
                          optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': lr}], lr=lr,
                                                      momentum=momentum,
                                                      weight_decay=weight_decay)
                          train_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3000, gamma=0.1)
                          loss = nn.CrossEntropyLoss()
                          b_auc = {"acc": 0, "auc": 0, "aupr": 0}
                          b_acc = {"acc": 0, "auc": 0, "aupr": 0}
                          arr = []
                          ten_auc = []
                          ten_aupr = []
                          all = []
                          pro_acc = 0
                          n = 0
                          best_auc = 0
                          best_auc_model = 0
                          model_stack = []
                          model_stack_id = 0
                          return_number = 5

                          for i in range(epeach):

                              lossnumber3 = 0.0
                              lossnumber = 0.0
                              all_accuracy = 0
                              auc = []
                              lossnumber2 = 0.0
                              auc_label = []
                              auc_out = []
                              all_accuracy2 = 0
                              model.train()
                              for data in train_loader:
                                  input, label = data
                                  input = input.cuda()
                                  label = label.cuda()
                                  out = model(input)
                                  out = out.cuda()
                                  optimizer.zero_grad()
                                  result_loss = loss(out, label)

                                  result_loss = result_loss.cuda()
                                  result_loss.backward()
                                  optimizer.step()
                                  lossnumber3 = lossnumber3 + result_loss
                                  accuracy2 = (out.argmax(1) == label).sum()
                                  all_accuracy2 = all_accuracy2 + accuracy2


                              train_scheduler.step()

                              model.eval()
                              with torch.no_grad():
                                  for data in validate_loader:
                                      input, label = data
                                      input = input.cuda()
                                      label = label.cuda()
                                      out = model(input)
                                      out = out.cuda()
                                      result_loss = loss(out, label)
                                      lossnumber = lossnumber + result_loss
                                      accuracy = (out.argmax(1) == label).sum()
                                      out_list = []

                                      auc_label.append(label.cpu().numpy())
                                      auc_out.append(out.cpu())

                                      all_accuracy = all_accuracy + accuracy

                              auc_sum = 0.0

                              auc_number, aupr = caculateAUC(auc_out, auc_label)

                              arr.append(all_accuracy.item() / 204)
                              if auc_number > best_auc:
                                  best_accuracy = (all_accuracy.item() / 204)
                                  best_auc = auc_number

                                  torch.save(model.state_dict(), 'best.pkl')
                              if auc_number > best_auc_model:
                                  best_auc_model = auc_number


                                  model_stack_id = len(model_stack)
                                  torch.save(model.state_dict(), 'model_' + str(model_stack_id) + '.pkl')
                                  model_stack.append({'auc': auc_number, 'model': 'model_' + str(model_stack_id) + '.pkl',
                                                      'op': optimizer.state_dict()})
                                  n = i
                              ten_auc.append(auc_number)
                              ten_aupr.append(aupr)

                              if (i - n) == return_spilit:

                                  model_stack_id = model_stack_id - return_number
                                  if model_stack_id < 0:
                                      model_stack_id = 0
                                  model.load_state_dict(torch.load(model_stack[model_stack_id]['model']))

                                  best_auc_model = model_stack[model_stack_id]['auc']
                                  optimizer.load_state_dict(model_stack[model_stack_id]['op'])
                                  while len(model_stack) > (model_stack_id + 1):
                                      model_stack.pop(model_stack_id + 1)
                                  n = i

                          for i in range(100):
                               lossnumber3 = 0.0
                               lossnumber = 0.0
                               all_accuracy = 0
                               auc = []
                               lossnumber2 = 0.0
                               auc_label = []
                               auc_out = []
                               all_accuracy2 = 0
                               model.load_state_dict(torch.load('best.pkl'))
                               optimizer = torch.optim.SGD([{'params': model.parameters(), 'initial_lr': 1e-10}], lr=1e-10,
                                                           weight_decay=5e-3)
                               model.train()
                               for data in train_loader:
                                   input, label = data
                                   input = input.cuda()
                                   label = label.cuda()
                                   out = model(input)
                                   out = out.cuda()
                                   optimizer.zero_grad()

                                   result_loss = loss(out, label)

                                   result_loss = result_loss.cuda()
                                   result_loss.backward()
                                   optimizer.step()
                                   lossnumber3 = lossnumber3 + result_loss
                                   accuracy2 = (out.argmax(1) == label).sum()
                                   all_accuracy2 = all_accuracy2 + accuracy2


                               train_scheduler.step()

                          lossnumber3 = 0.0
                          lossnumber = 0.0
                          all_accuracy = 0
                          auc = []
                          lossnumber2 = 0.0
                          auc_label = []
                          auc_out = []
                          all_accuracy2 = 0
                          all_accuracy2 = 0

                          model.eval()

                          with torch.no_grad():

                              for data in test_loader:
                                  input, label = data
                                  input = input.cuda()
                                  label = label.cuda()
                                  out = model(input)
                                  out = out.cuda()
                                  result_loss = loss(out, label)
                                  lossnumber2 = lossnumber2 + result_loss
                                  accuracy = (out.argmax(1) == label).sum()
                                  out_list = []

                                  auc_label.append(label.cpu().numpy())
                                  auc_out.append(out.cpu())

                                  all_accuracy = all_accuracy + accuracy

                              auc_sum = 0.0

                              auc_number, aupr = caculateAUC(auc_out, auc_label)


                              average_accuracy=average_accuracy+(all_accuracy / 102)
                              average_auc=average_auc+auc_number
                              average_auprc=average_auprc+aupr
                      print('参数:first_channel:{},second_channel:{},dropout:{},weight_decay:{},momentum:{},batch_size:{},accuracy:{},auc:{},auprc:{}'.format(first_channel,second_channel,dropout,weight_decay,momentum,batch_size,average_accuracy/ep_number,average_auc/ep_number,average_auprc/ep_number))