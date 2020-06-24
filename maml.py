import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
import glob
from tqdm import tqdm
import numpy as np
from collections import OrderedDict
from PIL import Image
#此为卷积块，这个部分就是图像中的卷积，内容是一次卷积，一次BN，然后是激活和池化
def ConvBlock(in_ch,out_ch):
    return nn.Sequential(nn.Conv2d(in_ch,out_ch,3,padding=1),
                         nn.BatchNorm2d(out_ch),
                         nn.ReLU(),
                         nn.MaxPool2d(kernel_size=2,stride=2))
#此为function forward卷积块，因为maml中存在的是新参数计算的loss对原参数的倒数，因此需要使用这个过程
#nn.functional中的卷积需要自己传入w和b，因此自定义时需要此方式而不能直接nn.conv2d
def ConvBlockFunction(x,w,b,w_bn,b_bn):
    x = F.conv2d(x,w,b,padding=1)
    x = F.batch_norm(x,running_mean=None,running_var=None,weight=w_bn,bias=b_bn,training=True)
    x = F.relu(x)
    x = F.max_pool2d(x,kernel_size = 2,stride=2)
    return x
#定义分类器，此为maml的内循环，这个模型可以进行更换
class Classifier(nn.Module):
    #定义分类器结构，四个卷积层带有bn，最后接一个全连接层
    def __init__(self,in_ch,k_way):
        super(Classifier,self).__init__()
        self.conv1 = ConvBlock(in_ch,64)
        self.conv2 = ConvBlock(64,64)
        self.conv3 = ConvBlock(64,64)
        self.conv4 = ConvBlock(64,64)
        self.logits = nn.Linear(64,k_way)
    #此为常规的前向传播
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        x = x.view(x.shape[0],-1)
        x = self.logits(x)
        return x
    def functional_forward(self,x,params):
        #此处params即模型的参数，是一个ordereddict
        for block in [1, 2, 3, 4]:
            x = ConvBlockFunction(x, params[f'conv{block}.0.weight'], params[f'conv{block}.0.bias'],
                                  params.get(f'conv{block}.1.weight'), params.get(f'conv{block}.1.bias'))
        x = x.view(x.shape[0], -1)
        x = F.linear(x, params['logits.weight'], params['logits.bias'])
        return x

#此方法用来产生每次的标签，例如5way，2shot，产生的是【0，0，1，1，2，2，3，3，4，4】
def create_label(n_way, k_shot):
    return torch.arange(n_way).repeat_interleave(k_shot).long()

def MAML(model,optimizer,x,n_way,k_shot,q_query,loss_fn,inner_train_step=1,inner_lr=0.4,train=True):
    #x就是图片，在每一个meta step中的输入，size是【batch_size,n_way*(k_shot+q_query),1,28,28]
    #batchsize可以理解为每次同事做的任务数，而后面的n_way是任务中有几类，（k_shot）是每类有几张图片，omiglot中图片是单通道，28*28大小
    #q_query是每次testing有多少张图片用来更新
    criterion=loss_fn
    task_loss = []#放入每个task的loss
    task_acc = []#放入每个task的精确度
    for meta_batch in x:
        train_set = meta_batch[:n_way*k_shot]#此为训练时使用的数据（更新内部参数)
        val_set = meta_batch[n_way*k_shot:]#此为验证测试时使用的数据(更新maml参数)
        fast_weights = OrderedDict(model.named_parameters())#此为存储初始参数，即maml参数
        for inner_step in range(inner_train_step):
            train_label = create_label(n_way,k_shot).cuda()
            logits=model.functional_forward(train_set,fast_weights) #此为预测结果
            loss = criterion(logits,train_label)    #此为loss
            grads=torch.autograd.grad(loss,fast_weights.values(),create_graph=True)
            fast_weights = OrderedDict((name, param - inner_lr * grad)
                                       for ((name, param), grad) in zip(fast_weights.items(), grads))#此处更新内部的参数
        val_label= create_label(n_way,q_query).cuda()
        logits=model.functional_forward(val_set,fast_weights)#此处用之前更新过的weight来计算
        loss = criterion(logits,val_label)
        task_loss.append(loss)#将loss加入列表中
        acc = np.asarray([torch.argmax(logits, -1).cpu().numpy() == val_label.cpu().numpy()]).mean()  # 算 accuracy
        task_acc.append(acc)#将精确度加入列表
    model.train()
    optimizer.zero_grad()
    meta_batch_loss = torch.stack(task_loss).mean()#使用各任务loss的平均值作为meta loss
    if train:
        meta_batch_loss.backward()
        optimizer.step()
    task_acc=np.mean(task_acc)
    return meta_batch_loss,task_acc
#此方法为提取图片方法，每次调用会随机从一个character中抽取出k_shot+q_query张图
class Omniglot(Dataset):
    def __init__(self, data_dir, k_shot, q_query):
        self.file_list = [f for f in glob.glob(data_dir + "**/character*", recursive=True)]
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.n = k_shot + q_query
    def __getitem__(self, idx):
        sample = np.arange(20)
        np.random.shuffle(sample) # 随机出一个character
        img_path = self.file_list[idx]
        img_list = [f for f in glob.glob(img_path + "**/*.png", recursive=True)]
        img_list.sort()
        imgs = [self.transform(Image.open(img_file)) for img_file in img_list]
        imgs = torch.stack(imgs)[sample[:self.n]] #每个character取n个图
        return imgs
    def __len__(self):
        return len(self.file_list)
#此方法为大的batch，表示的时一次抽取多个任务
def get_meta_batch(meta_batch_size, k_shot, q_query, data_loader, iterator):
    data = []
    for _ in range(meta_batch_size):
        try:
            task_data = iterator.next()
        except StopIteration:
            iterator = iter(data_loader)
            task_data = iterator.next()
    train_data = task_data[:, :k_shot].reshape(-1, 1, 28, 28)
    val_data = task_data[:, k_shot:].reshape(-1, 1, 28, 28)
    task_data = torch.cat((train_data, val_data), 0)
    data.append(task_data)
    return torch.stack(data).cuda(), iterator
if __name__=='__main__':
    n_way = 5
    k_shot = 1
    q_query = 1
    inner_train_step = 1
    inner_lr = 0.4
    meta_lr = 0.001
    meta_batch_size = 32
    max_epoch = 40
    eval_batches = test_batches = 20
    train_data_path = './Omniglot/images_background/'
    test_data_path = './Omniglot/images_evaluation/'
    # dataset = Omniglot(train_data_path, k_shot, q_query)
    train_set, val_set = torch.utils.data.random_split(Omniglot(train_data_path, k_shot, q_query), [3200, 656])
    train_loader = DataLoader(train_set,
                              batch_size=n_way,  #此处的batchsize其实是一个任务中有多少个类别
                              num_workers=8,
                              shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_set,
                            batch_size=n_way,
                            num_workers=8,
                            shuffle=True,
                            drop_last=True)
    test_loader = DataLoader(Omniglot(test_data_path, k_shot, q_query),
                             batch_size=n_way,
                             num_workers=8,
                             shuffle=True,
                             drop_last=True)
    train_iter = iter(train_loader)
    val_iter = iter(val_loader)
    test_iter = iter(test_loader)
    meta_model = Classifier(1, n_way).cuda()#模型是我们刚刚定下的分类器
    optimizer = torch.optim.Adam(meta_model.parameters(), lr=meta_lr)#优化器采用adam
    loss_fn = nn.CrossEntropyLoss().cuda()#loss就是普通的交叉熵损失
    for epoch in range(max_epoch):
        print("Epoch %d" % (epoch))
        train_meta_loss = []
        train_acc = []
        for step in tqdm(range(len(train_loader) // (meta_batch_size))):  # 這裡的 step 是一次 meta-gradinet update step
            x, train_iter = get_meta_batch(meta_batch_size, k_shot, q_query, train_loader, train_iter)
            meta_loss, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn)
            train_meta_loss.append(meta_loss.item())
            train_acc.append(acc)
        print("  Loss    : ", np.mean(train_meta_loss))
        print("  Accuracy: ", np.mean(train_acc))
        val_acc = []
        for eval_step in tqdm(range(len(val_loader) // (eval_batches))):
            x, val_iter = get_meta_batch(eval_batches, k_shot, q_query, val_loader, val_iter)
            _, acc = MAML(meta_model, optimizer, x, n_way, k_shot, q_query, loss_fn, inner_train_step=3,
                          train=False)  # testing時，我們更新三次 inner-step
            val_acc.append(acc)
        print("  Validation accuracy: ", np.mean(val_acc))









