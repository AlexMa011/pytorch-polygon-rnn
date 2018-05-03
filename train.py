from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.utils.data
from tensorboardX import SummaryWriter
import argparse
from data import load_data
from model import PolygonNet

parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument('--gpu_id', nargs='+',type=int)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--pretrained', type=str, default="False")
parser.add_argument('--num', type=int, default=45000)
parser.add_argument('--lr', type=float, default=0.0001)
args = parser.parse_args()

devices=args.gpu_id
print(devices)
batch_size = args.batch_size
num = args.num
lr = args.lr

torch.cuda.set_device(devices[0])
Dataloader = load_data(num, 'train', 60, batch_size)
len_dl = len(Dataloader)
print(len_dl)

net = PolygonNet()
net=nn.DataParallel(net,device_ids=devices)

if args.pretrained == "True":
    net.load_state_dict(torch.load('save/model'+'_'+str(num)+'.pth'))
net.cuda()
print('finished')

loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4000,100000], gamma=0.1)
writer = SummaryWriter()

dtype = torch.cuda.FloatTensor
dtype_t = torch.cuda.LongTensor

epoch_r = int(300000/len_dl)
for epoch in range(epoch_r):
    for step, data in enumerate(Dataloader):
        scheduler.step()
        x = Variable(data[0].type(dtype))
        x1 = Variable(data[1].type(dtype))
        x2 = Variable(data[2].type(dtype))
        x3 = Variable(data[3].type(dtype))
        ta = Variable(data[4].type(dtype_t))
        optimizer.zero_grad()
        
        r = net(x,x1,x2,x3)
        
        result = r.contiguous().view(-1,28*28+3)
        target = ta.contiguous().view(-1)

        
        loss = loss_function(result,target)

        loss.backward()
        
        result_index = torch.argmax(result,1)
        correct = (target==result_index).type(dtype).sum().item()
        acc = correct*1.0/target.shape[0]
        
#        scheduler.step(loss)
        optimizer.step()
        
        writer.add_scalar('data/loss',loss,epoch*len_dl+step)
        writer.add_scalar('data/accuracy',acc,epoch*len_dl+step)
        writer.add_scalar('data/correct',correct,epoch*len_dl+step)
        
        if step%100==0:
            torch.save(net.state_dict(),'save/model'+'_'+str(num)+'.pth')
            for param_group in optimizer.param_groups:
                print('epoch{} step{}:{}'.format(epoch,step,param_group['lr']))
        
    if epoch%5==0 and len_dl>200:
        torch.save(net.state_dict(),'save/'+str(epoch)+'_'+str(num)+'.pth')
        

writer.export_scalars_to_json("./all_scalars.json")
writer.close()



