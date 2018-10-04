from pathlib import Path

import torch
import torch.nn.init
import torch.utils.model_zoo as model_zoo
import wget
from torch import nn

from utils.convlstm import ConvLSTM


class PolygonNet(nn.Module):

    def __init__(self, load_vgg=True):
        super(PolygonNet, self).__init__()

        def _make_basic(input_size, output_size, kernel_size, stride, padding):
            """

            :rtype: nn.Sequential
            """
            return nn.Sequential(
                nn.Conv2d(input_size, output_size, kernel_size, stride,
                          padding),
                nn.ReLU(),
                nn.BatchNorm2d(output_size)
            )

        self.model1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.model2 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.model3 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.model4 = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.convlayer1 = _make_basic(128, 128, 3, 1, 1)
        self.convlayer2 = _make_basic(256, 128, 3, 1, 1)
        self.convlayer3 = _make_basic(512, 128, 3, 1, 1)
        self.convlayer4 = _make_basic(512, 128, 3, 1, 1)
        self.convlayer5 = _make_basic(512, 128, 3, 1, 1)
        self.poollayer = nn.MaxPool2d(2, 2).cuda()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear').cuda()
        self.convlstm = ConvLSTM(input_size=(28, 28),
                                 input_dim=131,
                                 hidden_dim=[32, 8],
                                 kernel_size=(3, 3),
                                 num_layers=2,
                                 batch_first=True,
                                 bias=True,
                                 return_all_layers=True)
        self.lstmlayer = nn.LSTM(28 * 28 * 8 + (28 * 28 + 3) * 2, 28 * 28 * 2,
                                 batch_first=True)
        self.linear = nn.Linear(28 * 28 * 2, 28 * 28 + 3)
        self.init_weights(load_vgg=load_vgg)

    def init_weights(self, load_vgg=True):
        '''
        Initialize weights of PolygonNet
        :param load_vgg: bool
                    load pretrained vgg model or not
        '''

        for name, param in self.convlstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)
        for name, param in self.lstmlayer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 1.0)
            elif 'weight' in name:
                # nn.init.xavier_normal_(param)
                nn.init.orthogonal_(param)
        for name, param in self.named_parameters():
            if 'bias' in name and 'convlayer' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name and 'convlayer' in name and '0' in name:
                nn.init.xavier_normal_(param)
        vgg_file = Path('vgg16_bn-6c64b313.pth')
        if load_vgg:
            if vgg_file.is_file():
                vgg16_dict = torch.load('vgg16_bn-6c64b313.pth')
            else:
                try:
                    wget.download(
                        'https://download.pytorch.org/models/vgg16_bn'
                        '-6c64b313.pth')
                    vgg16_dict = torch.load('vgg16_bn-6c64b313.pth')
                except:
                    vgg16_dict = torch.load(model_zoo.load_url(
                        'https://download.pytorch.org/models/vgg16_bn'
                        '-6c64b313.pth'))
            vgg_name = []
            for name in vgg16_dict:
                if 'feature' in name and 'running' not in name:
                    vgg_name.append(name)
            cnt = 0
            # print([x[0] for x in self.named_parameters()])
            for name, param in self.named_parameters():
                if 'model' in name:
                    param.data.copy_(vgg16_dict[vgg_name[cnt]])
                    cnt += 1

    def forward(self, input_data1, first, second, third):
        bs = second.shape[0]
        length_s = second.shape[1]
        output1 = self.model1(input_data1)
        output11 = self.poollayer(output1)
        output11 = self.convlayer1(output11)
        output2 = self.model2(output1)
        output22 = self.convlayer2(output2)
        output3 = self.model3(output2)
        output33 = self.convlayer3(output3)
        output4 = self.model4(output3)
        output44 = self.convlayer4(output4)
        output44 = self.upsample(output44)
        output = torch.cat([output11, output22, output33, output44], dim=1)
        output = self.convlayer5(output)
        output = output.unsqueeze(1)
        output = output.repeat(1, length_s, 1, 1, 1)
        padding_f = torch.zeros([bs, 1, 1, 28, 28]).cuda()

        input_f = first[:, :-3].view(-1, 1, 28, 28).unsqueeze(1).repeat(1,
                                                                        length_s - 1,
                                                                        1, 1,
                                                                        1)
        input_f = torch.cat([padding_f, input_f], dim=1)
        input_s = second[:, :, :-3].view(-1, length_s, 1, 28, 28)
        input_t = third[:, :, :-3].view(-1, length_s, 1, 28, 28)
        output = torch.cat([output, input_f, input_s, input_t], dim=2)

        output = self.convlstm(output)[0][-1]

        shape_o = output.shape
        output = output.contiguous().view(bs, length_s, -1)
        output = torch.cat([output, second, third], dim=2)
        output = self.lstmlayer(output)[0]
        output = output.contiguous().view(bs * length_s, -1)
        output = self.linear(output)
        output = output.contiguous().view(bs, length_s, -1)

        return output

    def test(self, input_data1, len_s):
        bs = input_data1.shape[0]
        result = torch.zeros([bs, len_s]).cuda()
        output1 = self.model1(input_data1)
        output11 = self.poollayer(output1)
        output11 = self.convlayer1(output11)
        output2 = self.model2(output1)
        output22 = self.convlayer2(output2)
        output3 = self.model3(output2)
        output33 = self.convlayer3(output3)
        output4 = self.model4(output3)
        output44 = self.convlayer4(output4)
        output44 = self.upsample(output44)
        output = torch.cat([output11, output22, output33, output44], dim=1)
        feature = self.convlayer5(output)

        padding_f = torch.zeros([bs, 1, 1, 28, 28]).float().cuda()
        input_s = torch.zeros([bs, 1, 1, 28, 28]).float().cuda()
        input_t = torch.zeros([bs, 1, 1, 28, 28]).float().cuda()

        output = torch.cat([feature.unsqueeze(1), padding_f, input_s, input_t],
                           dim=2)

        output, hidden1 = self.convlstm(output)
        output = output[-1]
        output = output.contiguous().view(bs, 1, -1)
        second = torch.zeros([bs, 1, 28 * 28 + 3]).cuda()
        second[:, 0, 28 * 28 + 1] = 1
        third = torch.zeros([bs, 1, 28 * 28 + 3]).cuda()
        third[:, 0, 28 * 28 + 2] = 1
        output = torch.cat([output, second, third], dim=2)

        output, hidden2 = self.lstmlayer(output)
        output = output.contiguous().view(bs, -1)
        output = self.linear(output)
        output = output.contiguous().view(bs, 1, -1)
        output = (output == output.max(dim=2, keepdim=True)[0]).float()
        first = output
        result[:, 0] = (output.argmax(2))[:, 0]

        for i in range(len_s - 1):
            second = third
            third = output
            input_f = first[:, :, :-3].view(-1, 1, 1, 28, 28)
            input_s = second[:, :, :-3].view(-1, 1, 1, 28, 28)
            input_t = third[:, :, :-3].view(-1, 1, 1, 28, 28)
            input1 = torch.cat(
                [feature.unsqueeze(1), input_f, input_s, input_t], dim=2)
            output, hidden1 = self.convlstm(input1, hidden1)
            output = output[-1]
            output = output.contiguous().view(bs, 1, -1)
            output = torch.cat([output, second, third], dim=2)
            output, hidden2 = self.lstmlayer(output, hidden2)
            output = output.contiguous().view(bs, -1)
            output = self.linear(output)
            output = output.contiguous().view(bs, 1, -1)
            output = (output == output.max(dim=2, keepdim=True)[0]).float()
            result[:, i + 1] = (output.argmax(2))[:, 0]

        return result
