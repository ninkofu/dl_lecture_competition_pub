import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.up(x)

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, n_coefficients):
        super(AttentionBlock, self).__init__()
        self.W_gate = nn.Sequential(
            nn.Conv2d(F_g, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, n_coefficients, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(n_coefficients)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(n_coefficients, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return skip_connection * psi

class AttentionUNet(nn.Module):
    def __init__(self, img_ch=3, output_ch=2):
        super(AttentionUNet, self).__init__()
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = ConvBlock(img_ch, 64)
        self.Conv2 = ConvBlock(64, 128)
        self.Conv3 = ConvBlock(128, 256)
        self.Conv4 = ConvBlock(256, 512)
        self.Conv5 = ConvBlock(512, 1024)

        self.Up5 = UpConv(1024, 512)
        self.Att5 = AttentionBlock(F_g=512, F_l=512, n_coefficients=256)
        self.UpConv5 = ConvBlock(1024, 512)

        self.Up4 = UpConv(512, 256)
        self.Att4 = AttentionBlock(F_g=256, F_l=256, n_coefficients=128)
        self.UpConv4 = ConvBlock(512, 256)

        self.Up3 = UpConv(256, 128)
        self.Att3 = AttentionBlock(F_g=128, F_l=128, n_coefficients=64)
        self.UpConv3 = ConvBlock(256, 128)

        self.Up2 = UpConv(128, 64)
        self.Att2 = AttentionBlock(F_g=64, F_l=64, n_coefficients=32)
        self.UpConv2 = ConvBlock(128, 64)

        self.Conv = nn.Conv2d(64, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        e1 = self.Conv1(x)
        e2 = self.MaxPool(e1)
        e2 = self.Conv2(e2)

        e3 = self.MaxPool(e2)
        e3 = self.Conv3(e3)

        e4 = self.MaxPool(e3)
        e4 = self.Conv4(e4)

        e5 = self.MaxPool(e4)
        e5 = self.Conv5(e5)

        d5 = self.Up5(e5)
        s4 = self.Att5(gate=d5, skip_connection=e4)
        d5 = torch.cat((s4, d5), dim=1)
        d5 = self.UpConv5(d5)

        d4 = self.Up4(d5)
        s3 = self.Att4(gate=d4, skip_connection=e3)
        d4 = torch.cat((s3, d4), dim=1)
        d4 = self.UpConv4(d4)

        d3 = self.Up3(d4)
        s2 = self.Att3(gate=d3, skip_connection=e2)
        d3 = torch.cat((s2, d3), dim=1)
        d3 = self.UpConv3(d3)

        d2 = self.Up2(d3)
        s1 = self.Att2(gate=d2, skip_connection=e1)
        d2 = torch.cat((s1, d2), dim=1)
        d2 = self.UpConv2(d2)

        out = self.Conv(d2)

        return [out]



# 他のモジュールや関数はそのまま残しておきます


# class EVFlowNet(nn.Module):
#     def __init__(self, args):
#         super(EVFlowNet, self).__init__()
#         self._args = args

#         self.encoder1 = general_conv2d(in_channels=4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
#         self.encoder2 = general_conv2d(in_channels=_BASE_CHANNELS, out_channels=2 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
#         self.encoder3 = general_conv2d(in_channels=2 * _BASE_CHANNELS, out_channels=4 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
#         self.encoder4 = general_conv2d(in_channels=4 * _BASE_CHANNELS, out_channels=8 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

#         self.resnet_block = nn.Sequential(*[build_resnet_block(8 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(2)])

#         self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16 * _BASE_CHANNELS,
#                                                          out_channels=4 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
#         self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8 * _BASE_CHANNELS + 2,
#                                                          out_channels=2 * _BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
#         self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4 * _BASE_CHANNELS + 2,
#                                                          out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
#         self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2 * _BASE_CHANNELS + 2,
#                                                          out_channels=int(_BASE_CHANNELS / 2), do_batch_norm=not self._args.no_batch_norm)

#     def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
#         # encoder
#         skip_connections = {}
#         inputs = self.encoder1(inputs)
#         skip_connections['skip0'] = inputs.clone()
#         inputs = self.encoder2(inputs)
#         skip_connections['skip1'] = inputs.clone()
#         inputs = self.encoder3(inputs)
#         skip_connections['skip2'] = inputs.clone()
#         inputs = self.encoder4(inputs)
#         skip_connections['skip3'] = inputs.clone()

#         # transition
#         inputs = self.resnet_block(inputs)

#         # decoder
#         flow_dict = {}
#         intermediate_outputs = []

#         inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
#         inputs, flow = self.decoder1(inputs)
#         flow_dict['flow0'] = flow.clone()
#         intermediate_outputs.append(flow)

#         inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
#         inputs, flow = self.decoder2(inputs)
#         flow_dict['flow1'] = flow.clone()
#         intermediate_outputs.append(flow)

#         inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
#         inputs, flow = self.decoder3(inputs)
#         flow_dict['flow2'] = flow.clone()
#         intermediate_outputs.append(flow)

#         inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
#         inputs, flow = self.decoder4(inputs)
#         flow_dict['flow3'] = flow.clone()
#         intermediate_outputs.append(flow)

#         return flow_dict, intermediate_outputs

# import torch
# from torch import nn
# from src.models.base import *
# from typing import Dict, Any

# _BASE_CHANNELS = 64

# class EVFlowNet(nn.Module):
#     def __init__(self, args):
#         super(EVFlowNet,self).__init__()
#         self._args = args

#         self.encoder1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
#         self.encoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
#         self.encoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
#         self.encoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

#         self.resnet_block = nn.Sequential(*[build_resnet_block(8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(2)])

#         self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
#                         out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

#         self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
#                         out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

#         self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
#                         out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

#         self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
#                         out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)

#     def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
#         # encoder
#         skip_connections = {}
#         inputs = self.encoder1(inputs)
#         skip_connections['skip0'] = inputs.clone()
#         inputs = self.encoder2(inputs)
#         skip_connections['skip1'] = inputs.clone()
#         inputs = self.encoder3(inputs)
#         skip_connections['skip2'] = inputs.clone()
#         inputs = self.encoder4(inputs)
#         skip_connections['skip3'] = inputs.clone()

#         # transition
#         inputs = self.resnet_block(inputs)

#         # decoder
#         flow_dict = {}
#         inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
#         inputs, flow = self.decoder1(inputs)
#         flow_dict['flow0'] = flow.clone()

#         inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
#         inputs, flow = self.decoder2(inputs)
#         flow_dict['flow1'] = flow.clone()

#         inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
#         inputs, flow = self.decoder3(inputs)
#         flow_dict['flow2'] = flow.clone()

#         inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
#         inputs, flow = self.decoder4(inputs)
#         flow_dict['flow3'] = flow.clone()

#         return flow
        

# if __name__ == "__main__":
#     from config import configs
#     import time
#     from data_loader import EventData
#     '''
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     input_ = torch.rand(8,4,256,256).cuda()
#     a = time.time()
#     output = model(input_)
#     b = time.time()
#     print(b-a)
#     print(output['flow0'].shape, output['flow1'].shape, output['flow2'].shape, output['flow3'].shape)
#     #print(model.state_dict().keys())
#     #print(model)
#     '''
#     import numpy as np
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     EventDataset = EventData(args.data_path, 'train')
#     EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)
#     #model = nn.DataParallel(model)
#     #model.load_state_dict(torch.load(args.load_path+'/model18'))
#     for input_, _, _, _ in EventDataLoader:
#         input_ = input_.cuda()
#         a = time.time()
#         (model(input_))
#         b = time.time()
#         print(b-a)