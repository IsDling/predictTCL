import torch
from torch import nn
from sup.resnet3d import resnet34
import copy
import numpy as np

class predictTCL(torch.nn.Module):
    def __init__(self, num_classes, dropout_value, run_type):
        super(predictTCL, self).__init__()
        self.num_classes = num_classes
        self.run_type = run_type
        resnet = resnet34(modal_num=1)
        backbone = list(resnet.children())
        dim_out = 512
        self.backbone_t1 = nn.Sequential(*backbone[:8])
        self.backbone_t2 = copy.deepcopy(self.backbone_t1)
        self.backbone_adc = copy.deepcopy(self.backbone_t1)

        self.concat_inv = nn.Conv3d(in_channels=512 * 3, out_channels=512, kernel_size=(2, 2, 2), stride=2,
                                           padding=(2, 2, 4))
        self.concat_share = nn.Conv3d(in_channels=512 * 3, out_channels=512, kernel_size=(2, 2, 2), stride=2,
                                    padding=(2, 2, 4))
        self.concat_men = nn.Conv3d(in_channels=512 * 3, out_channels=512, kernel_size=(2, 2, 2), stride=2,
                                           padding=(2, 2, 4))


        self.pooling = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.flatten = nn.Flatten(1)

        self.linear_spec_inv_1 = nn.Linear(dim_out, 128)
        self.linear_share_inv = nn.Linear(dim_out, 128)
        self.linear_share_men = nn.Linear(dim_out, 128)
        self.linear_spec_men_1 = nn.Linear(dim_out, 128)

        self.linear_spec_inv_2 = nn.Linear(256, 32)
        self.linear_spec_inv_3 = nn.Linear(32, num_classes)

        self.linear_spec_men_2 = nn.Linear(256, 32)
        self.linear_spec_men_3 = nn.Linear(32, num_classes)

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout_value)

        # if self.run_type == 'train':
        self.linear_sup_inv_1 = nn.Linear(dim_out, 256)
        self.linear_sup_inv_2 = nn.Linear(256, 32)
        self.linear_sup_inv_3 = nn.Linear(32, num_classes)

        self.linear_sup_men_1 = nn.Linear(dim_out, 256)
        self.linear_sup_men_2 = nn.Linear(256, 32)
        self.linear_sup_men_3 = nn.Linear(32, num_classes)

    def forward(self, x):

        t1_data = x[:, 0, ...]
        if t1_data.ndim == 4:
            t1_data = torch.unsqueeze(t1_data, 1)
        x_t1 = self.backbone_t1(t1_data)

        t2_data = x[:, 1, ...]
        if t2_data.ndim == 4:
            t2_data = torch.unsqueeze(t2_data, 1)
        x_t2 = self.backbone_t2(t2_data)

        adc_data = x[:, 2, ...]
        if adc_data.ndim == 4:
            adc_data = torch.unsqueeze(adc_data, 1)
        x_adc = self.backbone_adc(adc_data)

        x_concat = torch.cat((x_t1,x_t2,x_adc),1)
        x_conv_inv = self.concat_inv(x_concat)
        x_conv_share = self.concat_share(x_concat)
        x_conv_men = self.concat_men(x_concat)
        x_sm_spec_inv = self.pooling(x_conv_inv)
        x_sm_spec_inv = self.flatten(x_sm_spec_inv)
        x_sm_share = self.pooling(x_conv_share)
        x_sm_share = self.flatten(x_sm_share)
        x_sm_spec_men = self.pooling(x_conv_men)
        x_sm_spec_men = self.flatten(x_sm_spec_men)

        x_sm_spec_inv = self.relu(x_sm_spec_inv)
        x_sm_spec_inv = self.drop(x_sm_spec_inv)
        x_sm_share = self.relu(x_sm_share)
        x_sm_share = self.drop(x_sm_share)
        x_sm_spec_men = self.relu(x_sm_spec_men)
        x_sm_spec_men = self.drop(x_sm_spec_men)

        # main flow
        # con
        x_con_spec_inv = self.linear_spec_inv_1(x_sm_spec_inv)
        x_con_share_inv = self.linear_share_inv(x_sm_share)
        x_con_share_men = self.linear_share_men(x_sm_share)
        x_con_spec_men = self.linear_spec_men_1(x_sm_spec_men)
        # inv
        x_spec_share_inv_1 = torch.cat((x_con_spec_inv, x_con_share_inv), 1)
        x_spec_share_inv_1 = self.relu(x_spec_share_inv_1)
        x_spec_share_inv_1 = self.drop(x_spec_share_inv_1)
        x_spec_share_inv_2 = self.linear_spec_inv_2(x_spec_share_inv_1)
        x_spec_share_inv_2 = self.relu(x_spec_share_inv_2)
        x_spec_share_inv_2 = self.drop(x_spec_share_inv_2)
        main_out_inv = self.linear_spec_inv_3(x_spec_share_inv_2)
        # men
        x_spec_share_men_1 = torch.cat((x_con_spec_men, x_con_share_men), 1)
        x_spec_share_men_1 = self.relu(x_spec_share_men_1)
        x_spec_share_men_1 = self.drop(x_spec_share_men_1)
        x_spec_share_men_2 = self.linear_spec_men_2(x_spec_share_men_1)
        x_spec_share_men_2 = self.relu(x_spec_share_men_2)
        x_spec_share_men_2 = self.drop(x_spec_share_men_2)
        main_out_men = self.linear_spec_men_3(x_spec_share_men_2)

        # sup flow
        # inv
        x_sup_inv_1 = self.linear_sup_inv_1(x_sm_spec_inv)
        x_sup_inv_1 = self.relu(x_sup_inv_1)
        x_sup_inv_1 = self.drop(x_sup_inv_1)
        x_sup_inv_2 = self.linear_sup_inv_2(x_sup_inv_1)
        x_sup_inv_2 = self.relu(x_sup_inv_2)
        x_sup_inv_2 = self.drop(x_sup_inv_2)
        sup_out_inv = self.linear_sup_inv_3(x_sup_inv_2)
        # men
        x_sup_men_1 = self.linear_sup_men_1(x_sm_spec_men)
        x_sup_men_1 = self.relu(x_sup_men_1)
        x_sup_men_1 = self.drop(x_sup_men_1)
        x_sup_men_2 = self.linear_sup_men_2(x_sup_men_1)
        x_sup_men_2 = self.relu(x_sup_men_2)
        x_sup_men_2 = self.drop(x_sup_men_2)
        sup_out_men = self.linear_sup_men_3(x_sup_men_2)


        if self.run_type == 'train':
             return x_con_share_inv, x_con_share_men, x_con_spec_inv, x_con_spec_men, sup_out_inv, sup_out_men, main_out_inv, main_out_men
        elif self.run_type == 'test':
            return main_out_inv, main_out_men


if __name__ == '__main__':
    model = predictTCL(num_classes=2,dropout_value=0.5, run_type='train')
    print(model)
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    print('num of pare:', sum([np.prod(p.size()) for p in model_parameters]))
    # t1 = torch.rand(size=(64, 1, 128, 128, 24))
    # t2 = torch.rand(size=(64, 1, 128, 128, 24))
    # adc = torch.rand(size=(64, 1, 128, 128, 24))
    all = torch.rand(size=(1, 3, 128, 128, 24))
    x_con_share_inv, x_con_share_men, x_con_spec_inv, x_con_spec_men, sup_out_inv, sup_out_men, main_out_inv, main_out_men = model(all)  # (2, 1000)
    print(main_out_men.shape)