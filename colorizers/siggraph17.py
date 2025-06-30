import os
import torch
import torch.nn as nn
from .base_color import BaseColor

class SIGGRAPHGenerator(BaseColor):
    def inference(self, input_l):
        # Generate dummy ab channels and mask for inference
        B, _, H, W = input_l.shape
        input_ab = torch.zeros(B, 2, H, W, device=input_l.device)
        mask_ab = torch.zeros(B, 1, H, W, device=input_l.device)
        return self.forward(input_l, input_ab, mask_ab)


    def __init__(self, norm_layer=nn.BatchNorm2d, classes=529):
        super().__init__()

        def conv(in_c, out_c, **kwargs):
            return nn.Conv2d(in_c, out_c, bias=True, **kwargs)

        def deconv(in_c, out_c):
            return nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=True)

        def relu(): return nn.ReLU(inplace=True)
        def norm(c): return norm_layer(c)

        # Encoder
        self.model1 = nn.Sequential(conv(4, 64, kernel_size=3, padding=1), relu(), conv(64, 64, kernel_size=3, padding=1), relu(), norm(64))
        self.model2 = nn.Sequential(conv(64, 128, kernel_size=3, padding=1), relu(), conv(128, 128, kernel_size=3, padding=1), relu(), norm(128))
        self.model3 = nn.Sequential(conv(128, 256, kernel_size=3, padding=1), relu(), conv(256, 256, kernel_size=3, padding=1), relu(), conv(256, 256, kernel_size=3, padding=1), relu(), norm(256))
        self.model4 = nn.Sequential(conv(256, 512, kernel_size=3, padding=1), relu(), conv(512, 512, kernel_size=3, padding=1), relu(), conv(512, 512, kernel_size=3, padding=1), relu(), norm(512))
        self.model5 = nn.Sequential(conv(512, 512, kernel_size=3, padding=2, dilation=2), relu(), conv(512, 512, kernel_size=3, padding=2, dilation=2), relu(), conv(512, 512, kernel_size=3, padding=2, dilation=2), relu(), norm(512))
        self.model6 = self.model5
        self.model7 = nn.Sequential(conv(512, 512, kernel_size=3, padding=1), relu(), conv(512, 512, kernel_size=3, padding=1), relu(), conv(512, 512, kernel_size=3, padding=1), relu(), norm(512))

        # Decoder
        self.model8up = nn.Sequential(deconv(512, 256))
        self.model3short8 = nn.Sequential(conv(256, 256, kernel_size=3, padding=1))
        self.model8 = nn.Sequential(relu(), conv(256, 256, kernel_size=3, padding=1), relu(), conv(256, 256, kernel_size=3, padding=1), relu(), norm(256))
        self.model9up = nn.Sequential(deconv(256, 128))
        self.model2short9 = nn.Sequential(conv(128, 128, kernel_size=3, padding=1))
        self.model9 = nn.Sequential(relu(), conv(128, 128, kernel_size=3, padding=1), relu(), norm(128))
        self.model10up = nn.Sequential(deconv(128, 128))
        self.model1short10 = nn.Sequential(conv(64, 128, kernel_size=3, padding=1))
        self.model10 = nn.Sequential(relu(), conv(128, 128, kernel_size=3, padding=1), nn.LeakyReLU(0.2))

        self.model_out = nn.Sequential(nn.Conv2d(128, 2, kernel_size=1), nn.Tanh())

    def forward(self, input_A, input_B=None, mask_B=None):
        if input_B is None:
            input_B = torch.zeros_like(input_A).repeat(1, 2, 1, 1)
        if mask_B is None:
            mask_B = torch.zeros_like(input_A)

        x = torch.cat([self.normalize_l(input_A), self.normalize_ab(input_B), mask_B], dim=1)

        conv1 = self.model1(x)
        conv2 = self.model2(conv1[:, :, ::2, ::2])
        conv3 = self.model3(conv2[:, :, ::2, ::2])
        conv4 = self.model4(conv3[:, :, ::2, ::2])
        conv5 = self.model5(conv4)
        conv6 = self.model6(conv5)
        conv7 = self.model7(conv6)

        conv8 = self.model8(self.model8up(conv7) + self.model3short8(conv3))
        conv9 = self.model9(self.model9up(conv8) + self.model2short9(conv2))
        conv10 = self.model10(self.model10up(conv9) + self.model1short10(conv1))

        out = self.model_out(conv10)
        return self.unnormalize_ab(out)

def siggraph17(pretrained=True):
    model = SIGGRAPHGenerator()
    if(pretrained):
        import torch.utils.model_zoo as model_zoo
        model.load_state_dict(model_zoo.load_url('https://colorizers.s3.us-east-2.amazonaws.com/siggraph17-df00044c.pth',map_location='cpu',check_hash=True))
    return model
