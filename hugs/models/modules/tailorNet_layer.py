from TailorNet import TailorNet
import torch


class TailorNet_Layer():
    def __init__(self):
        self.tailor_net = TailorNet()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.tailor_net.to(self.device)
        self.tailor_net.eval()

    def forward(self, input):
        with torch.no_grad():
            input = input.to(self.device)
            output = self.tailor_net(input)
        return output