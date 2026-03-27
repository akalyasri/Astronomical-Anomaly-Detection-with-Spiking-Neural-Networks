import torch
import torch.nn as nn
import snntorch as snn

class SimpleSNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(1, 10)
        self.lif1 = snn.Leaky(beta=0.9)

        self.fc2 = nn.Linear(10, 1)
        self.lif2 = snn.Leaky(beta=0.9)

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        outputs = []

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            outputs.append(spk2)

        return torch.stack(outputs)