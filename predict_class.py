import torch
import torch.nn as nn

class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(cin,cout,3,padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout,cout,3,padding=1), nn.BatchNorm2d(cout)
        )
        self.skip = nn.Conv2d(cin,cout,1) if cin!=cout else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x) + self.skip(x))

class GarmentResNet(nn.Module):
    def __init__(self):
        super().__init__()
        chs, ic = [16,32,64,128], 3
        layers = []
        for oc in chs:
            layers += [ResBlock(ic,oc), nn.MaxPool2d(2)]
            ic = oc
        self.features   = nn.Sequential(*layers)
        flat_dim        = 128*(256//16)*(256//16)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(flat_dim,128), nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128,3)
        )
    def forward(self, x):
        return self.classifier(self.features(x))

_model = GarmentResNet()
_model.load_state_dict(torch.load("predict_class_weights.pth", map_location="cpu"))
_model.eval()

def predict(images: torch.Tensor) -> torch.Tensor:
    """
    images: FloatTensor (B,3,256,256), values in [0,1]
    returns: LongTensor (B,1) in {0,1,2}
    """
    images = (images - 0.5)/0.5   # normalize to [-1,+1]
    with torch.no_grad():
        logits = _model(images)
        return logits.argmax(dim=1, keepdim=True)

if __name__=="__main__":
    with open("predict_class.py","w") as f:
        f.write(torch.__version__ + "\n")  # dummy to avoid empty file
    print("predict_class.py written.")