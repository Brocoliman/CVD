import torch
import torchvision.transforms as transforms
import os
from model import CVDVAE
from dataset import get_image2image_dataloaders
from fast import make_image


rootdir = r'C:\Lucas\Project2024-2025\Code'
datadir = os.path.join(rootdir, 'dataset1')
modeldir = os.path.join(rootdir, 'saves', 'infer')
lutdir = os.path.join(rootdir, 'sim', 'lut')
imgdir = os.path.join(rootdir, 'view')

print(":a")
model = CVDVAE()
model.load_state_dict(torch.load(os.path.join(modeldir, 'model24.pth'), weights_only=True))
print("b")

dataloader = get_image2image_dataloaders(batch_size=16, shuffle=True, 
                                               input_dir=os.path.join(datadir, 'input', 'test'), 
                                               target_dir=os.path.join(datadir, 'target', 'test'))

toPIL = transforms.ToPILImage()

if __name__ == '__main__':
    # original images
    for inputs, targets in dataloader:
        input, target = inputs[0], targets[0]
        break

    # daltonize
    model.eval()
    dinput, _, _ = model(input[None])
    dinput = dinput.squeeze()

    # sim
    lv = 1
    sinput = make_image(lv, input, lutdir)
    starget = make_image(lv, target, lutdir)
    sdinput = make_image(lv, dinput, lutdir)

    
    toPIL(input).save(os.path.join(imgdir,'input.png'))
    toPIL(target).save(os.path.join(imgdir,'target.png'))
    toPIL(dinput).save(os.path.join(imgdir,'dinput.png'))
    toPIL(sdinput).save(os.path.join(imgdir,'sdinput.png'))
    toPIL(sinput).save(os.path.join(imgdir,'sinput.png'))
    toPIL(starget).save(os.path.join(imgdir,'starget.png'))




