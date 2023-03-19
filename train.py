import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from utils import *
from loss import YoloLoss
import time
import sys
        

seed = 4567
torch.manual_seed(seed)

# Hyperparameters
learning_rate = 2e-5
device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 16
weight_decay = 0
epochs = 100
num_workers = 6
pin_memory = True
load_model = False
load_model_file = "Yolov1_100images_optim_mAP_0_90.path.tar"
img_dir = 'data/images'
label_dir = 'data/labels'



# image resizing to 448*448 (see the paper)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor()])

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave = True)
    mean_loss = []
    model.train()

    for batch_idx, (x,y) in enumerate(loop):
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = loss_fn(out, y)
        ll = loss.item()
        mean_loss.append(ll)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update the progress bar
        loop.set_postfix(loss = loss.item())



def main(mAP_min_wanted=0.9):
    model = Yolov1(split_size = 7, num_boxes = 2, num_classes=20).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = YoloLoss()
    

    if load_model:
        load_checkpoint(torch.load(load_model_file), model, optimizer)

    train_dataset = VOCDataset('data/100examples.csv', transform=transform, img_dir=img_dir, label_dir=label_dir)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory,
                              shuffle=False, drop_last=True) # drop_last= True: if the last batch'size < batch_size, we'll ignore it
    
    for epoch in range(epochs):    
        print('epoch n°', epoch+1)
        pred_boxes, target_boxes, _ = get_bboxes(train_loader, model, iou_threshold=0.5, threshold=0.4, device=device)
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.5, box_format='midpoint')
        print(f'Train mAP: {mean_avg_prec}')
            
        train_fn(train_loader, model, optimizer, loss_fn)
        
        # to save the model
        if epoch == epochs-1 or mean_avg_prec > mAP_min_wanted: 
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename="TL_Yolov1.path.tar")
            time.sleep(10)
            break


if __name__== '__main__':
    main()

