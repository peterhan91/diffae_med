import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from templates import *
from templates_cls import *
from experiment_classifier import ClsModel

class PatchDataset(Dataset):
    def __init__(self, path_to_images, fold='test', sample=0, transform=None):
        self.transform = transform
        self.path_to_images = path_to_images
        self.df = pd.read_csv('labels/uka_chest.csv')
        self.df = self.df[self.df['split'] == fold]
        if(sample > 0 and sample < len(self.df)):
            self.df = self.df.sample(n=sample, random_state=42)
        self.df = self.df.set_index('Anforderungsnummer')
        self.PRED_LABEL = self.df.columns[:-1]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = Image.open(
            os.path.join(self.path_to_images, str(self.df.index[idx])+'.jpg')
            )
        image = image.convert('RGB')
        label = np.zeros(len(self.PRED_LABEL), dtype=int)
        for i in range(0, len(self.PRED_LABEL)):
            if(self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int') > 0):
                label[i] = self.df[self.PRED_LABEL[i].strip()].iloc[idx].astype('int')
        if self.transform:
            image = self.transform(image)

        return (image, label, str(self.df.index[idx]))
    

if __name__ == '__main__':
    device = 'cuda:2'
    conf = padchest256_autoenc()
    model = LitModel(conf)
    state = torch.load(f'checkpoints/{conf.name}/last.ckpt', map_location='cpu')
    model.load_state_dict(state['state_dict'], strict=False)
    model.ema_model.eval()
    model.ema_model.to(device)

    cls_conf = ukachest256_autoenc_cls()
    cls_model = ClsModel(cls_conf)
    state = torch.load(f'checkpoints/{cls_conf.name}/last.ckpt',
                        map_location='cpu')
    print('latent step:', state['global_step'])
    cls_model.load_state_dict(state['state_dict'], strict=False)
    cls_model.to(device)

    te_dataset = PatchDataset(path_to_images='/data/UKA_CHEST/UKA_preprocessed_all/',
                        fold='test',
                        sample = 6,
                        transform=transforms.Compose([
                        transforms.Resize(256),
                        transforms.CenterCrop(256),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                        ]))
    
    for (batch, y, name) in tqdm(te_dataset):
        batch = batch[None]
        cond = model.encode(batch.to(device))
        xT = model.encode_stochastic(batch.to(device), cond, T=500)
        for cls_id, label in enumerate(te_dataset.PRED_LABEL):
            cond2 = cls_model.normalize(cond)
            fig, ax = plt.subplots(1, 5)
            for i, alpha in enumerate([-0.5, -0.3, 0.0, 0.3, 0.5]):
                cond_ = cond2 + alpha * math.sqrt(512) * F.normalize(cls_model.classifier.weight[cls_id][None, :], dim=1)
                cond_ = cls_model.denormalize(cond_)
                img = model.render(xT, cond_, T=1000)
                _ = ax[i].imshow(img[0].permute(1, 2, 0).cpu())
                _ = ax[i].axis('off')
                _ = ax[i].set_title(str(alpha) + ' ' + label, fontsize=4)
            _ = plt.subplots_adjust(wspace=0, hspace=0)
            save_path = os.path.join('../uka_chest/diffae/', name)
            Path(save_path).mkdir(parents=True, exist_ok=True)
            _ = plt.savefig(os.path.join(save_path, name+'_'+label+'.png'), dpi=500, 
                            bbox_inches = 'tight', pad_inches = 0)
            plt.close()