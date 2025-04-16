import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.waveformer_utils import optimizer, loss
from Models.waveformer import Waveformer

class Training:
    def __init__(self, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 8
        self.fix_lr_epochs = 50

        self.train_mix = None
        self.test_mix = None
        self.val_mix = None

        self.train_labels = None
        self.test_labels = None
        self.val_labels = None

        self.train_gt = None
        self.test_gt = None
        self.val_gt = None

        self.model = model
        
    def get_files(self, directory):
        all_files = sorted(os.listdir(directory))
        files = []
        for file in all_files:
            if os.path.isfile(os.path.join(directory, file)):
                full_path = os.path.join(directory, file)
                full_path = full_path.replace("\\", "/")
                files.append(full_path)

        if os.path.isfile(os.path.join(directory, 'mysoundscape.jams')):
            files.remove(os.path.join(directory, 'mysoundscape.jams'))

        if os.path.isfile(os.path.join(directory, 'mysoundscape.txt')):
            files.remove(os.path.join(directory, 'mysoundscape.txt'))

        files = sorted(files, key = lambda x: int(x.split('_')[-1].split('.')[0]))
            
        return files

    def load_preprocess_audio(self, files_list):
        waveform_list = []
        
        for filename in files_list:
            waveform, sample_rate = torchaudio.load(filename)
            resampler = T.Resample(orig_freq = sample_rate, new_freq = 44100)
            waveform = resampler(waveform)
            waveform_list.append(waveform)

        return torch.stack(waveform_list)

    def preprocess_labels(self, files_list):
        labels = []
        
        for filename in files_list:
            tensor = torch.load(filename)
            labels.append(tensor)

        return torch.stack(labels)
    
    def get_training_utils(self, model):
        num_epochs = 100
        criterion = loss
        opt = optimizer(model, lr=5e-4, weight_decay = 0.0)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, mode = "max", factor = 0.1, 
                                                        patience = 5, min_lr = 5e-6, 
                                                        threshold = 0.1, threshold_mode = "abs")

        return num_epochs, criterion, opt, lr_scheduler

    def load_checkpoint(self, filename='checkpoint.pth'):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        loss = checkpoint['loss']
        
        return epoch, model_state_dict, optimizer_state_dict, loss
    
    def train_test_val_split(self):
        mixture_files = self.get_files('')  # Add your dataset path here (Containing mixture of Target speakers + background)
        gt_files = self.get_files('')  # Add your dataset path here (Containing mixture of background)
        label_files = self.get_files('')  # Add your dataset path here (Containing one hot labels)

        train_mix_files = mixture_files[:3300]
        val_mix_files = mixture_files[3300:3500]
        test_mix_files = mixture_files[3500:]

        train_clean_files = gt_files[:3300]
        val_clean_files = gt_files[3300:3500]
        test_clean_files = gt_files[3500:]

        train_label_files = label_files[:3300]
        val_label_files = label_files[3300:3500]
        test_label_files = label_files[3500:]

        self.train_mix = self.load_preprocess_audio(train_mix_files)
        self.test_mix = self.load_preprocess_audio(test_mix_files)
        self.val_mix = self.load_preprocess_audio(val_mix_files)

        self.train_labels = self.preprocess_labels(train_label_files)
        self.test_labels = self.preprocess_labels(test_label_files)
        self.val_labels = self.preprocess_labels(val_label_files)

        self.train_gt = self.load_preprocess_audio(train_clean_files)
        self.test_gt = self.load_preprocess_audio(test_clean_files)
        self.val_gt = self.load_preprocess_audio(val_clean_files)

    def get_dataloaders(self):
        train_dataset = torch.utils.data.TensorDataset(self.train_mix, self.train_labels, self.train_gt)
        val_dataset = torch.utils.data.TensorDataset(self.val_mix, self.val_labels, self.val_gt)
        test_dataset = torch.utils.data.TensorDataset(self.test_mix, self.test_labels, self.test_gt)

        train_loader = DataLoader(dataset = train_dataset, batch_size = self.batch_size, shuffle = True)
        val_loader = DataLoader(dataset = val_dataset, batch_size = self.batch_size, shuffle = False)
        test_loader = DataLoader(dataset = test_dataset, batch_size = self.batch_size, shuffle = False)

        return train_loader, val_loader, test_loader

    def train_model(self):
        
        num_epochs, criterion, opt, lr_scheduler = self.get_training_utils(self.model)

        train_loader, val_loader, test_loader = self.get_dataloaders()

        for epoch in range(epoch, 150):
            self.model.train()
            total_loss = 0
            for batch_idx, (x, y, z) in enumerate(train_loader):
                tensor = x.to(self.device)
                one_hot = y.to(self.device)
                gt = z.to(self.device)

                opt.zero_grad()
                
                outputs, mask = self.model(tensor, one_hot)
                
                loss_value = criterion(outputs, gt)
                
                total_loss += loss_value.item()
            
                loss_value.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                
                opt.step()
            
            avg_train_loss = total_loss / len(train_loader)

            total_loss = 0
            
            self.model.eval()
            
            for batch_idx, (x, y, z) in enumerate(val_loader):
                tensor = x.to(self.device)
                one_hot = y.to(self.device)
                gt = z.to(self.device)
                
                output, mask = self.model(tensor, one_hot)
                loss = criterion(output, gt)
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(val_loader)
                
            if epoch >= self.fix_lr_epochs:
                lr_scheduler.step(avg_loss)
                
            print(f'Epoch [{epoch+1}/{150}], Loss: {avg_train_loss:.4f}')

        print("Training complete.")

    def load_checkpoint(self, filename = "checkpoint.pth"):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        loss = checkpoint['loss']
        
        return epoch, model_state_dict, optimizer_state_dict, loss
    
    def save_checkpoint(self, epoch, model, optimizer, loss_value, filename = "checkpoint.pth"):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_value,
        }, filename)
