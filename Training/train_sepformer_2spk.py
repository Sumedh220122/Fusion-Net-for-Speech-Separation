import os
import torch
import torchaudio
import torchaudio.transforms as T
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.waveformer_utils import optimizer, loss
from Models.waveformer import Waveformer
from speechbrain.nnet.schedulers import ReduceLROnPlateau

class Training:
    def __init__(self, model):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 8
        self.fix_lr_epochs = 50

        self.train_mix = None
        self.test_mix = None
        self.val_mix = None

        self.train_spk1 = None
        self.test_spk1 = None
        self.val_spk1 = None

        self.train_spk2 = None
        self.test_spk2 = None
        self.val_spk2 = None

        self.model = model

        self.batch_size = 1
        
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
            resampler = T.Resample(orig_freq = sample_rate, new_freq = 8000)
            waveform = resampler(waveform)
            waveform_list.append(waveform)

        return torch.cat(waveform_list)

    
    def get_training_utils(self, model):
        num_epochs = 100
        criterion = loss
        opt = optimizer(model, lr = 0.00015, weight_decay = 0)
        scheduler = ReduceLROnPlateau(factor = 0.5, patience = 2, dont_halve_until_epoch = 85)

        return num_epochs, criterion, opt, scheduler

    def load_checkpoint(self, filename='checkpoint.pth'):
        checkpoint = torch.load(filename)
        epoch = checkpoint['epoch']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        loss = checkpoint['loss']
        
        return epoch, model_state_dict, optimizer_state_dict, loss
    
    def train_test_val_split(self):
        mixture_files = self.get_files('')  # Add your dataset path here (Containing mixture of 2 speakers)
        spk1_files = self.get_files('')  # Add your dataset path here (Containing mixture of speaker 1)
        spk2_files = self.get_files('')  # Add your dataset path here (Containing mixture of speaker 2)

        train_mix_files = mixture_files[:3300]
        val_mix_files = mixture_files[3300:3500]
        test_mix_files = mixture_files[3500:]

        train_spk1_files = spk1_files[:3300]
        val_spk1_files = spk1_files[3300:3500]
        test_spk1_files = spk1_files[3500:]

        train_spk2_files = spk2_files[:3300]
        val_spk2_files = spk2_files[3300:3500]
        test_spk2_files = spk2_files[3500:]

        self.train_mix = self.load_preprocess_audio(train_mix_files)
        self.test_mix = self.load_preprocess_audio(test_mix_files)
        self.val_mix = self.load_preprocess_audio(val_mix_files)

        self.train_spk1 = self.load_preprocess_audio(train_spk1_files)
        self.test_spk1 = self.load_preprocess_audio(test_spk1_files)
        self.val_spk1 = self.load_preprocess_audio(val_spk1_files)

        self.train_spk2 = self.load_preprocess_audio(train_spk2_files)
        self.test_spk2 = self.load_preprocess_audio(test_spk2_files)
        self.val_spk2 = self.load_preprocess_audio(val_spk2_files)

    def get_dataloaders(self):
        train_dataset = torch.utils.data.TensorDataset(self.train_mix, self.train_spk1, self.train_spk2)
        val_dataset = torch.utils.data.TensorDataset(self.val_mix, self.val_spk1, self.val_spk2)
        test_dataset = torch.utils.data.TensorDataset(self.test_mix, self.test_spk1, self.test_spk2)

        train_loader = DataLoader(dataset = train_dataset, batch_size = self.batch_size, shuffle = True)
        val_loader = DataLoader(dataset = val_dataset, batch_size = self.batch_size, shuffle = False)
        test_loader = DataLoader(dataset = test_dataset, batch_size = self.batch_size, shuffle = False)

        return train_loader, val_loader, test_loader

    def train_model(self):
        
        num_epochs, criterion, opt, scheduler = self.get_training_utils(self.model)

        train_loader, val_loader, test_loader = self.get_dataloaders()

        start_epoch = 0
        loss_value = 0

        for epoch in range(start_epoch, 100):
            self.model.train()
            epoch_loss = 0
            for batch_idx, (w, x, y) in enumerate(train_loader):
                waveform = w.to(self.device)
                spk1 = x.to(self.device)
                spk2 = y.to(self.device)
                
                opt.zero_grad() 
                
                with torch.cuda.amp.autocast():  
                    output = self.model(waveform)
                    loss_value = criterion(output[0], spk1) + criterion(output[1], spk2)
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), max_norm = 5)
                    epoch_loss += loss_value.item()

                loss_value.backward()
                opt.step()
                
        #scheduler.step(loss_value)

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_value.item():.4f}')

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
