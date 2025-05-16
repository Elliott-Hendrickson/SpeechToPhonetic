import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import librosa
import textgrid
import pickle
import matplotlib.pyplot as plt
import torchaudio

labelLexicon = {'f': 1, 'ɒ': 2, 'ɹ': 3, 'd̪': 4, 'ə':5, 'tʷ':6, 'ɛ':7, 'ɲ':8, 'i':9, 'θ':10, 'tʰ':11, 'aj':12, 'm':13, 'æ':14, 't':15, 'iː':16, 'v':17, 'ɪ':18, 'ŋ':19, 'ʉː':20, 'n':21, 'ʃ':22, 'ʊ':23, 'k':24, 'h':25, 'z':26, 'w':27, 'ɫ':28, 'ɚ':29, 'ɡ':30, 'ɑ':31, 'd':32, 'b':33, 'l':34, 's':35, 'ow':36, 'p':37, 'ɫ̩':38, 'j':39, 'ɑː':40, 'ej':41, 'ʉ':42, 'pʰ':43, 'ɝ':44, 'tʲ':45, 'dʲ':46, 'ɟ':47, 'cʰ':48, 'dʒ':49, 'ɐ':50, 'ʎ':51, 'aw':52, 'ç':53, 'tʃ':54, 'fʲ':55, 'kʰ':56, 'ɒː':57, 'bʲ':58, 'c':59, 'mʲ':60, 'pʲʲ':61, 'cʷ':62, 'vʲ':63, 't̪':64, 'ð':65, 'kʷ':66, 'ɾ':67, 'spn':68, 'ɔj':69, 'ʒ':70, 'ɾʲ':71, 'n̩':72, 'ɟʷ':73, 'm̩':74, 'ɾ̃':75}

class AudioSegmentDataset(Dataset):
    def __init__(self, audio_path, textgrid_path, sr=16000):
        self.audio_path = audio_path
        self.textgrid_path = textgrid_path
        self.sr = sr

        self.audio, self.actual_sr = librosa.load(audio_path, sr=sr)
        if self.actual_sr != sr:
            print(f"Resampled audio from {self.actual_sr} to {sr} Hz.")

        tg = textgrid.TextGrid.fromFile(textgrid_path)
        self.segments = []
        phones_tier = None

        for tier in tg.tiers:
            if tier.name.lower() == "phones":
                phones_tier = tier
                break

        if phones_tier is None:
            raise ValueError("No tier named 'phones' found in the TextGrid file.")

        # Extract intervals that have non-empty labels
        for interval in phones_tier:
            if interval.mark.strip():
                self.segments.append((interval.minTime, interval.maxTime, interval.mark.strip()))

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        start_time, end_time, label = self.segments[idx]
        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)
        audio_segment = self.audio[start_sample:end_sample]
        audio_tensor = torch.tensor(audio_segment, dtype=torch.float)
        return (
            audio_tensor,
            label,
        )

def createMegaDataSet():
    listOfDataSets = []

    for file in os.listdir("mfaAligned"):
        listOfDataSets.append(AudioSegmentDataset(os.path.join("mfaAligner", (file.split(".")[0])+ ".wav"), os.path.join("mfaAligned", file), sr=16000))

    bigDataSet = ConcatDataset(listOfDataSets)
    return bigDataSet

class soundToEnglishModel(nn.Module):
    def __init__(self, input_channels=1, n_mels=64, time_steps=64):
        super().__init__()
        self.input_channels = input_channels
        self.n_mels = n_mels
        self.time_steps = time_steps
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=24, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(24, 24, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(24),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=(3, 3), stride=(2, 2), padding=1),
            nn.BatchNorm2d(48),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2),
        )
        
        # Calculate the output size after all convolutions
        # After 2 stride-2 convolutions, the dimensions are reduced by a factor of 4
        mel_reduced = n_mels // 4
        time_reduced = time_steps // 4
        
        # Final feature map size: [batch_size, 64, mel_reduced, time_reduced]
        feature_size = 64 * mel_reduced * time_reduced
        
        self.connected_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_size, 200),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(200, len(labelLexicon) + 1),  # +1 for potential padding/blank class
        )

    def forward(self, input_data):
        # input_data shape should be [batch_size, channels, n_mels, time_steps]
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.connected_layer(x)
        return x

def extract_features(audio, sr=16000, n_mels=64, n_fft=1024, hop_length=512, fixed_length=64):
    """
    Convert audio to mel spectrogram with fixed dimensions.
    Returns a tensor of shape [1, n_mels, fixed_length]
    """
    # Check if audio is too short for even one frame
    if len(audio) < n_fft:
        # Pad the audio to minimum viable length
        padding_needed = n_fft - len(audio)
        audio = F.pad(audio, (0, padding_needed))
    
    # Create mel spectrogram extractor
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to spectrogram
    with torch.no_grad():
        spec = mel_spec(audio)
    
    # Apply log transform (adding small constant to avoid log(0))
    spec = torch.log1p(spec)
    
    # Normalize
    if spec.numel() > 0:  # Check if tensor is not empty
        spec = (spec - spec.mean()) / (spec.std() + 1e-10)
    
    # Handle the time dimension padding/truncation
    # spec shape is [n_mels, time]
    curr_time = spec.shape[1]
    
    if curr_time < fixed_length:
        # Pad the time dimension - pad the right side only
        padding = fixed_length - curr_time
        spec = F.pad(spec, (0, padding))
    elif curr_time > fixed_length:
        # Truncate to fixed length
        spec = spec[:, :fixed_length]
    
    return spec.unsqueeze(0)  # Add channel dimension: [1, n_mels, fixed_length]

def my_collate_fn(batch):
    """
    Custom collate function that ensures all spectrograms have the same dimensions.
    """
    audios, labels = zip(*batch)
    
    # Convert each audio to a fixed-size spectrogram (1, n_mels, time_steps)
    spectrograms = [extract_features(audio) for audio in audios]
    
    # Stack spectrograms into a batch - should result in [batch_size, 1, n_mels, time_steps]
    spectrogram_batch = torch.stack(spectrograms)
    
    # Convert labels to tensor
    label_batch = torch.tensor([labelLexicon.get(label, 0) for label in labels], dtype=torch.long)
    
    return spectrogram_batch, label_batch

def trainingFunction():
    dataset = createMegaDataSet()
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # Use our modified collate function
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=my_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=my_collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize model with correct input parameters
    model = soundToEnglishModel(input_channels=1, n_mels=64, time_steps=64).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()  # Define criterion properly

    epochs = 5
    patience = 5
    patience_counter = 0
    best_accuracy = 0.0  # Track best accuracy instead of loss for early stopping
    
    # For Graphing
    train_accuracy_list = []
    test_accuracy_list = []

    for epoch in range(1, epochs + 1):
        print(f"Epoch {epoch}/{epochs}")
        train_accuracy = train(model, device, train_loader, optimizer, criterion, epoch)  # Added criterion
        test_accuracy = test(model, device, test_loader)
        print(f"Train Accuracy: {train_accuracy:.4f} | Test Accuracy: {test_accuracy:.4f}")

        train_accuracy_list.append(train_accuracy)
        test_accuracy_list.append(test_accuracy)

        # Early stopping check
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), "best_model.pth")
            print("Model improved. Saving checkpoint.")
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter: {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered. Stopping training.")
                break
            
    torch.save(model.state_dict(), "final_model.pth")

    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracy_list, label='Train Accuracy', marker='o')
    plt.plot(test_accuracy_list, label='Test Accuracy', marker='x')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.savefig("accuracy.png")
    plt.show()

def train(model, device, train_loader, optimizer, criterion, epoch_number):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    print("------------------------------- Epoch:", epoch_number,"-------------------------------")

    for batch_idx, (spectrograms, labels) in enumerate(train_loader):
        # Move data to device - spectrograms shape should be [batch_size, 1, n_mels, time_steps]
        spectrograms, labels = spectrograms.to(device), labels.to(device)
        
        optimizer.zero_grad()
        # No need to add extra dimension, spectrograms is already batched
        outputs = model(spectrograms)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if batch_idx % 20 == 0:
            print(f'Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.6f}')

    avg_loss = train_loss / len(train_loader)
    accuracy = 100. * correct / total
    print(f'Training set: Average loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%')
    return accuracy

def test(model, device, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for spectrograms, labels in test_loader:
            spectrograms, labels = spectrograms.to(device), labels.to(device)
            
            outputs = model(spectrograms)
            
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100. * correct / total
    print(f'Test set: Accuracy: {accuracy:.2f}%')
    return accuracy

# Call the training function
if __name__ == "__main__":
    trainingFunction()
