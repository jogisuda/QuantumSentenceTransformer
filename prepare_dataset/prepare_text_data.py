import pandas as pd

from torch.utils.data import Dataset, DataLoader, random_split


class CustomTextDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text
    def __len__(self):
            return len(self.labels)
    def __getitem__(self, idx):
            label = self.labels[idx]
            text = self.text[idx]
            return text, label

##########################################
#Load your dataset in DataFrame in the following format:
# ----------------------------------
#|   text         |      label     |
#|   example 1    |        0       |
#|   example 2    |        1       |
#|     ...        |       ...      |
#-----------------------------------
##########################################


df = pd.read_pickle("your_text_dataset.pkl")
        
    
#create the CustomTextDataSet object and split in training and validation
#using the chosen number of samples
TextDataset = CustomTextDataset(df['text'], df['label'])
train_set, validation_set = random_split(TextDataset, [282, 32])


DL_Train = DataLoader(train_set, batch_size=32, shuffle=True)
DL_Validation = DataLoader(validation_set, batch_size=32, shuffle=True)


dataloaders = {
    "train": DL_Train,
    "validation": DL_Validation
}
dataset_sizes = {
    "train": len(train_set),
    "validation": len(validation_set)
}
        