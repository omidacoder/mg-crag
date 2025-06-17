import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.Helpers.ResNet import ResNet
import pickle
import random
from src.Helpers.Utils import accuracy, precision_recall_f1_high
from configuration import num_of_labeled_data
class ClassificationHeadTrainer:
    def __init__(self):
        pass
    def prepare_data(self,clustering_result_path="clustering.pickle"):
        with open(clustering_result_path, 'rb') as handle:
            dataset = pickle.load(handle)
            texts = dataset["texts"]
            labels = dataset["labels"]
            embeddings = dataset["embeddings"]
            self.test_embeddings = embeddings[:num_of_labeled_data]
            embeddings = embeddings[num_of_labeled_data:]
            self.test_texts = texts[:num_of_labeled_data]
            texts = texts[num_of_labeled_data:]
            self.test_labels = labels[:num_of_labeled_data]
            labels = labels[num_of_labeled_data:]
            low_label_data = [(emb, text, label) for emb, text, label in zip(embeddings,texts, labels) if label == 0]
            ambigous_label_data = [(emb, text, label) for emb, text, label in zip(embeddings,texts, labels) if label == 1]
            high_label_data = [(emb, text, label) for emb, text, label in zip(embeddings,texts, labels) if label == 2]
            # Balancing Data
            min_len = min(len(low_label_data),len(high_label_data),len(ambigous_label_data))
            selected_low_label_data= random.sample(low_label_data, min_len)
            selected_high_label_data= random.sample(high_label_data, min_len)
            selected_ambigous_label_data = random.sample(ambigous_label_data, min_len)
            dataset = selected_low_label_data + selected_ambigous_label_data + selected_high_label_data
            self.embeddings = []
            self.texts = []
            self.labels = []
            for d in dataset:
                self.embeddings.append(d[0])
                self.texts.append(d[1])
                self.labels.append(int(d[2]))
            self.input_len = embeddings[0].shape[0]

    def train(self, save_path="/content"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = ResNet(self.input_len).to(device)
        criterion = nn.CrossEntropyLoss()
        train = TensorDataset(
            torch.tensor(self.embeddings, device=device),
            torch.tensor(self.labels, dtype=torch.long, device=device)
        )
        train_dataloader = DataLoader(train, batch_size=10, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
        # Train the network
        for epoch in tqdm(range(50), desc="Training net ..."):
            running_loss = 0.0
            for i, data in enumerate(train_dataloader, 0):
                optimizer.zero_grad()
                inputs, real_labels = data
                inputs, real_labels = inputs.to(device), real_labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, real_labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                running_loss += loss.item()
                if i % 100 == 99:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
            scheduler.step(loss)
            # Train Accuracy
            embeddings_device = torch.tensor(self.embeddings, device=device)
            labels_device = torch.tensor(self.labels, dtype=torch.long, device=device)
            train_accuracy = accuracy(model, embeddings_device, labels_device)
            print(f"Train Accuracy in epoch {epoch} is : {str(train_accuracy)}")
            # Test Accuracy
            test_embeddings_device = torch.tensor(self.test_embeddings, device=device)
            test_labels_device = torch.tensor(self.test_labels, dtype=torch.long, device=device)
            test_accuracy = accuracy(model, test_embeddings_device, test_labels_device)
            print(f"Test Accuracy in epoch {epoch} is : {str(test_accuracy)}")
            # Train Precision, Recall, F1
            train_precision, train_recall, train_f1 = precision_recall_f1_high(
                model, embeddings_device, labels_device
            )
            print(f"Train Precision for 'high' class in epoch {epoch} is : {str(train_precision)}")
            print(f"Train Recall for 'high' class in epoch {epoch} is : {str(train_recall)}")
            print(f"Train F1 for 'high' class in epoch {epoch} is : {str(train_f1)}")
            # Test Precision, Recall, F1
            test_precision, test_recall, test_f1 = precision_recall_f1_high(
                model, test_embeddings_device, test_labels_device
            )
            print(f"Test Precision for 'high' class in epoch {epoch} is : {str(test_precision)}")
            print(f"Test Recall for 'high' class in epoch {epoch} is : {str(test_recall)}")
            print(f"Test F1 for 'high' class in epoch {epoch} is : {str(test_f1)}")
            torch.save(model.state_dict(), f'{save_path}/model_epoch_{epoch}.pt')

        print('Finished Training')