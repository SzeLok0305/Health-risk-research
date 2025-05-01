import torch
import torch.nn as nn
import torch.optim as optim

class NeuralPredictor:
    def __init__(self, input_size, hidden_layers=[64, 32], dropout_rate=0.2, learning_rate=0.001, output_size=1):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.output_size = output_size
        
        self.model = self._build_model()
        self.criterion = nn.BCELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
    def _build_model(self):
        layers = []
        prev_size = self.input_size
        
        for hidden_size in self.hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout_rate))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, self.output_size))
        layers.append(nn.Sigmoid())
        
        return nn.Sequential(*layers)
    
    def train(self, train_loader, test_loader=None, num_epochs=100, verbose=True, eval_every=1, print_every=10):
        history = {
            'train_loss': [], 
            'train_accuracy': [],
            'test_loss': [], 
            'test_accuracy': [],
            'epochs': []
        }
        
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                predicted = (outputs > 0.5).float()
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            avg_train_loss = train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            if (epoch+1) % eval_every == 0:
                history['train_loss'].append(avg_train_loss)
                history['train_accuracy'].append(train_accuracy)
                history['epochs'].append(epoch+1)
                
                if test_loader is not None:
                    test_loss, test_accuracy = self.evaluate(test_loader)
                    history['test_loss'].append(test_loss)
                    history['test_accuracy'].append(test_accuracy)
                    
                    if verbose and (epoch+1) % print_every == 0:
                        print(f'Epoch {epoch+1}/{num_epochs}, '
                              f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                              f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.4f}')
                elif verbose and (epoch+1) % print_every == 0:
                    print(f'Epoch {epoch+1}/{num_epochs}, '
                          f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
                
        if len(history['epochs']) > 1:
            self.plot_history(history)
            
        return history
    
    def predict(self, X, threshold=0.5):
        self.model.eval()
        with torch.no_grad():
            if not isinstance(X, torch.Tensor):
                X = torch.FloatTensor(X)
            
            probabilities = self.model(X)
            predictions = (probabilities > threshold).float()
            
        return probabilities, predictions
    
    def evaluate(self, test_loader, threshold=0.5):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                predicted = (outputs > threshold).float()
                correct += (predicted == labels).sum().item()
                total += labels.size(0)
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def plot_history(self, history):
        try:
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(history['epochs'], history['train_loss'], label='Train Loss')
            if 'test_loss' in history and len(history['test_loss']) > 0:
                plt.plot(history['epochs'], history['test_loss'], label='Test Loss')
            plt.title('Loss over epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True)
            
            plt.subplot(1, 2, 2)
            plt.plot(history['epochs'], history['train_accuracy'], label='Train Accuracy')
            if 'test_accuracy' in history and len(history['test_accuracy']) > 0:
                plt.plot(history['epochs'], history['test_accuracy'], label='Test Accuracy')
            plt.title('Accuracy over epochs')
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib is required for plotting. Please install it with 'pip install matplotlib'")
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()
