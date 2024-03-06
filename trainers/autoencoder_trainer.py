import torch
from torch.optim import Adagrad
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

def vae_loss(recon_x, x, z, z_q, beta=0.25):
    rec_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='mean')
    quantization_loss = torch.nn.functional.mse_loss(z.detach(), z_q, reduction='mean') + beta * torch.nn.functional.mse_loss(z, z_q.detach(), reduction='mean')
    return rec_loss + quantization_loss, rec_loss, quantization_loss

class VQVAE_Trainer():
    def __init__(self, model, config):
        self.model = model.to(config['device'])
        self.optimizer = Adagrad(model.parameters(), lr=config['lr'])
        self.device = config['device']
        self.count_low_usage = config['count_low_usage']
        self.writer = SummaryWriter(log_dir='./logs')
        self.best_loss = float('inf')
        self.early_stop_patience = config['early_stop_patience']
        self.patience_counter = 0
        self.epochs = config['epochs']
        self.lr = config['lr']

    def train(self, train_loader, validation_loader):
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.epochs * len(train_loader), eta_min=self.lr/10)
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            train_rec_loss = 0.0
            train_quantization_loss = 0.0
            batch_num = 0
            total_count = 0
            progress_bar = tqdm(train_loader)
            for data in train_loader:
                batch_num += 1
                data = data.to(self.device)
                self.optimizer.zero_grad()
                if self.count_low_usage:
                    recon_x, z_q, z, count = self.model(data, count_low_usage=self.count_low_usage)
                    total_count += count
                else:
                    recon_x, z_q, z = self.model(data)
                loss, rec_loss, quantization_loss = vae_loss(recon_x, data, z, z_q)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                train_loss += loss.item()
                train_rec_loss += rec_loss.item()
                train_quantization_loss += quantization_loss.item()
                if total_count > 0:
                    progress_bar.set_description(f"Epoch {epoch+1}, Loss: {(train_loss/batch_num):.4f}, Rec: {(train_rec_loss/batch_num):.4f}, Quant: {(train_quantization_loss/batch_num):.4f}, Count: {(total_count/batch_num):.1f}")
                else:
                    progress_bar.set_description(f"Epoch {epoch+1}, Loss: {(train_loss/batch_num):.4f}, Rec: {(train_rec_loss/batch_num):.4f}, Quant: {(train_quantization_loss/batch_num):.4f}")
                progress_bar.update(1)
            progress_bar.close()
            avg_train_loss = train_loss / len(train_loader)
            avg_rec_loss = train_rec_loss / len(train_loader)
            avg_quantization_loss = train_quantization_loss / len(train_loader)
            self.writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)
            self.writer.add_scalar('Epoch/train_rec_loss', avg_rec_loss, epoch)
            self.writer.add_scalar('Epoch/train_quantization_loss', avg_quantization_loss, epoch)

            val_loss = self.validate(validation_loader, epoch)
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), './results/best_model.pth')
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter > self.early_stop_patience:
                    print("Early stopping triggered.")
                    break

            if epoch % 5 == 0:
                torch.save(self.model.state_dict(), f'./results/model_epoch_{epoch}.pth')

    def validate(self, validation_loader, epoch):
        self.model.eval()
        val_loss = 0.0
        val_rec_loss = 0.0
        val_quantization_loss = 0.0
        batch_num = 0
        with torch.no_grad():
            for data in validation_loader:
                batch_num += 1
                data = data.to(self.device)
                if self.count_low_usage:
                    recon_x, z_q, z, count = self.model(data, count_low_usage=self.count_low_usage)
                else:
                    recon_x, z_q, z = self.model(data)
                loss, rec_loss, quantization_loss = vae_loss(recon_x, data, z, z_q)
                val_loss += loss.item()
                val_rec_loss += rec_loss.item()
                val_quantization_loss += quantization_loss.item()
                # Log validation losses

        avg_val_loss = val_loss / len(validation_loader)
        avg_rec_loss = val_rec_loss / len(validation_loader)
        self.writer.add_scalar('Epoch/val_loss', avg_val_loss, epoch)
        avg_quantization_loss = val_quantization_loss / len(validation_loader)
        self.writer.add_scalar('Epoch/val_rec_loss', avg_rec_loss, epoch)
        self.writer.add_scalar('Epoch/val_quantization_loss', avg_quantization_loss, epoch)
        print(f'Epoch {epoch+1}, Val Loss: {avg_val_loss:.3f}, Rec: {avg_rec_loss:.3f}, Quant: {avg_quantization_loss:.3f}')
        return avg_rec_loss

    def test(self, test_loader):
        self.model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data.to(self.device)
                if self.count_low_usage:
                    recon_x, z_q, z, count = self.model(data, count_low_usage=self.count_low_usage)
                else:
                    recon_x, z_q, z = self.model(data)
                _, loss, _ = vae_loss(recon_x, data, z, z_q)
                test_loss += loss.item()

        avg_test_loss = test_loss / len(test_loader)
        print(f'Test rec Loss: {avg_test_loss}')
