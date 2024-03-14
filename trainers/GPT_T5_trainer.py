import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class GPT_T5_Trainer():
    def __init__(self, model, config):
        self.model = model.to(config['device'])
        self.lr = float(config['lr'])
        self.optimizer = AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.device = config['device']
        self.best_loss = float('inf')
        self.early_stop_patience = config['early_stop_patience']
        self.patience_counter = 0
        self.epochs = config['epochs']
        self.exp_id = config['exp_id']
        self.writer = SummaryWriter(log_dir=f'./logs_GPT_T5/exp{self.exp_id}/')
        self.hier = config['hier']
        print(model)
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'{total_params:,} total parameters.')

    def train(self, train_loader, validation_loader):
        self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=len(train_loader), eta_min=self.lr/10)
        eval_size = 3
        eval_stops = [i * len(train_loader) // eval_size for i in range(1, eval_size)] + [len(train_loader) - 1]
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = 0.0
            batch_num = 0
            progress_bar = tqdm(train_loader)
            eval_idx = 0
            for idx, (top, bottom) in train_loader:
                batch_num += 1
                self.optimizer.zero_grad()
                top = top.to(self.device)

                if self.hier == 'top':
                    loss = self.model(top)
                else:
                    bottom = bottom.to(self.device)
                    loss = self.model(bottom, condition=top)

                if idx == eval_stops[eval_idx]:
                    eval_idx += 1
                    val_loss = self.evaluate(validation_loader, epoch)
                    if val_loss < self.best_loss:
                        self.best_loss = val_loss
                        if isinstance(self.model, torch.nn.DataParallel):
                            # Save the original model which is accessible via model.module
                            torch.save(self.model.module.state_dict(), f'./results/best_model_{self.hier}.pth')
                        else:
                            # Save the model directly if not using DataParallel
                            torch.save(self.model.state_dict(), f'./results/best_model_{self.hier}.pth')
                        self.patience_counter = 0
                    else:
                        self.patience_counter += 1
                        if self.patience_counter > self.early_stop_patience:
                            print("Early stopping triggered.")
                            break

                loss.backward()
                
                self.optimizer.step()
                self.scheduler.step()
                train_loss += loss.item()
                progress_bar.set_description(f"Epoch {epoch+1}, Loss: {(train_loss/batch_num):.4f}")
                progress_bar.update(1)
            progress_bar.close()
            avg_train_loss = train_loss / len(train_loader)
            self.writer.add_scalar('Epoch/train_loss', avg_train_loss, epoch)

            if epoch % 25 == 0:
                if isinstance(self.model, torch.nn.DataParallel):
                    # Save the original model which is accessible via model.module
                    torch.save(self.model.module.state_dict(), f'./results/best_model_{self.hier}.pth')
                else:
                    # Save the model directly if not using DataParallel
                    torch.save(self.model.state_dict(), f'./results/best_model_{self.hier}.pth')

    def evaluate(self, validation_loader, epoch):
        self.model.eval()
        val_loss = 0.0
        batch_num = 0
        with torch.no_grad():
            for top, bottom in tqdm(validation_loader):
                batch_num += 1
                top = top.to(self.device)

                if self.hier == 'top':
                    loss = self.model(top)
                elif self.hier == 'bottom':
                    bottom = bottom.to(self.device)
                    loss = self.model(bottom, condition=top)

                val_loss += loss.item()
        avg_val_loss = val_loss / len(validation_loader)
        self.writer.add_scalar('Epoch/val_loss', avg_val_loss, epoch)
        print(f'Epoch {epoch+1}, Val Loss: {avg_val_loss:.4f}')
        return avg_val_loss
