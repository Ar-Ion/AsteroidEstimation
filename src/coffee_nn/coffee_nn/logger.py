import wandb
import torch

# Cloud reporting
class Logger:
    def __init__(self, name):
        wandb.init(project=name)
        
    def report(self, iteration, train_losses, train_stats, val_losses, val_stats):
        # Compute statistics
        avg_train_loss = torch.tensor(train_losses).mean()
        avg_val_loss = torch.tensor(val_losses).mean()
        avg_train_accuracy = torch.tensor(list(map(lambda x: x.accuracy(), train_stats))).mean()
        avg_val_accuracy = torch.tensor(list(map(lambda x: x.accuracy(), val_stats))).mean()
        avg_train_precision = torch.tensor(list(map(lambda x: x.precision(), train_stats))).mean()
        avg_val_precision = torch.tensor(list(map(lambda x: x.precision(), val_stats))).mean()
        avg_train_recall = torch.tensor(list(map(lambda x: x.recall(), train_stats))).mean()
        avg_val_recall = torch.tensor(list(map(lambda x: x.recall(), val_stats))).mean()
        avg_train_f1 = torch.tensor(list(map(lambda x: x.f1(), train_stats))).mean()
        avg_val_f1 = torch.tensor(list(map(lambda x: x.f1(), val_stats))).mean()
        avg_pixel_error = torch.tensor(list(map(lambda x: x.pixel_error(), val_stats))).mean()

        # Log statistics
        print(f"Iteration {iteration} finished with train loss {avg_train_loss:.4}, val loss {avg_val_loss:.4}, val f1 {avg_val_f1:.1%}")
        
        wandb.log({
            "Train loss": avg_train_loss, 
            "Validation Loss": avg_val_loss, 
            "Train accuracy": avg_train_accuracy*100, 
            "Validation accuracy": avg_val_accuracy*100, 
            "Train precision": avg_train_precision*100, 
            "Validation precision": avg_val_precision*100, 
            "Train recall": avg_train_recall*100, 
            "Validation recall": avg_val_recall*100, 
            "Train F1-Score": avg_train_f1*100, 
            "Validation F1-Score": avg_val_f1*100, 
            "Pixel error": avg_pixel_error 
        })