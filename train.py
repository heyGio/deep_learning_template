import torch
import mlflow
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

from model import Model
from utils import setup_seed
from engine import train_step, val_step
from data_setup import create_dataloeaders
from torchvision.transforms import ToTensor, Compose, Normalize, Resize

import warnings
warnings.filterwarnings('ignore')


def main(args):
    # Init 
    setup_seed(args.seed)
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'

    # Setup mlflow
    mlflow.set_tracking_uri(uri=f'http://localhost:{args.port}')
    mlflow.set_experiment(args.experiment_name)

    # Creating dataset
    transform = Compose([ToTensor(), Resize((256, 256)), Normalize(0.5, 0.5)])
    train_dl, val_dl  = create_dataloeaders(args.dataset, transform, args.batch_size, args.num_workers, args.train_val_ratio)

    # Creating model
    model = Model(input_shape=3, hidden_units=10, output_shape=32).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    with mlflow.start_run(run_name = args.run_name):
        # Log parameters
        params = vars(args) 
        params['loss_function'] = loss_fn.__class__.__name__
        params['optimizer'] = optimizer.__class__.__name__
        mlflow.log_params(params)
        mlflow.log_input(mlflow.data.from_pandas(pd.read_csv(args.dataset), name='VidTIMIT_uv', targets='class'))
        example_input = next(iter(val_dl))[0]
        signature = mlflow.models.infer_signature(example_input.numpy(), model(example_input.to(device)).detach().cpu().numpy())

        # Training 
        best_val_loss = np.inf
        patience_count = 1

        for epoch in range(args.epochs):
            # Train step
            train_loss, train_acc = train_step(model=model,
                                                dataloader=train_dl,
                                                loss_fn=loss_fn,
                                                optimizer=optimizer,
                                               device=device)
            # Validation step 
            val_loss, val_acc = val_step(model=model,
                                                dataloader=val_dl,
                                                loss_fn=loss_fn,
                                                device=device)

            # Patience check 
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 1
                print(f'Saving new best model at epoch {epoch} with val loss = {best_val_loss}')
                mlflow.pytorch.log_model(model, f'best_model_{args.run_name}', signature=signature)


            # Print out what's happening
            print(
                f"Epoch: {epoch+1} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.4f} | "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {val_acc:.4f}"
            )


            mlflow.log_metric("train_loss", f"{train_loss:.4f}", step=epoch)
            mlflow.log_metric("val_loss", f"{val_loss:.4f}", step=epoch)
            mlflow.log_metric("train_acc", f"{100*train_acc:.2f}", step=epoch)
            mlflow.log_metric("val_acc", f"{100*val_acc:.2f}", step=epoch)

            patience_count += 1
            if patience_count == args.patience:
                print(f'Training stopped at epoch {epoch} becuase of impatience')
                break
        


    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Mlflow 
    parser.add_argument('--experiment_name', type=str, default='experiment1')
    parser.add_argument('--run_name', type=str, default='run1')
    parser.add_argument('--port', type=int, default=7777)
    
    # Model parameters
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--patience', type=int, default=5)

    # Dataset
    parser.add_argument('--dataset', type=str, default='data/dataset.csv')
    parser.add_argument('--train_val_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_workers', type=int, default=5)

    # Other parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=int, default=1)

    main(parser.parse_args())
