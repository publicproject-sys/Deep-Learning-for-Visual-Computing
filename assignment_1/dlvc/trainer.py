import torch
from typing import Tuple
from abc import ABCMeta, abstractmethod
from pathlib import Path
from tqdm import tqdm
from dlvc.wandb_logger import WandBLogger
import numpy as np

class BaseTrainer(metaclass=ABCMeta):
    '''
    Base class for all Trainers.
    '''

    @abstractmethod
    def train(self) -> None:
        '''
        Holds training logic.
        '''
        pass

    @abstractmethod
    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        '''
        Holds training logic for one epoch.
        '''
        pass

    @abstractmethod
    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        '''
        Holds validation logic for one epoch.
        '''
        pass


class ImgClassificationTrainer(BaseTrainer):
    """
    Class that stores the logic for training a model for image classification.
    """
    def __init__(
        self, 
        model, 
        optimizer,
        loss_fn,
        lr_scheduler,
        train_metric,
        val_metric,
        train_data,
        val_data,
        device,
        num_epochs,
        training_save_dir,
        batch_size=4,
        val_frequency=5
    ):
        '''
        Args:
            - model: Deep neural network to train
            - optimizer: Optimizer to use during training
            - loss_fn: Loss function used during training
            - lr_scheduler: Learning rate scheduler (optional)
            - train_metric: Instance of the Accuracy class for measuring training accuracy
            - val_metric: Instance of the Accuracy class for measuring validation accuracy
            - train_data: Training dataset
            - val_data: Validation dataset
            - device: Device to use for training (e.g., "cuda" or "cpu")
            - num_epochs: Number of epochs for training
            - training_save_dir: Path to save the best-performing model
            - batch_size: Number of samples per batch
            - val_frequency: Frequency of validation (e.g., every 5 epochs)
        '''

        # Store parameters
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_scheduler = lr_scheduler
        self.train_metric = train_metric
        self.val_metric = val_metric
        self.train_data = train_data
        self.val_data = val_data
        self.device = device
        self.num_epochs = num_epochs
        self.training_save_dir = training_save_dir
        self.batch_size = batch_size
        self.val_frequency = val_frequency

        # DataLoaders for training and validation
        self.train_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True
        )
        self.val_loader = torch.utils.data.DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False
        )

        # initialize wandb logger
        self.wandb_logger = WandBLogger(enabled=True, model=self.model)

        # Initialize mean per-class accuracy
        self.best_per_class_accuracy = 0.0

    def _train_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Training logic for one epoch.
        Returns loss, mean accuracy, and mean per-class accuracy for this epoch.
        """
        self.model.train()
        running_loss = 0.0
        self.train_metric.reset()

        # Loop over batches
        for batch in tqdm(self.train_loader, desc=f"Training Epoch {epoch_idx}"):
            images, labels = batch
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass and loss calculation
            outputs = self.model(images)
            loss = self.loss_fn(outputs, labels)

            # Backward pass and optimization step
            loss.backward()
            self.optimizer.step()

            # Update running loss and accuracy metric
            running_loss += loss.item()
            self.train_metric.update(outputs, labels)

        # Calculate mean loss and accuracy
        mean_loss = running_loss / len(self.train_loader)
        mean_accuracy = self.train_metric.accuracy()
        mean_per_class_accuracy = self.train_metric.per_class_accuracy()
        
        mean_class_accuracy = np.mean(list(mean_per_class_accuracy.values()))

        # log train metrics to wandb
        self.wandb_logger.log({
            'epoch' : epoch_idx,
            'train/loss' : mean_loss,
            'train/mAcc' : mean_accuracy,
            'train/mClassAcc' : mean_class_accuracy,
            'train/mPerClassAcc' : mean_per_class_accuracy
        })

        return mean_loss, mean_accuracy, mean_per_class_accuracy

    def _val_epoch(self, epoch_idx: int) -> Tuple[float, float, float]:
        """
        Validation logic for one epoch.
        Returns loss, mean accuracy, and mean per-class accuracy for this epoch.
        """
        self.model.eval()
        running_loss = 0.0
        self.val_metric.reset()

        with torch.no_grad():
            # Loop over validation batches
            for batch in tqdm(self.val_loader, desc=f"Validation Epoch {epoch_idx}"):
                images, labels = batch
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Forward pass and loss calculation
                outputs = self.model(images)
                loss = self.loss_fn(outputs, labels)

                # Update running loss and accuracy metric
                running_loss += loss.item()
                self.val_metric.update(outputs, labels)

        # Calculate mean loss and accuracy
        mean_loss = running_loss / len(self.val_loader)
        mean_accuracy = self.val_metric.accuracy()
        mean_per_class_accuracy = self.val_metric.per_class_accuracy()

        mean_class_accuracy = np.mean(list(mean_per_class_accuracy.values()))

        # log val metrics to wandb
        self.wandb_logger.log({
            'epoch' : epoch_idx,
            'val/loss' : mean_loss,
            'val/mAcc' : mean_accuracy,
            'val/mClassAcc' : mean_class_accuracy,
            'val/mPerClassAcc' : mean_per_class_accuracy
        })

        return mean_loss, mean_accuracy, mean_per_class_accuracy

    def train(self) -> None:
        """
        Full training logic that loops over num_epochs and
        uses the _train_epoch and _val_epoch methods.
        Save the model if mean per-class accuracy on validation dataset is higher
        than the current best.
        """
        for epoch_idx in range(1, self.num_epochs + 1):
            # Train for one epoch
            train_loss, train_accuracy, train_per_class_accuracy = self._train_epoch(epoch_idx)

            print(f"______epoch {epoch_idx}")
            print(f"Training loss: {train_loss:.4f}")
            print(f"Training accuracy: {train_accuracy:.4f}")
            print("Training per-class accuracy:")
            for class_idx, acc in train_per_class_accuracy.items():
                print(f"Accuracy for class {class_idx} is {acc:.2%}")

            # Validate if it's time to do so based on val_frequency
            if epoch_idx % self.val_frequency == 0:
                val_loss, val_accuracy, val_per_class_accuracy = self._val_epoch(epoch_idx)

                print(f"Validation loss: {val_loss:.4f}")
                print(f"Validation accuracy: {val_accuracy:.4f}")
                print("Validation per-class accuracy:")
                for class_idx, acc in val_per_class_accuracy.items():
                    print(f"Accuracy for class {class_idx} is {acc:.2%}")

                # Check if this is the best performance so far
                current_per_class_accuracy = sum(val_per_class_accuracy.values()) / len(val_per_class_accuracy)
                if current_per_class_accuracy > self.best_per_class_accuracy:
                    self.best_per_class_accuracy = current_per_class_accuracy
                    # Save the model if it performed the best on the validation set
                    save_path = self.training_save_dir / f"best_model_epoch_{epoch_idx}.pth"
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved at {save_path}")

            # Adjust the learning rate if you have a scheduler
            if self.lr_scheduler:
                self.lr_scheduler.step()

        self.wandb_logger.finish()