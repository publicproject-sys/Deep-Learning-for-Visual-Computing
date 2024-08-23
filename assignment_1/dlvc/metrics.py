from abc import ABCMeta, abstractmethod
import torch

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''

        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''

        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''

        pass



class Accuracy(PerformanceMeasure):
    '''
    Average classification accuracy.
    '''

    def __init__(self, classes) -> None:
        self.classes = classes

        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        ## TODO implement
        self.correct = 0 
        self.total = 0 
        self.correct_per_class = torch.zeros(self.classes) 
        self.total_per_class = torch.zeros(self.classes)

    def update(self, prediction: torch.Tensor, 
               target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (s,c) with each row being a class-score vector.
        target must have shape (s,) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        '''

        ## TODO implement
        # Validate input shapes and values
        if prediction.ndim != 2 or target.ndim != 1:
            raise ValueError("Prediction must have shape (s, c) and target must have shape (s,).")
        
        if prediction.shape[0] != target.shape[0]:
            raise ValueError("The number of predictions must match the number of targets.")
        
        if not torch.all(target >= 0) or not torch.all(target < self.classes):
            raise ValueError("Target class values must be between 0 and c-1.")

        # Get the predicted class
        pred_class = torch.argmax(prediction, dim=1)

        # Count correct predictions
        correct_predictions = torch.sum(pred_class == target).item()

        # Update
        self.correct += correct_predictions
        self.total += target.shape[0]

        # Update per-class accuracy
        for i in range(self.classes):
            class_mask = (target == i)
            self.correct_per_class[i] += torch.sum(pred_class[class_mask] == target[class_mask]).item()
            self.total_per_class[i] += torch.sum(class_mask).item()

    def __str__(self):
        '''
        Return a string representation of the performance, accuracy and per class accuracy.
        '''

        ## TODO implement
        return f"Overall accuracy: {self.accuracy():.2%}, " \
               f"Per-class accuracy: {self.per_class_accuracy()}"


    def accuracy(self) -> float:
        '''
        Compute and return the accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''

        ## TODO implement
        if self.total == 0:
            return 0.0
        return self.correct / self.total
    
    def per_class_accuracy(self) -> float:
        '''
        Compute and return the per class accuracy as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        '''
        ## TODO implement
        accuracies = {}
        for i in range(self.classes):
            if self.total_per_class[i] == 0:
                accuracies[i] = 0.0
            else:
                accuracies[i] = self.correct_per_class[i] / self.total_per_class[i]
        return accuracies
       