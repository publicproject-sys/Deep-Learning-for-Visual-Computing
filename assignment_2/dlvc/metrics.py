from abc import ABCMeta, abstractmethod
import torch
import torch.nn.functional as F

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


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes
        self.num_classes = len(classes)
        self.confusion_matrix = torch.zeros(self.num_classes, self.num_classes)

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.confusion_matrix.zero_()

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''
        if prediction.shape[1] != self.num_classes:
            raise ValueError("Number of classes in prediction does not match the number of classes provided.")

        if prediction.shape[0] != target.shape[0] or prediction.shape[2:] != target.shape[1:]:
            raise ValueError("Prediction and target shapes do not match.")

        mask = (target != 255)
        prediction = F.softmax(prediction, dim=1).argmax(dim=1)
        
        # Convert target to long tensor
        target = target.long()

        # Calculate indices for confusion matrix
        flat_target = target[mask].view(-1)
        flat_prediction = prediction[mask].view(-1)
        indices = flat_target * self.num_classes + flat_prediction

        # Update confusion matrix
        self.confusion_matrix += torch.bincount(indices, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)
    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIou: {self.mIoU():.2f}"

    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        intersection = torch.diag(self.confusion_matrix)
        union = self.confusion_matrix.sum(dim=0) + self.confusion_matrix.sum(dim=1) - intersection
        iou = intersection / union
        iou[torch.isnan(iou)] = 0
        return iou.mean().item()
