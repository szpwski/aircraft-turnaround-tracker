"""
Module: model.py
Description: This module defines the ObjectDetector class, which encapsulates the functionality for running object detection
"""
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.utils import draw_bounding_boxes


class ObjectDetector:
    def __init__(self, device='cuda', confidence_threshold=0.5):
        """
        Initialize the object detection model and set up device and confidence threshold.
        
        Args:
            device (str): The device to run the model on ('cuda' or 'cpu').
            confidence_threshold (float): Minimum confidence score for keeping detections.
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.coco_classes = self.weights.meta["categories"]
        self.model = fasterrcnn_resnet50_fpn(weights=self.weights).to(self.device)
        self.model.eval()
        self.confidence_threshold = confidence_threshold
        
    def detect(self, image_rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Run object detection on the input RGB image and return bounding boxes, labels, and scores.
        
        Args:
            image_rgb (np.ndarray): Input image in RGB format as a NumPy array.
        
        Returns:
            boxes (np.ndarray): Array of bounding box coordinates (x1, y1, x2, y2).
            labels (list[str]): List of class labels for each detected object.
            scores (np.ndarray): Array of confidence scores for each detected object.
        """
        image_tensor = F.to_tensor(image_rgb).to(self.device)
        with torch.no_grad():
            outputs = self.model([image_tensor])[0]
        
        keep_idx = outputs["scores"].cpu() > self.confidence_threshold
        boxes = outputs["boxes"].cpu()[keep_idx].numpy()
        labels = [self.coco_classes[label.item()] for label in outputs["labels"].cpu()[keep_idx]]
        scores = outputs["scores"].cpu()[keep_idx].numpy()

        return boxes, labels, scores