import torch
from pytorchvideo.transforms import ApplyTransformToKey, UniformTemporalSubsample, RandomShortSideScale, \
    ShortSideScale, Normalize
from torch import nn
from torchvision.transforms import Compose, Lambda, RandomCrop, RandomHorizontalFlip, CenterCrop

side_size = 400
max_size = 400
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 400
num_frames = 32
sampling_rate = 1
frames_per_second = 5
clip_duration = 6
num_classes = 3


class PackPathway(nn.Module):
    """
    Transform for converting video frames as a list of tensors.
    """

    def __init__(self, alpha=4):
        super().__init__()
        self.alpha = alpha

    def forward(self, frames):
        fast_pathway = frames
        # perform temporal sampling from the fast pathway.
        slow_pathway = torch.index_select(frames, 1,
                                          torch.linspace(0, frames.shape[1] - 1, frames.shape[1] // self.alpha).long())
        frame_list = [slow_pathway, fast_pathway]
        return frame_list


train_transform = ApplyTransformToKey(key="video", transform=Compose(
    [UniformTemporalSubsample(num_frames), Lambda(lambda x: x / 255.0), Normalize(mean, std),
     PackPathway()]))
test_transform = ApplyTransformToKey(key="video", transform=Compose(
    [UniformTemporalSubsample(num_frames), Lambda(lambda x: x / 255.0), Normalize(mean, std),
    PackPathway()]))
