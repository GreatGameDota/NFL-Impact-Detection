import torch
from torch import nn, Tensor

from torch.nn.modules.utils import _pair
from torch.jit.annotations import List, BroadcastingList2

from torchvision_modified.extension import _assert_has_ops
from ._utils import convert_boxes_to_roi_format, check_roi_boxes_shape


def roi_pool(
    input: Tensor,
    boxes: Tensor,
    output_size: BroadcastingList2[int],
    spatial_scale: float = 1.0,
) -> Tensor:
    """
    Performs Region of Interest (RoI) Pool operator described in Fast R-CNN

    Arguments:
        input (Tensor[N, C, H, W]): input tensor
        boxes (Tensor[K, 5] or List[Tensor[L, 4]]): the box coordinates in (x1, y1, x2, y2)
            format where the regions will be taken from. If a single Tensor is passed,
            then the first column should contain the batch index. If a list of Tensors
            is passed, then each Tensor will correspond to the boxes for an element i
            in a batch
        output_size (int or Tuple[int, int]): the size of the output after the cropping
            is performed, as (height, width)
        spatial_scale (float): a scaling factor that maps the input coordinates to
            the box coordinates. Default: 1.0

    Returns:
        output (Tensor[K, C, output_size[0], output_size[1]])
    """
    _assert_has_ops()
    check_roi_boxes_shape(boxes)
    rois = boxes
    output_size = _pair(output_size)
    if not isinstance(rois, torch.Tensor):
        rois = convert_boxes_to_roi_format(rois)
    output, _ = torch.ops.torchvision.roi_pool(input, rois, spatial_scale,
                                               output_size[0], output_size[1])
    return output


class RoIPool(nn.Module):
    """
    See roi_pool
    """
    def __init__(self, output_size: BroadcastingList2[int], spatial_scale: float):
        super(RoIPool, self).__init__()
        self.output_size = output_size
        self.spatial_scale = spatial_scale

    def forward(self, input: Tensor, rois: Tensor) -> Tensor:
        return roi_pool(input, rois, self.output_size, self.spatial_scale)

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + '('
        tmpstr += 'output_size=' + str(self.output_size)
        tmpstr += ', spatial_scale=' + str(self.spatial_scale)
        tmpstr += ')'
        return tmpstr
