from typing import Any, Callable, Optional, Dict

import torch
from torchmetrics import Metric
from torchmetrics.classification.accuracy import Accuracy


class mIoU(Metric):
    full_state_update: Optional[bool] = False
    higher_is_better: Optional[bool] = False

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super(mIoU, self).__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
        )
        self.add_state("miou", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(
        self,
        outputs: Dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> None:
        """
        outputs: [A, T, 2]
        target: [A, T, 2]
        """
        with torch.no_grad():
            occ_mask, target_occ_mask = outputs["occ_mask"], outputs["target_occ_mask"]
            
            if occ_mask is not None:       
                occ_mask_pred = (occ_mask.sigmoid() > 0.5)
                intersection = torch.logical_and(target_occ_mask, occ_mask_pred).float().sum((2, 3))
                union = torch.logical_or(target_occ_mask, occ_mask_pred).float().sum((2, 3))

                iou = intersection / (union + 1e-6)  # 1e-6은 0으로 나누는 것을 방지하기 위한 작은 값
                miou = iou.mean()  # 평균 IoU 계산 (mIoU)

                self.miou += miou
                self.count += 1
            else:
                self.miou += 0
                self.count += 1
                

    def compute(self) -> torch.Tensor:
        return self.miou / self.count
