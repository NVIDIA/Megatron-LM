# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
import torch.nn.functional as F
import torch
from megatron.training import print_rank_0, get_args
from megatron.core import mpu
from megatron.legacy.data.vit_dataset import ClassificationTransform
from megatron.legacy.data.image_folder import ImageFolder

_FEATURE_BANK = None


def build_data_loader(dataset, drop_last=True, shuffle=False):
    """Data loader. Note that batch-size is the local (per GPU) batch-size."""
    # Sampler.
    args = get_args()
    micro_batch_size = 16
    num_workers = args.num_workers
    world_size = mpu.get_data_parallel_world_size()
    rank = mpu.get_data_parallel_rank()
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        drop_last=drop_last, shuffle=shuffle
    )

    # Data loader. Note that batch size is the per GPU batch size.
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=micro_batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        drop_last=not drop_last,
        pin_memory=True,
    )
    return data_loader


def compute_feature_bank(model):
    args = get_args()
    global _FEATURE_BANK
    feature_bank = []
    feature_label = []

    train_ds = ImageFolder(
        root=args.data_path[0],
        transform=ClassificationTransform((args.img_h, args.img_w), train=False),
        data_per_class_fraction=1.0
    )
    classes = len(train_ds.classes)
    dataloader = build_data_loader(train_ds)
     
    for m in model:
        m.eval()

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch[0].cuda().contiguous()
            labels = batch[1].cuda().contiguous()
            student_feature, teacher_feature = model[0](images)
            feature = F.normalize(teacher_feature.float(), dim=1)
            feature_bank.append(feature)
            feature_label.append(labels)
    
    for m in model:
        m.train()

    # [N', D]
    feature_bank = torch.cat(feature_bank, dim=0).contiguous()
    feature_label = torch.cat(feature_label, dim=0).contiguous()

    feature_banks = [torch.zeros_like(feature_bank)
                     for i in range(mpu.get_data_parallel_world_size())]
    torch.distributed.all_gather(feature_banks,
                                 feature_bank,
                                 group=mpu.get_data_parallel_group())

    assert torch.all(torch.eq(feature_banks[mpu.get_data_parallel_rank()],
                              feature_bank))

    feature_labels = [torch.zeros_like(feature_label)
                      for i in range(mpu.get_data_parallel_world_size())]
    torch.distributed.all_gather(feature_labels,
                                 feature_label,
                                 group=mpu.get_data_parallel_group())

    # [D, N]
    feature_banks = torch.cat(feature_banks, dim=0).t().contiguous()
    # [N]
    feature_labels = torch.cat(feature_labels, dim=0).contiguous()
    print_rank_0("feature_banks size is {}".format(feature_banks.size()))
    print_rank_0("feature labels size is {}".format(feature_labels.size()))

    _FEATURE_BANK = (feature_banks, feature_labels, classes)


def get_feature_bank():
    global _FEATURE_BANK
    assert _FEATURE_BANK is not None
    return _FEATURE_BANK


# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and
# https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1),
                              dim=-1,
                              index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k,
                                classes,
                                device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1,
                                          index=sim_labels.view(-1, 1),
                                          value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
            one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1),
            dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels
