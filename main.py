import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig
from torch.utils.data import DataLoader
import random
import numpy as np
from src.models.evflownet import AttentionUNet
from src.datasets import DatasetProvider, train_collate
from enum import Enum, auto
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Any
import os
import time
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau

class RepresentationType(Enum):
    VOXEL = auto()
    STEPAN = auto()

def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)

def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
    epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
    return epe

def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
    np.save(f"{file_name}.npy", flow.cpu().numpy())

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    scaler = GradScaler()  # AMP用のスケーラーを初期化

    for inputs in dataloader:
        if "flow_gt" not in inputs or inputs["flow_gt"] is None:
            print("Skipping batch: flow_gt is None")
            continue  # 'flow_gt' が存在しない場合、次のバッチへ

        event_image = torch.cat([inputs["event_volume"], inputs["event_volume_next"]], dim=1).to(device).float()
        ground_truth_flow = inputs["flow_gt"].to(device).float()[:, :2, :, :]  # チャンネル数を2に制限

        optimizer.zero_grad()

        with autocast():  # 自動混合精度のコンテキストマネージャを使用
            intermediate_outputs = model(event_image)

            loss = 0.0
            for output in intermediate_outputs:
                scaled_targets = nn.functional.interpolate(ground_truth_flow, size=output.shape[2:], mode='bilinear', align_corners=False)
                loss += criterion(output, scaled_targets)

        scaler.scale(loss).backward()  # AMPを使用したスケーリング付きのバックプロパゲーション
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    return total_loss / len(dataloader)


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(args: DictConfig):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.enabled = False

    loader = DatasetProvider(
        dataset_path=Path(args.dataset_path),
        representation_type=RepresentationType.VOXEL,
        delta_t_ms=100,
        num_bins=4
    )
    train_set = loader.get_train_dataset()
    test_set = loader.get_test_dataset()

    train_batch_size = max(1, args.data_loader.train.batch_size // 2)
    test_batch_size = max(1, args.data_loader.test.batch_size // 2)

    train_data = DataLoader(train_set,
                            batch_size=train_batch_size,
                            shuffle=args.data_loader.train.shuffle,
                            collate_fn=train_collate,
                            drop_last=False)
    test_data = DataLoader(test_set,
                           batch_size=test_batch_size,
                           shuffle=args.data_loader.test.shuffle,
                           collate_fn=train_collate,
                           drop_last=False)

    model = AttentionUNet(img_ch=8, output_ch=2).to(device)  # インスタンス作成時の引数を修正

    optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    criterion = nn.SmoothL1Loss()

    for epoch in range(args.train.epochs):
        print("on epoch: {}".format(epoch+1))
        train_loss = train_one_epoch(model, train_data, optimizer, criterion, device)
        print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}')
        
        scheduler.step(train_loss)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    
    current_time = time.strftime("%Y%m%d%H%M%S")
    model_path = f"checkpoints/model_{current_time}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    flow: torch.Tensor = torch.tensor([]).to(device)
    with torch.no_grad():
        print("start test")
        for batch in tqdm(test_data):
            batch: Dict[str, Any]
            event_image = torch.cat([batch["event_volume"], batch["event_volume_next"]], dim=1).to(device).float()  # テストフェーズでもチャンネル数を8に設定
            batch_flow = model(event_image)[-1]  # リストの最後の要素を取得
            flow = torch.cat((flow, batch_flow), dim=0)
        print("test done")

    file_name = "submission"
    save_optical_flow_to_npy(flow, file_name)

if __name__ == "__main__":
    main()


# import torch
# import torch.nn as nn  
# import hydra
# from omegaconf import DictConfig
# from torch.utils.data import DataLoader
# import random
# import numpy as np
# from src.models.evflownet import EVFlowNet
# from src.datasets import DatasetProvider
# from enum import Enum, auto
# from src.datasets import train_collate
# from tqdm import tqdm
# from pathlib import Path
# from typing import Dict, Any
# import os
# import time


# class RepresentationType(Enum):
#     VOXEL = auto()
#     STEPAN = auto()

# def set_seed(seed):
#     random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     np.random.seed(seed)

# def compute_epe_error(pred_flow: torch.Tensor, gt_flow: torch.Tensor):
#     '''
#     end-point-error (ground truthと予測値の二乗誤差)を計算
#     pred_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 予測したオプティカルフローデータ
#     gt_flow: torch.Tensor, Shape: torch.Size([B, 2, 480, 640]) => 正解のオプティカルフローデータ
#     '''
#     epe = torch.mean(torch.mean(torch.norm(pred_flow - gt_flow, p=2, dim=1), dim=(1, 2)), dim=0)
#     return epe

# def save_optical_flow_to_npy(flow: torch.Tensor, file_name: str):
#     '''
#     optical flowをnpyファイルに保存
#     flow: torch.Tensor, Shape: torch.Size([2, 480, 640]) => オプティカルフローデータ
#     file_name: str => ファイル名
#     '''
#     np.save(f"{file_name}.npy", flow.cpu().numpy())

# # def train_one_epoch(model, dataloader, optimizer, criterion, device):
# #     model.train()
# #     total_loss = 0.0

# #     for inputs in dataloader:
# #         event_image = inputs["event_volume"].to(device).float()  # データ型を Float に変換
# #         ground_truth_flow = inputs["flow_gt"].to(device).float()  # データ型を Float に変換

# #         optimizer.zero_grad()
# #         flow_dict, intermediate_outputs = model(event_image)

# #         # 複数スケールでの損失計算
# #         loss = 0.0
# #         for i, output in enumerate(intermediate_outputs):
# #             # 出力サイズにターゲットをリサイズ
# #             scaled_targets = nn.functional.interpolate(ground_truth_flow, size=output.shape[2:], mode='bilinear', align_corners=False)
# #             loss += criterion(output, scaled_targets)

# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()

# #     return total_loss / len(dataloader)

# def train_one_epoch(model, dataloader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0.0

#     for inputs in dataloader:
#         event_image = inputs["event_volume"].to(device).float()  # Floatに変換
#         event_image_next = inputs["event_volume_next"].to(device).float()  # Floatに変換
#         ground_truth_flow = inputs["flow_gt"].to(device).float()  # Floatに変換

#         optimizer.zero_grad()
#         flow_dict, intermediate_outputs = model({'event_volume': event_image, 'event_volume_next': event_image_next})

#         # 複数スケールでの損失計算
#         loss = 0.0
#         for i, output in enumerate(intermediate_outputs):
#             scaled_targets = nn.functional.interpolate(ground_truth_flow, size=output.shape[2:], mode='bilinear', align_corners=False)
#             loss += criterion(output, scaled_targets)

#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     return total_loss / len(dataloader)


# @hydra.main(version_base=None, config_path="configs", config_name="base")
# def main(args: DictConfig):
#     set_seed(args.seed)
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     '''
#         ディレクトリ構造:

#         data
#         ├─test
#         |  ├─test_city
#         |  |    ├─events_left
#         |  |    |   ├─events.h5
#         |  |    |   └─rectify_map.h5
#         |  |    └─forward_timestamps.txt
#         └─train
#             ├─zurich_city_11_a
#             |    ├─events_left
#             |    |       ├─ events.h5
#             |    |       └─ rectify_map.h5
#             |    ├─ flow_forward
#             |    |       ├─ 000134.png
#             |    |       |.....
#             |    └─ forward_timestamps.txt
#             ├─zurich_city_11_b
#             └─zurich_city_11_c
#         '''
    
#     # ------------------
#     #    Dataloader
#     # ------------------
#     loader = DatasetProvider(
#         dataset_path=Path(args.dataset_path),
#         representation_type=RepresentationType.VOXEL,
#         delta_t_ms=100,
#         num_bins=4
#     )
#     train_set = loader.get_train_dataset()
#     test_set = loader.get_test_dataset()
#     collate_fn = train_collate
#     train_data = DataLoader(train_set,
#                                  batch_size=args.data_loader.train.batch_size,
#                                  shuffle=args.data_loader.train.shuffle,
#                                  collate_fn=collate_fn,
#                                  drop_last=False)
#     test_data = DataLoader(test_set,
#                                  batch_size=args.data_loader.test.batch_size,
#                                  shuffle=args.data_loader.test.shuffle,
#                                  collate_fn=collate_fn,
#                                  drop_last=False)

#     '''
#     train data:
#         Type of batch: Dict
#         Key: seq_name, Type: list
#         Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
#         Key: flow_gt, Type: torch.Tensor, Shape: torch.Size([Batch, 2, 480, 640]) => オプティカルフローデータのバッチ
#         Key: flow_gt_valid_mask, Type: torch.Tensor, Shape: torch.Size([Batch, 1, 480, 640]) => オプティカルフローデータのvalid. ベースラインでは使わない
    
#     test data:
#         Type of batch: Dict
#         Key: seq_name, Type: list
#         Key: event_volume, Type: torch.Tensor, Shape: torch.Size([Batch, 4, 480, 640]) => イベントデータのバッチ
#     '''
#     # ------------------
#     #       Model
#     # ------------------
#     model = EVFlowNet(args.train).to(device)

#     # ------------------
#     #   optimizer
#     # ------------------
#     optimizer = torch.optim.Adam(model.parameters(), lr=args.train.initial_learning_rate, weight_decay=args.train.weight_decay)
#     # ------------------
#     #   Start training
#     # ------------------
#     model.train()
#     # for epoch in range(args.train.epochs):
#     #     total_loss = 0
#     #     print("on epoch: {}".format(epoch+1))
#     #     for i, batch in enumerate(tqdm(train_data)):
#     #         batch: Dict[str, Any]
#     #         event_image = batch["event_volume"].to(device) # [B, 4, 480, 640]
#     #         ground_truth_flow = batch["flow_gt"].to(device) # [B, 2, 480, 640]
#     #         flow = model(event_image) # [B, 2, 480, 640]
#     #         loss: torch.Tensor = compute_epe_error(flow, ground_truth_flow)
#     #         print(f"batch {i} loss: {loss.item()}")
#     #         optimizer.zero_grad()
#     #         loss.backward()
#     #         optimizer.step()

#     #         total_loss += loss.item()
#     #     print(f'Epoch {epoch+1}, Loss: {total_loss / len(train_data)}')

#     # 損失関数の定義
#     criterion = nn.MSELoss()

#     for epoch in range(args.train.epochs):
#         print("on epoch: {}".format(epoch+1))
#         train_loss = train_one_epoch(model, train_data, optimizer, criterion, device)
#         print(f'Epoch {epoch+1}, Loss: {train_loss:.4f}')

    
#     # current_time = time.strftime("%Y%m%d%H%M%S")
#     # model_path = f"checkpoints/model_{current_time}.pth"
#     # torch.save(model.state_dict(), model_path)
#     # print(f"Model saved to {model_path}")

#     # Create the directory if it doesn't exist
#     if not os.path.exists('checkpoints'):
#         os.makedirs('checkpoints')
    
#     current_time = time.strftime("%Y%m%d%H%M%S")
#     model_path = f"checkpoints/model_{current_time}.pth"
#     torch.save(model.state_dict(), model_path)
#     print(f"Model saved to {model_path}")

#     # ------------------
#     #   Start predicting
#     # ------------------
#     # model.load_state_dict(torch.load(model_path, map_location=device))
#     # model.eval()
#     # flow: torch.Tensor = torch.tensor([]).to(device)
#     # with torch.no_grad():
#     #     print("start test")
#     #     for batch in tqdm(test_data):
#     #         batch: Dict[str, Any]
#     #         event_image = batch["event_volume"].to(device)
#     #         batch_flow = model(event_image) # [1, 2, 480, 640]
#     #         flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
#     #     print("test done")


#    # 予測の開始
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model.eval()
#     flow: torch.Tensor = torch.tensor([]).to(device)

#     # 予測ループ
#     with torch.no_grad():
#         print("start test")
#         for batch in tqdm(test_data):
#             batch: Dict[str, Any]
#             event_image = batch["event_volume"].to(device)
#             event_image_next = batch["event_volume_next"].to(device)
#             _, batch_flow = model({'event_volume': event_image, 'event_volume_next': event_image_next})
#             batch_flow = batch_flow[-1]
#             flow = torch.cat((flow, batch_flow), dim=0)
#         print("test done")

#     # with torch.no_grad():
#     #     print("start test")
#     #     for batch in tqdm(test_data):
#     #         batch: Dict[str, Any]
#     #         event_image = batch["event_volume"].to(device)
#     #         _, batch_flow = model(event_image)  # 必要な出力のみを取得
#     #         batch_flow = batch_flow[-1]  # 最後のスケールのフローを使用する場合
#     #         flow = torch.cat((flow, batch_flow), dim=0)  # [N, 2, 480, 640]
#     #     print("test done")

#     # ------------------
#     #  save submission
#     # ------------------
#     file_name = "submission"
#     save_optical_flow_to_npy(flow, file_name)

# if __name__ == "__main__":
#     main()
