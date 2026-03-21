# train_optimized.py
import os
import argparse
import mindspore as ms
from mindspore import save_checkpoint, load_checkpoint, load_param_into_net
from mindspore.nn import Adam, CrossEntropyLoss, MSELoss

# 导入你的模块
from utils.dataset import create_ultra_dataset
from models.ultra_net import UniversalExoNet


# ====== Ascend 专用上下文配置 ======
ms.set_context(
    mode=ms.GRAPH_MODE,
    device_target="Ascend",
    max_call_depth=10000,
    memory_optimize_level="O1",
    graph_kernel_flags="--disable_expand_ops=Softmax --enable_parallel_fusion=true"
)


def get_param_groups(net, weight_decay=1e-4):
    """将 BatchNorm 和 bias 参数排除在 weight decay 之外"""
    decay_params = []
    no_decay_params = []
    for param in net.trainable_params():
        if len(param.shape) == 1 or param.name.endswith(".bias") or "bn" in param.name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]


def main(args):
    print("📂 Loading datasets...")
    train_data_path = os.path.join(args.data_dir, "ultra_train.npz")
    val_data_path = os.path.join(args.data_dir, "ultra_val.npz")

    # Ascend 建议 num_parallel_workers=1 避免冲突
    train_ds = create_ultra_dataset(
        data_path=train_data_path,
        batch_size=args.batch_size,
        shuffle=True,
        num_parallel_workers=1,
        load_traj=True
    )
    val_ds = create_ultra_dataset(
        data_path=val_data_path,
        batch_size=args.batch_size,
        shuffle=False,
        num_parallel_workers=1,
        load_traj=True
    )

    model = UniversalExoNet(
        input_channels=39,
        seq_len=50,
        num_intents=args.num_intents,
        traj_dim=72
    )

    # 初始化训练状态
    start_epoch = 0
    best_val_loss = float('inf')

    # 恢复训练（如果指定）
    if args.resume:
        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        print(f"🔁 Loading checkpoint: {args.resume}")
        param_dict = load_checkpoint(args.resume)
        load_param_into_net(model, param_dict)
        # 尝试恢复训练状态
        if "epoch" in param_dict:
            start_epoch = int(param_dict["epoch"]) + 1
        if "best_val_loss" in param_dict:
            best_val_loss = float(param_dict["best_val_loss"])
        print(f"   Resumed from epoch {start_epoch}, best val loss: {best_val_loss:.6f}")

    cls_criterion = CrossEntropyLoss()
    traj_criterion = MSELoss()

    param_group = get_param_groups(model, weight_decay=1e-4)
    optimizer = Adam(param_group, learning_rate=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n🚀 Training on Ascend with batch_size={args.batch_size}, lr={args.lr}")
    print(f"   Traj weight: {args.traj_weight} | Epochs: {args.epochs}\n")

    for epoch in range(start_epoch, args.epochs):
        model.set_train()
        total_loss = total_cls = total_traj = steps = 0

        for batch in train_ds.create_tuple_iterator():
            x, y_cls, y_traj = batch

            def forward_fn(x, y_cls, y_traj):
                logits, traj_pred = model(x)
                loss_cls = cls_criterion(logits, y_cls)
                loss_traj = traj_criterion(traj_pred, y_traj)
                loss = loss_cls + args.traj_weight * loss_traj
                return loss, (loss_cls, loss_traj)

            grad_fn = ms.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)
            (loss, (loss_cls, loss_traj)), grads = grad_fn(x, y_cls, y_traj)
            optimizer(grads)

            total_loss += loss.asnumpy()
            total_cls += loss_cls.asnumpy()
            total_traj += loss_traj.asnumpy()
            steps += 1

            # 每 10 步打印一次，避免日志刷屏但能及时监控
            if steps % 10 == 0:
                print(f"  Step {steps}: loss={loss.asnumpy():.4f} "
                      f"(cls={loss_cls.asnumpy():.4f}, traj={loss_traj.asnumpy():.4f})")

        avg_train_loss = total_loss / steps
        avg_cls_loss = total_cls / steps
        avg_traj_loss = total_traj / steps

        # Validation
        model.set_train(False)
        val_loss = 0
        val_steps = 0
        for x, y_cls, y_traj in val_ds.create_tuple_iterator():
            logits, traj_pred = model(x)
            loss_cls = cls_criterion(logits, y_cls)
            loss_traj = traj_criterion(traj_pred, y_traj)
            loss = loss_cls + args.traj_weight * loss_traj
            val_loss += loss.asnumpy()
            val_steps += 1
        avg_val_loss = val_loss / val_steps

        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train: {avg_train_loss:.4f} (cls={avg_cls_loss:.4f}, traj={avg_traj_loss:.4f}) "
              f"Val: {avg_val_loss:.4f}")

        # 保存当前 epoch checkpoint（含训练状态）
        ckpt_file = os.path.join(args.output_dir, f"model_epoch_{epoch+1}.ckpt")
        save_checkpoint(
            model,
            ckpt_file,
            append_dict={"epoch": epoch, "best_val_loss": best_val_loss}
        )
        print(f"💾 Saved checkpoint: {ckpt_file}")

        # 保存 best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_ckpt = os.path.join(args.output_dir, "best_model.ckpt")
            save_checkpoint(
                model,
                best_ckpt,
                append_dict={"epoch": epoch, "best_val_loss": best_val_loss}
            )
            print(f"🎉 New best model saved to {best_ckpt}")

    print(f"\n✅ Training finished. Best Val Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Ultra-MoCap Model on Ascend")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_intents", type=int, default=5)
    parser.add_argument("--traj_weight", type=float, default=0.1)
    parser.add_argument("--resume", type=str, default="", 
                        help="Path to checkpoint to resume from (e.g., ./outputs/model_epoch_5.ckpt)")
    args = parser.parse_args()

    main(args)