import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import torch.nn.functional as F
import json
import time

def train(
        model,
        config,
        train_loader,
        valid_loader=None,
        valid_epoch_interval=20,
        foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)

                loss.backward()

                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best cate_loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )

    if foldername != "":
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, 1)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j: j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], 1)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def calc_JS_Divergence(target, forecast):
    B, L, K = target.shape
    t = target.reshape(B, -1)
    f = forecast.reshape(B, -1)
    M = (t + f) * 0.5
    M = M.log_softmax(-1)
    t = t.softmax(-1)
    f = f.softmax(-1)
    j = (0.5 * F.kl_div(M, t, reduction='mean') + 0.5 * F.kl_div(M, f, reduction='mean'))

    return j


def calc_spatial_JS_Divergence(target, forecast):
    t = target.sum(dim=2)
    f = forecast.sum(dim=2)
    M = (t + f) * 0.5
    M = M.log_softmax(-1)
    t = t.softmax(-1)
    f = f.softmax(-1)
    j = (0.5 * F.kl_div(M, t, reduction='mean') + 0.5 * F.kl_div(M, f, reduction='mean'))

    return j


def calc_type_JS_Divergence(target, forecast):
    B, L, K = target.shape
    target = list(target.split(1, dim=-1))
    forecast = list(forecast.split(1, dim=-1))

    jsd = []
    for k in range(K):
        t = target[k].squeeze(2)
        f = forecast[k].squeeze(2)
        M = (t + f) * 0.5
        M = M.log_softmax(-1)
        t = t.softmax(-1)
        f = f.softmax(-1)
        j = (0.5 * F.kl_div(M, t, reduction='mean') + 0.5 * F.kl_div(M, f, reduction='mean'))
        jsd.append(j.sum().item())

    return sum(jsd) / len(jsd)


def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
    with torch.no_grad():
        model.eval()

        all_app_id_targets = []
        all_app_id_samples = []

        # cate level
        cate_mse_total = 0
        cate_mae_total = 0
        cate_evalpoints_total = 0
        cate_all_jsd = []
        all_spatial_jsd = []
        all_type_jsd = []
        cate_all_mtv = 0
        cate_all_targets = []
        cate_all_samples = []

        # app_id level
        app_mse_total = 0
        app_mae_total = 0
        app_evalpoints_total = 0
        app_all_jsd = []
        app_all_mtv = 0
        app_all_targets = []
        app_all_samples = []

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        with open('data/shanghai/app2category.json', 'r') as f:
            app2cate = json.load(f)

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):

                start_time = time.time()
                output = model.evaluate(test_batch, nsample)
                end_time = time.time()
                print(f"程序运行时间：{end_time - start_time:.4f} 秒")

                app_id_samples, app_id_targets = output


                app_id_samples = app_id_samples.mean(dim=1)
                app_id_samples = torch.argmax(app_id_samples, dim=-1)

                all_app_id_targets.append(app_id_targets)
                all_app_id_samples.append(app_id_samples)

                B = app_id_targets.shape[0]
                cate_evalpoints_total += B
                app_evalpoints_total += B

                # id_mask = torch.zeros(app_id_targets.shape).to(device)
                # nonzero_index = torch.where(app_id_targets)
                # id_mask[nonzero_index] = 1
                # all_app_id_samples.append((app_ids * id_mask))

                app_id_samples = np.array(app_id_samples.cpu())
                app_id_targets = np.array(app_id_targets.cpu())

                # app level
                app_sample = []
                app_target = []
                for b in range(B):
                    v, c = np.unique(app_id_targets[b], return_counts=True)
                    dic = dict(zip(v, c))
                    tmp = []
                    for i in range(2001):
                        tmp.append(dic[i] if i in dic.keys() else 0)
                    tmp = torch.tensor(tmp).to(device).float()
                    app_target.append(tmp.unsqueeze(0))
                    v, c = np.unique(app_id_samples[b], return_counts=True)
                    dic = dict(zip(v, c))
                    tmp = []
                    for i in range(2001):
                        tmp.append(dic[i] if i in dic.keys() else 0)
                    tmp = torch.tensor(tmp).to(device).float()
                    app_sample.append(tmp.unsqueeze(0))
                app_target = torch.cat(app_target, dim=0)
                app_sample = torch.cat(app_sample, dim=0)

                app_all_targets.append(app_target)
                app_all_samples.append(app_sample)

                mse_current = ((app_sample - app_target) ** 2) * (scaler ** 2)
                mae_current = (torch.abs((app_sample - app_target))) * scaler

                app_mse_total += mse_current.sum().item()
                app_mae_total += mae_current.sum().item()

                M = (app_target + app_sample) * 0.5
                M = M.log_softmax(-1)
                jsd_current = (
                        0.5 * F.kl_div(M, app_target.softmax(-1), reduction='mean') + 0.5 * F.kl_div(M, app_sample.softmax(-1), reduction='mean'))
                app_all_jsd.append(jsd_current.sum().item())

                mtv_current = 0.5 * (torch.abs((app_sample - app_target))) * scaler
                app_all_mtv += mtv_current.sum().item()

                # cate level
                cate_sample = []
                cate_target = []
                for b in range(B):
                    values = [app2cate[str(a)] if a != 0 else 20 for a in app_id_targets[b]]
                    # a = app_id_targets[b]
                    # values = [app2cate[str(int(a))] if a != 0 else 20]
                    v, c = np.unique(values, return_counts=True)
                    dic = dict(zip(v, c))
                    tmp = []
                    for i in range(21):
                        tmp.append(dic[i] if i in dic.keys() else 0)
                    tmp = torch.tensor(tmp).to(device).float()
                    cate_target.append(tmp.unsqueeze(0))

                    values = [app2cate[str(a)] if a != 0 else 20 for a in app_id_samples[b]]
                    # a = app_id_samples[b]
                    # values = [app2cate[str(int(a))] if a != 0 else 20]
                    v, c = np.unique(values, return_counts=True)
                    dic = dict(zip(v, c))
                    tmp = []
                    for i in range(21):
                        tmp.append(dic[i] if i in dic.keys() else 0)
                    tmp = torch.tensor(tmp).to(device).float()
                    cate_sample.append(tmp.unsqueeze(0))

                cate_target = torch.cat(cate_target, dim=0)
                cate_sample = torch.cat(cate_sample, dim=0)

                cate_all_targets.append(cate_target)
                cate_all_samples.append(cate_sample)

                mse_current = ((cate_sample - cate_target) ** 2) * (scaler ** 2)
                mae_current = (torch.abs((cate_sample - cate_target))) * scaler

                cate_mse_total += mse_current.sum().item()
                cate_mae_total += mae_current.sum().item()

                M = (cate_target + cate_sample) * 0.5
                M = M.log_softmax(-1)
                jsd_current = (
                        0.5 * F.kl_div(M, cate_target.softmax(-1), reduction='mean') +
                        0.5 * F.kl_div(M, cate_sample.softmax(-1), reduction='mean')
                )
                cate_all_jsd.append(jsd_current.sum().item())

                mtv_current = 0.5 * (torch.abs((cate_sample - cate_target))) * scaler
                cate_all_mtv += mtv_current.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "batch_no": batch_no
                    },
                    refresh=True,
                )

            with open(
                    foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                all_app_id_targets = torch.cat(all_app_id_targets, dim=0)
                all_app_id_samples = torch.cat(all_app_id_samples, dim=0)

                pickle.dump(
                    [
                        all_app_id_targets,
                        all_app_id_samples,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )
            app_all_targets = torch.cat(app_all_targets, dim=0)
            app_all_samples = torch.cat(app_all_samples, dim=0)
            app_CRPS = calc_quantile_CRPS(app_all_targets, app_all_samples.unsqueeze(1).expand(-1, nsample, -1),
                                          mean_scaler, scaler)

            cate_all_targets = torch.cat(cate_all_targets, dim=0)
            cate_all_samples = torch.cat(cate_all_samples, dim=0)
            cate_CRPS = calc_quantile_CRPS(cate_all_targets, cate_all_samples.unsqueeze(1).expand(-1, nsample, -1),
                                        mean_scaler, scaler)

            print("App Level:\n")
            print("RMSE:", np.sqrt(app_mse_total / app_evalpoints_total))
            print("MAE:", app_mae_total / app_evalpoints_total)
            print("CRPS:", app_CRPS)
            print("JSD:", sum(app_all_jsd) / len(app_all_jsd))
            print("M-TV:", app_all_mtv / app_evalpoints_total)

            print("\n")
            print("Cate Level:\n")
            print("RMSE:", np.sqrt(cate_mse_total / cate_evalpoints_total))
            print("MAE:", cate_mae_total / cate_evalpoints_total)
            print("CRPS:", cate_CRPS)
            print("JSD:", sum(cate_all_jsd) / len(cate_all_jsd))
            print("M-TV:", cate_all_mtv / cate_evalpoints_total)
