import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diff_models import diff_model
from Modules import *


class soft_attn(nn.Module):
    def __init__(self, dim):
        super(soft_attn, self).__init__()

        self.weight1 = nn.Linear(dim, dim, bias=False)
        self.weight2 = nn.Linear(dim, dim, bias=True)
        self.weight_c = nn.Linear(dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, L, emb_dim = x.shape
        mean_embed = torch.mean(x, dim=1).unsqueeze(1).expand(-1, L, -1)  # (B,L,dim)
        attn = self.weight1(mean_embed) + self.weight2(x)
        attn = self.sigmoid(attn)
        attn = self.weight_c(attn)  # (B,L,1)

        out = torch.matmul(attn.permute(0, 2, 1), x).squeeze(1)

        return out


class model_base(nn.Module):
    def __init__(self, config, device, app_emb_all, loc_emb_dic=None):
        super().__init__()
        self.device = device

        # app id encoder
        self.app_emb_all = app_emb_all
        self.app_emb_dim = 16

        # parameters for condition
        self.emb_time_dim = config["model"]["timeemb"]
        self.loc_emb_all = loc_emb_dic
        self.emb_loc_dim = 32
        # self.emb_loc_dim = 0

        self.cond_total_dim = self.emb_time_dim + self.emb_loc_dim
        self.total_dim = self.emb_time_dim + self.emb_loc_dim + self.app_emb_dim

        # hidden state
        self.self_attention = EncoderLayer(
            self.total_dim,
            self.total_dim,
            4,
            self.total_dim,
            self.total_dim,
            dropout=0.01
        )
        self.soft_attention = soft_attn(self.total_dim)

        # parameters for diffusion models
        config_diff = config["diffusion"]
        config_diff["appemb_dim"] = self.app_emb_dim
        config_diff["cond_dim"] = self.cond_total_dim
        config_diff["total_dim"] = self.total_dim

        input_dim = 1
        self.diffmodel = diff_model(config_diff, input_dim)

        self.num_steps = config_diff["num_steps"]
        if config_diff["schedule"] == "quad":
            self.beta = np.linspace(
                config_diff["beta_start"] ** 0.5, config_diff["beta_end"] ** 0.5, self.num_steps
            ) ** 2
        elif config_diff["schedule"] == "linear":
            self.beta = np.linspace(
                config_diff["beta_start"], config_diff["beta_end"], self.num_steps
            )

        self.alpha = 1 - self.beta
        self.alpha_hat = np.cumprod(self.alpha)
        self.alpha_torch = torch.tensor(self.alpha_hat).float().to(self.device).unsqueeze(1)

        # loss function
        self.loss_1 = nn.MSELoss()
        self.loss_2 = nn.MSELoss(reduce=False)

    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], d_model).to(self.device)
        position = pos.unsqueeze(1)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def app_embedding(self, app_ids):
        B, L = app_ids.shape
        app_emb = torch.cat([
            torch.cat([
                torch.index_select(self.app_emb_all, dim=0, index=app_ids[b][l]) for l in range(L)],
                dim=0).unsqueeze(0) for b in range(B)],
            dim=0)
        return app_emb  # (B,L,emb_dim)

    def loc_embedding(self, loc_ids):
        B, L = loc_ids.shape
        loc_emb = torch.cat([
            torch.cat([
                torch.index_select(self.loc_emb_all, dim=0, index=loc_ids[b][l]) for l in range(L)],
                dim=0).unsqueeze(0) for b in range(B)],
            dim=0)
        return loc_emb  # (B,L,emb_dim)

    def get_ST_info(self, tp, app_emb, loc):
        B, emb_dim = app_emb.shape
        if len(loc.shape) != 2:
            loc = loc.unsqueeze(1)

        time_embed = self.time_embedding(tp, self.emb_time_dim)  # (B,time_emb_dim)

        loc_embed = self.loc_embedding(loc)  # (B,loc_emb_dim)

        time_embed = time_embed.unsqueeze(1).expand(-1, emb_dim, -1)  # (B,emb_dim,tim_emb_dim)
        loc_embed = loc_embed.expand(-1, emb_dim, -1)  # (B,emb_dim,loc_emb_dim)

        side_info = torch.cat([time_embed, loc_embed], dim=-1)  # (B,emb_dim,total_side_dim)

        side_info = side_info.permute(0, 2, 1)  # (B,*,emb_dim)

        return side_info

    def get_hist_info(self, app_emb, tp, loc):
        _, L, emb_dim = app_emb.shape
        if len(loc.shape) != 2:
            loc = loc.unsqueeze(1)

        time_embed = self.time_embedding(tp, self.emb_time_dim).unsqueeze(1).expand(-1, L, -1)

        loc_embed = self.loc_embedding(loc)

        hist_info = torch.cat([time_embed, loc_embed, app_emb], dim=-1)
        # hist_info = app_emb

        hist_info, _ = self.self_attention(hist_info)
        hist_info = hist_info[:, -1, :].unsqueeze(1)
        # if L > 10:
        #     hist_info = hist_info[:, L - 10: L, :]
        # hist_info = self.soft_attention(hist_info).unsqueeze(1)

        return hist_info

    def set_input_to_diffmodel(self, noisy_data):
        total_input = noisy_data.unsqueeze(1)  # (B,1,*,*)
        return total_input

    def calc_loss_valid(self, app_emb, tp, locs, is_train):
        loss_sum = 0
        for t in range(self.num_steps):  # calculate cate_loss for all t
            loss = self.calc_loss(app_emb, tp, locs, is_train, set_t=t)
            loss_sum += loss.detach()
        return loss_sum / self.num_steps

    def calc_loss(self, app_embs, tp, locs, is_train, set_t=-1):

        B, L, emb_dim = app_embs.shape
        loss = 0.0

        pre_app_emb = torch.zeros_like(app_embs[:, 0]).unsqueeze(1).to(self.device)
        # pre_app_emb = app_embs[:, :L-10]
        for l in range(0, L):
            app_emb = app_embs[:, l]
            if is_train != 1:  # for validation
                t = (torch.ones(B) * set_t).long().to(self.device)
            else:
                t = torch.randint(0, self.num_steps, [B]).to(self.device)
            current_alpha = self.alpha_torch[t]  # (B,1,1)
            noise = torch.randn_like(app_emb)
            noisy_data = (current_alpha ** 0.5) * app_emb + (1.0 - current_alpha) ** 0.5 * noise

            # if l == 0:
            #     hist_info = self.get_hist_info(pre_app_emb, tp, locs[:, l])
            # else:
            #     hist_info = self.get_hist_info(pre_app_emb, tp, locs[:, : l])
            hist_info = None

            ST_info = self.get_ST_info(tp, app_emb, locs[:, l])

            total_input = self.set_input_to_diffmodel(noisy_data)  # (B,1,emb)
            predict_noise = self.diffmodel(total_input, hist_info, ST_info, t)  # (B,emb)

            predict_app_emb = (noisy_data - (1.0 - current_alpha) ** 0.5 * predict_noise) / (current_alpha ** 0.5)

            if l == 0:
                pre_app_emb = app_emb.unsqueeze(1)
            else:
                pre_app_emb = torch.cat([pre_app_emb, app_emb.unsqueeze(1)], dim=1)

            loss_1 = self.loss_1(noise, predict_noise)
            loss_2 = self.loss_2(predict_app_emb, app_emb).mean(axis=1)

            alpha = 0.1
            lambda_t = 1 - alpha * t / self.num_steps
            loss += loss_1 + (lambda_t * loss_2).mean().item()

        return loss

    def impute(self, app_embs, tp, locs, n_samples):

        B, L, emb = app_embs.shape
        app_id_samples = torch.zeros(B, n_samples, 10, 2001).to(self.device)

        for i in range(n_samples):

            pre_app_predict = torch.zeros_like(app_embs[:, 0]).unsqueeze(1).to(self.device)
            # pre_app_predict = app_embs[:, :L - 10]

            for l in range(0, L):
                app_emb = app_embs[:, l]
                current_sample = torch.randn_like(app_emb)

                if l == 0:
                    hist_info = self.get_hist_info(pre_app_predict, tp, locs[:, l])
                else:
                    hist_info = self.get_hist_info(pre_app_predict, tp, locs[:, : l])

                ST_info = self.get_ST_info(tp, app_emb, locs[:, l])

                for t in range(self.num_steps - 1, -1, -1):
                    diff_input = current_sample
                    diff_input = diff_input.unsqueeze(1)  # (B,1,emb_dim)

                    predicted = self.diffmodel(diff_input, hist_info, ST_info, torch.tensor([t]).to(self.device))

                    coeff1 = 1 / self.alpha_hat[t] ** 0.5
                    coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                    current_sample = coeff1 * (current_sample - coeff2 * predicted)

                    if t > 0:
                        noise = torch.randn_like(current_sample)
                        sigma = ((1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]) ** 0.5
                        current_sample += sigma * noise

                if l == 0:
                    pre_app_predict = app_emb.unsqueeze(1)
                else:
                    pre_app_predict = torch.cat([pre_app_predict, current_sample.unsqueeze(1)], dim=1)

                app_ids_predict = F.cosine_similarity(current_sample.unsqueeze(-2), self.app_emb_all, dim=-1)
                app_id_samples[:, i, l] = app_ids_predict.detach()

        return app_id_samples

    def forward(self, batch, is_train=1):
        (
            # cate_observed_data,
            tp,
            locs,
            # app_observed_data,
            app_ids
        ) = self.process_data(batch)

        app_emb = self.app_embedding(app_ids)

        loss_func = self.calc_loss if is_train == 1 else self.calc_loss_valid

        return loss_func(app_emb, tp, locs, is_train)

    def evaluate(self, batch, n_samples):
        (
            # cate_observed_data,
            tp,
            locs,
            # app_observed_data,
            app_ids
        ) = self.process_data(batch)

        with torch.no_grad():
            B, L =app_ids.shape
            app_emb = self.app_embedding(app_ids)

            app_id_samples = self.impute(app_emb, tp, locs, n_samples)

        return app_id_samples, app_ids


class CSDI_Shanghai(model_base):
    def __init__(self, config, device, app_emb_all, loc_emb_dic):
        super(CSDI_Shanghai, self).__init__(config, device, app_emb_all, loc_emb_dic)

    def process_data(self, batch):
        # cate_observed_data = batch["cate_observed_data"].to(self.device).float()  # (B,N,L,K)
        observed_tp = batch["timepoints"].to(self.device).float()  # (B,L)
        observed_locs = batch["locations"].to(self.device)  # (B,L,K)
        # app_observed_data = batch["app_observed_data"].to(self.device).float()  # (B,L,3,R)
        app_ids = batch["app_ID"].to(self.device)

        return (
            # cate_observed_data,
            observed_tp,
            observed_locs,
            # app_observed_data,
            app_ids
        )


class CSDI_Nanchang(model_base):
    def __init__(self, config, device, app_emb_all, loc_emb_dic):
        super(CSDI_Nanchang, self).__init__(config, device, app_emb_all, loc_emb_dic)

    def process_data(self, batch):
        # cate_observed_data = batch["cate_observed_data"].to(self.device).float()  # (B,N,L,K)
        observed_tp = batch["timepoints"].to(self.device).float()  # (B,L)
        # observed_locs = batch["locations"].to(self.device)  # (B,L,K)
        # app_observed_data = batch["app_observed_data"].to(self.device).float()  # (B,L,3,R)
        app_ids = batch["app_ID"].to(self.device)

        return (
            # cate_observed_data,
            observed_tp,
            None,
            # app_observed_data,
            app_ids
        )
