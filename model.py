

import numpy as np
import torch
import math
import torch.nn.functional as F
import pickle


def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def linear_beta_schedule(timesteps, beta_start, beta_end):
    beta_start = beta_start
    beta_end = beta_end
    return torch.linspace(beta_start, beta_end, timesteps)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


def exp_beta_schedule(timesteps, beta_min=0.1, beta_max=10):
    x = torch.linspace(1, 2 * timesteps + 1, timesteps)
    betas = 1 - torch.exp(- beta_min / timesteps - x * 0.5 * (beta_max - beta_min) / (timesteps * timesteps))
    return betas


def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

class PointWiseFeedForward(torch.nn.Module):
    def __init__(self, hidden_units, dropout_rate):

        super(PointWiseFeedForward, self).__init__()

        self.conv1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, inputs):
        outputs = self.dropout2(self.conv2(self.relu(self.dropout1(self.conv1(inputs.transpose(-1, -2))))))
        outputs = outputs.transpose(-1, -2) # as Conv1D requires (N, C, Length)
        outputs += inputs
        return outputs

# pls use the following self-made multihead attention layer
# in case your pytorch version is below 1.16 or for other reasons
# https://github.com/pmixer/TiSASRec.pytorch/blob/master/model.py

class SASRec(torch.nn.Module):
    def __init__(self, item_num,args):
        super(SASRec, self).__init__()

        self.user_num = args.num_users
        self.item_num = item_num
        self.dev = args.device


        # TODO: loss += args.l2_emb for regularizing embedding vectors during training
        # https://stackoverflow.com/questions/42704283/adding-l1-l2-regularization-in-pytorch
        self.item_emb = torch.nn.Embedding(self.item_num + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = torch.nn.Embedding(args.max_len, args.hidden_units)  # TO IMPROVE
        self.emb_dropout = torch.nn.Dropout(p=args.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()  # to be Q for self-attention
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)

        for _ in range(args.num_blocks):
            new_attn_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = torch.nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(args.hidden_units, args.dropout_rate)
            self.forward_layers.append(new_fwd_layer)

            # self.pos_sigmoid = torch.nn.Sigmoid()
            # self.neg_sigmoid = torch.nn.Sigmoid()

    def get_item_embedding(self,seq):
        return self.item_emb(seq)

    def forward(self, log_seqs):  # for training
        seqs = self.item_emb(log_seqs.to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)

        timeline_mask = torch.BoolTensor(log_seqs.cpu() == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        tl = seqs.shape[1]  # time dim len for enforce causality
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))  # true shelter

        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs,
                                                      attn_mask=attention_mask)
            # key_padding_mask=timeline_mask
            # need_weights=False) this arg do not work?
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *= ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)  # (U, T, C) -> (U, -1, C)

        return log_feats


class DiffuSAS(torch.nn.Module):
    def __init__(self,args):
        super(DiffuSAS,self).__init__()
        self.args = args
        self.target_dataset=args.target_dataset
        self.temperature=args.temperature


        if self.target_dataset=="toy":
            self.Source_SASModel=SASRec(args.game_item_num,args).to(args.device)
            self.Target_SASModel=SASRec(args.toy_item_num,args).to(args.device)
            self.Mix_SASModel = SASRec(args.game_item_num + args.toy_item_num, args).to(args.device)
        elif self.target_dataset=="game":
            self.Target_SASModel = SASRec(args.game_item_num, args).to(args.device)
            self.Source_SASModel = SASRec(args.toy_item_num, args).to(args.device)
            self.Mix_SASModel = SASRec(args.game_item_num + args.toy_item_num, args).to(args.device)
        elif self.target_dataset == "douban_book":
            self.Source_SASModel = SASRec(args.douban_music_item_num, args).to(args.device)
            self.Target_SASModel = SASRec(args.douban_book_item_num, args).to(args.device)
            self.Mix_SASModel = SASRec(args.douban_music_item_num + args.douban_book_item_num, args).to(args.device)
        elif self.target_dataset == "douban_music":
            self.Target_SASModel = SASRec(args.douban_music_item_num, args).to(args.device)
            self.Source_SASModel = SASRec(args.douban_book_item_num, args).to(args.device)
            self.Mix_SASModel = SASRec(args.douban_music_item_num + args.douban_book_item_num, args).to(args.device)

        self.hidden_size = args.hidden_units
        self.interval = args.interval
        self.dev = args.device

        self.domain_mlp = torch.nn.Sequential(
            torch.nn.Linear(3 *self.hidden_size,self.hidden_size),
            torch.nn.Sigmoid(),
            torch.nn.Linear(self.hidden_size,self.hidden_size)
        )
        #
        # self.load_pretrain_weight()

        self.domain_cat=torch.nn.Sequential(
            torch.nn.Linear(3*self.hidden_size,1*self.hidden_size),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(1*self.hidden_size,self.hidden_size)
        )

        self.timesteps = args.timesteps
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end

        # self.w = w

        if args.beta_sche == 'linear':
            self.betas = linear_beta_schedule(timesteps=self.timesteps, beta_start=self.beta_start,
                                              beta_end=self.beta_end)
        elif args.beta_sche == 'exp':
            self.betas = exp_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == 'cosine':
            self.betas = cosine_beta_schedule(timesteps=self.timesteps)
        elif args.beta_sche == 'sqrt':
            self.betas = torch.tensor(betas_for_alpha_bar(self.timesteps, lambda t: 1 - np.sqrt(t + 0.0001), )).float()

        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (
                1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        self.w_q=torch.nn.Linear(self.hidden_size,self.hidden_size,bias=False)
        self.init(self.w_q)
        self.w_k=torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.init(self.w_k)
        self.w_v=torch.nn.Linear(self.hidden_size,self.hidden_size)
        self.init(self.w_v)
        self.ln=torch.nn.LayerNorm(self.hidden_size,elementwise_affine=False)

        #
        self.sigma = 0.01
        # self.users_mix_perferences=self.get_user_perference("mix_train")
        # self.users_mix_perferences=self.users_mix_perferences.detach()
        self.users_mix_perferences=self.get_split_mix_perference("train","mix_train")
        self.users_mix_perferences=self.users_mix_perferences.detach()
        self.top_k=10
        self.up_mix_p=False
        self.train_data_length=args.train_lengths
        self.begin_train=0
        self.w=args.w


    def init(self,m):
        if isinstance(m,torch.nn.Linear):
            torch.nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.constant_(m.bias,0)
        elif isinstance(m,torch.nn.Parameter):
            torch.nn.init.xavier_normal_(m)

    def load_pretrain_full_weight(self):
        print("load pretrain full weight...")
        # source_pretrain_file = "./data/" + self.args.dataset + "/ce_douban_music.pth"
        # target_pretrain_file = "./data/" + self.args.dataset + "/ce_douban_book.pth"
        # mix_pretrain_file = "./data/" + self.args.dataset + "/douban_music_book.pth"
        source_pretrain_file = "./data/" + self.args.dataset + "/ce_game.pth"
        target_pretrain_file = "./data/" + self.args.dataset + "/ce_toy.pth"
        mix_pretrain_file = "./data/" + self.args.dataset + "/"+ self.args.dataset +".pth"
        source_pretrain = torch.load(source_pretrain_file, weights_only=True)
        target_pretrain = torch.load(target_pretrain_file, weights_only=True)
        mix_pretrain=torch.load(mix_pretrain_file,weights_only=True)
        self.Mix_SASModel.load_state_dict(mix_pretrain)
        if self.target_dataset == "douban_book":
            self.Source_SASModel.load_state_dict(source_pretrain)
            self.Target_SASModel.load_state_dict(target_pretrain)
        elif self.target_dataset == "douban_music":
            self.Source_SASModel.load_state_dict(target_pretrain)
            self.Target_SASModel.load_state_dict(source_pretrain)

    def load_pretrain_embedding_weight(self):
        print("load pretrain embedding weight...")
        if self.args.target_dataset=="douban_book" or self.args.target_dataset=="douban_music":
            source_pretrain_file = "./data/" + self.args.dataset + "/ce_douban_music.pth"
            target_pretrain_file = "./data/" + self.args.dataset + "/ce_douban_book.pth"
            mix_pretrain_file = "./data/" + self.args.dataset + "/" + self.args.dataset +".pth"
        elif self.args.target_dataset=="game" or self.args.target_dataset=="toy":
            source_pretrain_file = "./data/" + self.args.dataset + "/ce_game.pth"
            target_pretrain_file = "./data/" + self.args.dataset + "/ce_toy.pth"
            mix_pretrain_file = "./data/" + self.args.dataset + "/" + self.args.dataset + ".pth"
        source_pretrain = torch.load(source_pretrain_file, weights_only=True)
        target_pretrain = torch.load(target_pretrain_file, weights_only=True)
        print("souce_pretrain:",source_pretrain.keys())
        mix_pretrain=torch.load(mix_pretrain_file,weights_only=True)
        self.Mix_SASModel.item_emb.weight.data.copy_(mix_pretrain["item_emb.weight"])
        if self.target_dataset == "toy":
            self.Source_SASModel.load_state_dict(source_pretrain)
            self.Target_SASModel.load_state_dict(target_pretrain)
        elif self.target_dataset == "game":
            self.Source_SASModel.load_state_dict(target_pretrain)
            self.Target_SASModel.load_state_dict(source_pretrain)
        elif self.target_dataset == "douban_book":
            self.Source_SASModel.item_emb.weight.data.copy_(source_pretrain["item_emb.weight"])
            # self.Source_SASModel.item_emb.weight.data.copy_(torch.randn(source_pretrain["item_emb.weight"].size()))
            self.Target_SASModel.item_emb.weight.data.copy_(target_pretrain["item_emb.weight"])
        elif self.target_dataset == "douban_music":
            self.Source_SASModel.item_emb.weight.data.copy_(target_pretrain["item_emb.weight"])
            # self.Source_SASModel.item_emb.weight.data.copy_(torch.randn(source_pretrain["item_emb.weight"].size()))
            self.Target_SASModel.item_emb.weight.data.copy_(source_pretrain["item_emb.weight"])

    def get_user_perference(self,domain_name):
        path_data = './data/' + self.args.dataset + '/dataset.pkl'
        with open(path_data, 'rb') as f:
            data_raw = pickle.load(f)
        users_perferences=[]
        for user,items in data_raw[domain_name].items():
            user_p=torch.mean(self.Mix_SASModel.item_emb(torch.LongTensor(items).to(self.dev)),dim=0)
            users_perferences.append(user_p)
        users_perferences=torch.stack(users_perferences)
        return users_perferences

    def get_sequence_distribution(self,sequence_embed):
        score=torch.matmul(sequence_embed,self.users_mix_perferences.t())
        _,top_score_k=torch.topk(score, k=self.top_k, dim=-1)
        top_k_perference=self.users_mix_perferences[top_score_k]

        sequence_mean=torch.mean(top_k_perference,dim=1)
        sequence_var=torch.var(top_k_perference,dim=1)

        noise=torch.randn(sequence_embed.size()).to(self.dev)
        return sequence_mean+torch.sqrt(sequence_var) * noise

    def get_split_mix_perference(self,target_domain,mix_domain):
        path_data = './data/' + self.args.dataset + '/dataset.pkl'
        with open(path_data, 'rb') as f:
            data_raw = pickle.load(f)
        user_mix_perference=[]
        for user_temp, seq_temp in data_raw[target_domain].items():
            mix_train_items = data_raw[mix_domain][user_temp]
            for star in range(len(seq_temp) - 1):
                last_target_item = seq_temp[star + 1]
                mix_train_index = np.argwhere(np.array(mix_train_items) == last_target_item)[-1].item()
                mix_items=mix_train_items[:mix_train_index + 1]
                mix_items_emb=self.Mix_SASModel.item_emb(torch.LongTensor(mix_items).to(self.dev))

                weight_x=self.GaussianTemporalAttention(len(mix_items_emb))
                g_mix_item_emb=torch.matmul(weight_x,mix_items_emb)
                user_mix_perference.append(g_mix_item_emb)
                # user_mix_perference.append(torch.mean(mix_items_emb,dim=0))
        user_mix_perference = torch.stack(user_mix_perference).squeeze(1)
        print("user_mix_perference:",user_mix_perference.size())
        return user_mix_perference

    def GaussianTemporalAttention(self, length):
        location=torch.arange(1,length+1).float()
        weight_location=1.5-1/(1+(location/length)**2)
        weight_location=weight_location.unsqueeze(0)
        return weight_location.to(self.dev)

    def get_val_split_mix_perference(self):
        path_data = './data/' + self.args.dataset + '/dataset.pkl'
        with open(path_data, 'rb') as f:
            data_raw = pickle.load(f)
        user_mix_perference=[]
        for user_temp, seq_temp in data_raw["train"].items():
            seq_temp+=data_raw["val"][user_temp]
            mix_train_items = data_raw["mix_train"][user_temp]+data_raw["mix_val"][user_temp]
            for star in range(len(seq_temp) - 1):
                last_target_item = seq_temp[star + 1]
                mix_train_index = np.argwhere(np.array(mix_train_items) == last_target_item)[-1].item()
                mix_items = mix_train_items[:mix_train_index + 1]
                mix_items_emb = self.Mix_SASModel.item_emb(torch.LongTensor(mix_items).to(self.dev))
                user_mix_perference.append(torch.mean(mix_items_emb, dim=0))
        user_mix_perference = torch.stack(user_mix_perference)
        print("user_mix_perference:",user_mix_perference.size())
        return user_mix_perference

    def get_test_split_mix_perference(self):
        path_data = './data/' + self.args.dataset + '/dataset.pkl'
        with open(path_data, 'rb') as f:
            data_raw = pickle.load(f)
        user_mix_perference=[]
        for user_temp, seq_temp in data_raw["train"].items():
            last_target_item=data_raw["val"][user_temp]
            mix_train_items = data_raw["mix_train"][user_temp]+data_raw["mix_val"][user_temp]
            mix_train_index = np.argwhere(np.array(mix_train_items) == last_target_item)[-1].item()
            mix_items = mix_train_items[:mix_train_index + 1]
            mix_items_emb = self.Mix_SASModel.item_emb(torch.LongTensor(mix_items).to(self.dev))
            user_mix_perference.append(torch.mean(mix_items_emb, dim=0))
        user_mix_perference = torch.stack(user_mix_perference)
        # print("user_mix_perference:",user_mix_perference.size())
        return user_mix_perference

    def get_val_user_mix_perference(self):
        path_data = './data/' + self.args.dataset + '/dataset.pkl'
        with open(path_data, 'rb') as f:
            data_raw = pickle.load(f)
        users_perferences = []
        for user, items in data_raw["mix_train"].items():
            items+=data_raw["mix_val"][user]
            user_p = torch.mean(self.Mix_SASModel.item_emb(torch.LongTensor(items).to(self.dev)), dim=0)
            users_perferences.append(user_p)
        users_perferences = torch.stack(users_perferences)
        return users_perferences


    def forward(self, target_seqs,source_seqs,mix_seqs):
        #ce loss
        target_log_feats = self.Target_SASModel(target_seqs)
        source_log_feats = self.Source_SASModel(source_seqs)
        mix_log_feats=self.Mix_SASModel(mix_seqs)

        self.begin_train+=1

        nxt_embedding=target_log_feats[:,-1,:]
        source_nxt_embedding=source_log_feats[:,-1,:]
        target_noise = self.get_sequence_distribution(nxt_embedding)
        source_noise=self.get_sequence_distribution(source_nxt_embedding)

        times_info = torch.randint(0, self.timesteps, (len(nxt_embedding),), device=self.dev).long()
        next_item_noise = self.q_sample(nxt_embedding, times_info, target_noise)
        source_next_item_noise=self.q_sample(source_nxt_embedding,times_info,source_noise)

        times_info_embedding = self.get_time_s(times_info)


        diffu_log_feats=torch.cat([next_item_noise.unsqueeze(1),source_next_item_noise.unsqueeze(1),
                                   mix_log_feats[:,-1,:].unsqueeze(1),times_info_embedding.unsqueeze(1)],dim=1)

        target_final_feats,source_final_feats,mean_final_feats=self.selfAttention(diffu_log_feats)

        u_p=(1-self.w)*(target_final_feats+source_final_feats)+self.w*nxt_embedding

        items_indices_embedding=self.Target_SASModel.item_emb.weight.t()
        return torch.matmul(u_p,items_indices_embedding)*self.temperature,F.mse_loss(target_final_feats,nxt_embedding)+F.mse_loss(source_final_feats,source_nxt_embedding)


    def predict(self, target_seqs, source_seqs,mix_seqs):  # for inference

        target_log_feats=self.Target_SASModel(target_seqs)
        source_log_feats = self.Source_SASModel(source_seqs)
        mix_log_feats = self.Mix_SASModel(mix_seqs)
        #diffusion reverse
        u_p = target_log_feats[:, -1, :]

        if self.up_mix_p:
            self.users_mix_perferences = self.get_val_split_mix_perference()
            self.users_mix_perferences = self.users_mix_perferences.detach()
            self.up_mix_p=False

        source_nxt_embedding = source_log_feats[:, -1, :]
        source_noise = self.get_sequence_distribution(source_nxt_embedding)
        target_noise = self.get_sequence_distribution(u_p)

        x_target = target_noise
        x_source=source_noise

        mix_info=mix_log_feats[:,-1,:]
        #
        for i in reversed(range(0, self.timesteps)):
            t = torch.tensor([i] * x_target.shape[0], dtype=torch.long).to(self.dev)
            # -----------------------
            times_info_embeddings = self.get_time_s(t)

            # --
            x_t_target = x_target
            x_t_source=x_source
            # x+=+times_info_embeddings
            diffu_log_feats = torch.cat([x_target.unsqueeze(1), x_source.unsqueeze(1),
                                         mix_info.unsqueeze(1),times_info_embeddings.unsqueeze(1)], dim=1)

            x_start_target,x_start_source,mean_final_feats=self.selfAttention(diffu_log_feats)
            # --

            model_mean_target = (
                    self.extract(self.posterior_mean_coef1, t, x_t_target.shape) * x_start_target +
                    self.extract(self.posterior_mean_coef2, t, x_t_target.shape) * x_t_target
            )

            model_mean_source = (
                    self.extract(self.posterior_mean_coef1, t, x_start_source.shape) * x_start_source +
                    self.extract(self.posterior_mean_coef2, t, x_start_source.shape) * x_t_source
            )

            if i == 0:
                x_target = model_mean_target
                x_source=model_mean_source
            else:
                # ---
                posterior_variance_t = self.extract(self.posterior_variance, t, x_target.shape)
                noise_target = torch.randn_like(x_target)
                noise_source=torch.randn_like(x_source)

                x_target = model_mean_target + torch.sqrt(posterior_variance_t) * noise_target
                x_source=model_mean_source + torch.sqrt(posterior_variance_t) * noise_source
                # x = model_mean
        #---

        items_indices_embedding = self.Target_SASModel.item_emb.weight
        x=(1-self.w)*(x_target+x_source)+self.w*u_p

        logits = x.matmul(items_indices_embedding.transpose(0, 1))
        return logits*self.temperature  # preds # (U, I)

    def selfAttention(self,features):
        features=self.ln(features)
        q=self.w_q(features)
        k=self.w_k(features)
        v=self.w_v(features)

        attn=q.mul(self.hidden_size**-0.5) @ k.transpose(-1,-2)
        attn=attn.softmax(dim=-1)

        features=attn @ v
        # print("features.size:",features.size())
        # y=features
        # print("y.size:",y.size())

        return features[:,0,:],features[:,1,:],torch.mean(features,dim=1)

    def get_time_s(self, time):
        half_dim = self.args.hidden_units // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=self.dev) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

    def extract(self, a, t, x_shape):
        # res = a.to(device=t.device)[t].float()
        # while len(res.shape) < len(x_shape):
        #     res = res[..., None]
        # return res.expand(x_shape)
        # batch_size = t.shape[0]
        # out = a.gather(-1, t.cpu())
        # return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(self.dev)
        res = a.to(device=t.device)[t].float()
        while len(res.shape) < len(x_shape):
            res = res[..., None]
        return res.expand(x_shape)

    def r_extract(self, a, t, x_shape):
        res = torch.from_numpy(a).to(device=t.device)[t].float()
        while len(res.shape) < len(x_shape):
            res = res[..., None]
        return res.expand(x_shape)

    def betas_for_alpha_bar(self, num_diffusion_timesteps, alpha_bar, max_beta=0.999):

        betas = []
        for i in range(num_diffusion_timesteps):
            t1 = i / num_diffusion_timesteps
            t2 = (i + 1) / num_diffusion_timesteps
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
        return np.array(betas)

    def q_sample(self, x_start, t, noise=None):
        # print(self.betas)
        if noise is None:
            noise = torch.randn_like(x_start)
            # noise = torch.randn_like(x_start) / 100
        sqrt_alphas_cumprod_t = self.extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self.extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        # mean=torch.mean(x_start,dim=0)
        # noise=mean+(x_start - mean).pow(2).mean().sqrt()*noise
        # return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise*torch.sign(x_start)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    @torch.no_grad()
    def p_sample(self, x, t, t_index, diffuer_mlp):
        times_info = torch.tensor([t_index] * x.shape[0], dtype=torch.long).to(self.dev)
        times_info_embeddings = self.get_time_s(times_info)
        x_start = diffuer_mlp(torch.cat([x, times_info_embeddings], dim=-1))
        x_t = x

        model_mean = (
                self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = self.extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)

            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def denoise(self, times_info, x_T, diffuer_mlp):
        # x = self.q_sample(x, times_info)
        # x=torch.randn_like(condtion_info)
        for i in reversed(range(0, self.timesteps)):
            x_T = self.p_sample(x_T, torch.full((times_info.shape[0],), i, device=self.dev, dtype=torch.long), i,diffuer_mlp)
        return x_T

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:
            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
                self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
                + self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )  ## \mu_t
        assert (posterior_mean.shape[0] == x_start.shape[0])
        return posterior_mean

    def p_mean_variance(self, seqs_emb, x_t, t,seqs):
        times_info_embeddings = self.get_time_s(t)
        x_t+=times_info_embeddings
        lambda_uncertainty = torch.normal(mean=torch.full(seqs_emb.shape, self.lambda_uncertainty),
                                    std=torch.full(seqs_emb.shape, self.lambda_uncertainty)).to(x_t.device)  ## distribution
        seqs_emb = seqs_emb + lambda_uncertainty * x_t.unsqueeze(1)

        model_output = self.SASModel(seqs_emb, seqs)

        x_0 = model_output[:,-1,:]  ##output predict
        # x_0 = self._predict_xstart_from_eps(x_t, t, model_output)  ## eps predict

        model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
        model_log_variance = self.r_extract(model_log_variance, t, x_t.shape)

        model_mean = self.q_posterior_mean_variance(x_start=x_0, x_t=x_t,
                                                    t=t)  ## x_start: candidante item embedding, x_t: inputseq_embedding + outseq_noise, output x_(t-1) distribution
        return model_mean, model_log_variance

    def pp_sample(self, item_rep, noise_x_t, t,seqs):
        model_mean, model_log_variance = self.p_mean_variance(item_rep, noise_x_t, t,seqs)
        noise = torch.randn_like(noise_x_t)
        nonzero_mask = ((t != 0).float().view(-1, *([1] * (len(noise_x_t.shape) - 1))))  # no noise when t == 0
        sample_xt = model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick
        # sample_xt = model_mean  ## sample x_{t-1} from the \mu(x_{t-1}) distribution based on the reparameter trick

        return sample_xt

    def reverse_p_sample(self, item_rep, noise_x_t,seqs):
        device = self.dev
        indices = list(range(self.timesteps))[::-1]

        for i in indices:  # from T to 0, reversion iteration
            t = torch.tensor([i] * item_rep.shape[0], device=device)
            with torch.no_grad():
                noise_x_t = self.pp_sample(item_rep, noise_x_t, t,seqs)
        return noise_x_t
