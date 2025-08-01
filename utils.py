import sys
import copy
from cProfile import label

import torch.utils.data as data_utils
import torch
import numpy as np

# target sequence: [17634, 13941, 29326, 1689, 3858, 30684, 4767, 3823, 8259]
# mix sequence: [17634, 49261, 45945, 41816, 13941, 40598, 46271, 45007, 45688, 49032, 47349, 29326, 46893, 48112, 42718, 37938, 1689, 43635, 46095, 45112, 48446, 3858, 30684, 38825, 4767, 45360, 45360, 46954, 48369, 45910, 39895, 48231, 37898, 3823, 42699, 8259, 39247, 40747]
# target_dataset.keys: dict_keys(['train', 'val', 'test', 'umap', 'smap', 'mix_train', 'mix_val', 'mix_test'])
# train: [17634, 13941, 29326, 1689, 3858, 30684, 4767]
# val: [3823]
# test: [8259]
# mix_train: [17634, 49261, 45945, 41816, 13941, 40598, 46271, 45007, 45688, 49032, 47349, 29326, 46893, 48112, 42718, 37938, 1689, 43635, 46095, 45112, 48446, 3858, 30684, 38825, 4767]
# mix_val: [45360, 45360, 46954, 48369, 45910, 39895, 48231, 37898, 3823]
# mix_test: [42699, 8259]

#amazon toy-game 37868
#douban book-music 50448
class TrainDataset(data_utils.Dataset):
    def __init__(self, id2seq, source_id2_seq,mix_id2_seq,max_len):
        self.id2seq = id2seq
        self.source_id2_seq=source_id2_seq
        self.mix_id2_seq=mix_id2_seq
        self.max_len = max_len

    def __len__(self):
        return len(self.id2seq)

    def __getitem__(self, index):
        target_seq = self._getseq(index)
        labels = [target_seq[-1]]
        target_tokens = target_seq[:-1]
        target_tokens = target_tokens[-self.max_len:]
        mask_len = self.max_len - len(target_tokens)
        target_tokens = [0] * mask_len + target_tokens
        #--
        mix_tokens=self.mix_id2_seq[index]
        mix_tokens=mix_tokens[-self.max_len:]
        mix_mask_len=self.max_len-len(mix_tokens)
        mix_tokens=[0]*mix_mask_len+mix_tokens
        #--
        #--
        source_tokens=self.source_id2_seq[index]
        source_tokens=source_tokens[-self.max_len:]
        source_mask_len=self.max_len-len(source_tokens)
        source_tokens=[0]*source_mask_len+source_tokens
        return torch.LongTensor(target_tokens), torch.LongTensor(source_tokens),torch.LongTensor(mix_tokens),torch.LongTensor(labels)

    def _getseq(self, idx):
        return self.id2seq[idx]


# data[user:[item1,item2,...,itemn]
class Data_Train():
    def __init__(self, target_data_train, mix_data_train,args):
        self.u2seq = target_data_train
        self.max_len = args.max_len
        self.batch_size = args.batch_size
        self.mix_data_train=mix_data_train
        self.interval=args.interval
        self.target_dataset=args.target_dataset
        self.split_onebyone()

    def split_onebyone(self):
        self.id_seq = {}
        self.id_seq_user = {}
        self.source_id_seq={}
        self.mix_id_seq={}
        idx = 0
        if self.target_dataset=="toy":
            for user_temp, seq_temp in self.u2seq.items():
                # for star in range(len(seq_temp) - 1):
                #     self.id_seq[idx] = seq_temp[:star + 2]
                #     self.id_seq_user[idx] = user_temp
                #     idx += 1
                mix_train_items=self.mix_data_train[user_temp]
                for star in range(len(seq_temp)-1):
                    last_target_item=seq_temp[star+1]
                    mix_train_index = np.argwhere(np.array(mix_train_items) == last_target_item)[-1].item()
                    #--
                    self.id_seq[idx]=seq_temp[:star+2]
                    #--
                    self.mix_id_seq[idx] = mix_train_items[:mix_train_index]
                    #--
                    source_sequence=[]
                    for i in range(mix_train_index):
                        if mix_train_items[i]>self.interval:
                            source_sequence.append(mix_train_items[i]-self.interval)
                    self.source_id_seq[idx]=source_sequence
                    self.id_seq_user[idx]=user_temp
                    #--

                    idx+=1
        elif self.target_dataset=="game":
            for user_temp, seq_temp in self.u2seq.items():

                mix_train_items = self.mix_data_train[user_temp]
                for star in range(len(seq_temp) - 1):
                    last_target_item = seq_temp[star + 1]
                    mix_train_index = np.argwhere(np.array(mix_train_items) == last_target_item)[-1].item()
                    # --
                    self.id_seq[idx] = [ii-self.interval for ii in seq_temp[:star + 2]]
                    # --
                    self.mix_id_seq[idx] = mix_train_items[:mix_train_index]
                    # --
                    source_sequence = []
                    for i in range(mix_train_index):
                        if mix_train_items[i] <= self.interval:
                            source_sequence.append(mix_train_items[i])
                    self.source_id_seq[idx] = source_sequence
                    self.id_seq_user[idx] = user_temp
                    # --

                    idx += 1
        elif self.target_dataset=="douban_book":
            for user_temp, seq_temp in self.u2seq.items():
                # for star in range(len(seq_temp) - 1):
                #     self.id_seq[idx] = seq_temp[:star + 2]
                #     self.id_seq_user[idx] = user_temp
                #     idx += 1
                mix_train_items=self.mix_data_train[user_temp]
                for star in range(len(seq_temp)-1):
                    last_target_item=seq_temp[star+1]
                    mix_train_index = np.argwhere(np.array(mix_train_items) == last_target_item)[-1].item()
                    #--
                    self.id_seq[idx]=seq_temp[:star+2]
                    #--
                    self.mix_id_seq[idx] = mix_train_items[:mix_train_index]
                    #--
                    source_sequence=[]
                    for i in range(mix_train_index):
                        if mix_train_items[i]>self.interval:
                            source_sequence.append(mix_train_items[i]-self.interval)
                    self.source_id_seq[idx]=source_sequence
                    self.id_seq_user[idx]=user_temp
                    #--

                    idx+=1
        elif self.target_dataset=="douban_music":
            for user_temp, seq_temp in self.u2seq.items():

                mix_train_items = self.mix_data_train[user_temp]
                for star in range(len(seq_temp) - 1):
                    last_target_item = seq_temp[star + 1]
                    mix_train_index = np.argwhere(np.array(mix_train_items) == last_target_item)[-1].item()
                    # --
                    self.id_seq[idx] = [ii-self.interval for ii in seq_temp[:star + 2]]
                    # --
                    self.mix_id_seq[idx] = mix_train_items[:mix_train_index]
                    # --
                    source_sequence = []
                    for i in range(mix_train_index):
                        if mix_train_items[i] <= self.interval:
                            source_sequence.append(mix_train_items[i])
                    self.source_id_seq[idx] = source_sequence
                    self.id_seq_user[idx] = user_temp
                    # --

                    idx += 1


    def get_pytorch_dataloaders(self):
        dataset = TrainDataset(self.id_seq, self.source_id_seq,self.mix_id_seq,self.max_len)
        return data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True)


class ValDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2answer, mix_train_dataset,mix_val_dataset,max_len,interval,target_dataset):
        self.u2seq = u2seq
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.mix_val_dataset=mix_val_dataset
        self.mix_train_dataset=mix_train_dataset
        self.max_len = max_len
        self.target_dataset=target_dataset
        self.interval=interval

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        if self.target_dataset=="toy":
            user = self.users[index]
            target_seq = self.u2seq[user]
            answer = self.u2answer[user]
            target_tokens = target_seq[-self.max_len:]
            padding_len = self.max_len - len(target_tokens)
            target_tokens = [0] * padding_len + target_tokens
            #
            mix_tokens=self.mix_train_dataset[user]+self.mix_val_dataset[user][:-1]
            mix_tokens=mix_tokens[-self.max_len:]
            mix_mask_len=self.max_len-len(mix_tokens)
            mix_tokens=[0]*mix_mask_len+mix_tokens
            #
            source_seq = []
            for i in self.mix_train_dataset[user]:
                if i>self.interval:
                    source_seq.append(i-self.interval)
            for i in self.mix_val_dataset[user]:
                if i>self.interval:
                    source_seq.append(i-self.interval)
            source_tokens = source_seq[-self.max_len:]
            source_mask_len = self.max_len - len(source_tokens)
            source_tokens = [0] * source_mask_len + source_tokens
        elif self.target_dataset=="game":
            user = self.users[index]
            target_seq = [ii-self.interval for ii in self.u2seq[user]]
            answer = [ii - self.interval for ii in self.u2answer[user]]
            target_tokens = target_seq[-self.max_len:]
            padding_len = self.max_len - len(target_tokens)
            target_tokens = [0] * padding_len + target_tokens
            #
            mix_tokens = self.mix_train_dataset[user] + self.mix_val_dataset[user][:-1]
            mix_tokens = mix_tokens[-self.max_len:]
            mix_mask_len = self.max_len - len(mix_tokens)
            mix_tokens = [0] * mix_mask_len + mix_tokens
            #
            source_seq = []
            for i in self.mix_train_dataset[user]:
                if i <= self.interval:
                    source_seq.append(i)
            for i in self.mix_val_dataset[user]:
                if i <= self.interval:
                    source_seq.append(i)
            source_tokens = source_seq[-self.max_len:]
            source_mask_len = self.max_len - len(source_tokens)
            source_tokens = [0] * source_mask_len + source_tokens
        elif self.target_dataset=="douban_book":
            user = self.users[index]
            target_seq = self.u2seq[user]
            answer = self.u2answer[user]
            target_tokens = target_seq[-self.max_len:]
            padding_len = self.max_len - len(target_tokens)
            target_tokens = [0] * padding_len + target_tokens
            #
            mix_tokens = self.mix_train_dataset[user] + self.mix_val_dataset[user][:-1]
            mix_tokens = mix_tokens[-self.max_len:]
            mix_mask_len = self.max_len - len(mix_tokens)
            mix_tokens = [0] * mix_mask_len + mix_tokens
            #
            source_seq = []
            for i in self.mix_train_dataset[user]:
                if i > self.interval:
                    source_seq.append(i - self.interval)
            for i in self.mix_val_dataset[user]:
                if i > self.interval:
                    source_seq.append(i - self.interval)
            source_tokens = source_seq[-self.max_len:]
            source_mask_len = self.max_len - len(source_tokens)
            source_tokens = [0] * source_mask_len + source_tokens
        elif self.target_dataset=="douban_music":
            user = self.users[index]
            target_seq = [ii-self.interval for ii in self.u2seq[user]]
            answer = [ii - self.interval for ii in self.u2answer[user]]
            target_tokens = target_seq[-self.max_len:]
            padding_len = self.max_len - len(target_tokens)
            target_tokens = [0] * padding_len + target_tokens
            #
            mix_tokens = self.mix_train_dataset[user] + self.mix_val_dataset[user][:-1]
            mix_tokens = mix_tokens[-self.max_len:]
            mix_mask_len = self.max_len - len(mix_tokens)
            mix_tokens = [0] * mix_mask_len + mix_tokens
            #
            source_seq = []
            for i in self.mix_train_dataset[user]:
                if i <= self.interval:
                    source_seq.append(i)
            for i in self.mix_val_dataset[user]:
                if i <= self.interval:
                    source_seq.append(i)
            source_tokens = source_seq[-self.max_len:]
            source_mask_len = self.max_len - len(source_tokens)
            source_tokens = [0] * source_mask_len + source_tokens


        return torch.LongTensor(target_tokens), torch.LongTensor(source_tokens),torch.LongTensor(mix_tokens),torch.LongTensor(answer)


class Data_Val():
    def __init__(self, data_train, data_val, mix_train_dataset,mix_val_dataset,args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2answer = data_val
        self.max_len = args.max_len
        self.mix_val_dataset=mix_val_dataset
        self.mix_train_dataset=mix_train_dataset
        self.interval=args.interval
        self.target_dataset = args.target_dataset

    def get_pytorch_dataloaders(self):
        dataset = ValDataset(self.u2seq, self.u2answer, self.mix_train_dataset,self.mix_val_dataset,self.max_len,self.interval,self.target_dataset)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader


class TestDataset(data_utils.Dataset):
    def __init__(self, u2seq, u2_seq_add, u2answer,mix_train_dataset,mix_val_dataset,mix_test_dataset, max_len,interval,target_dataset):
        self.u2seq = u2seq
        self.u2seq_add = u2_seq_add
        self.users = sorted(self.u2seq.keys())
        self.u2answer = u2answer
        self.mix_train_dataset = mix_train_dataset
        self.mix_val_dataset = mix_val_dataset
        self.mix_test_dataset=mix_test_dataset
        self.max_len = max_len
        self.interval=interval
        self.target_dataset = target_dataset

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        if self.target_dataset=="toy":
            user = self.users[index]
            target_seq = self.u2seq[user] + self.u2seq_add[user]
            # seq = self.u2seq[user]
            answer = self.u2answer[user]
            target_tokens = target_seq[-self.max_len:]
            padding_len = self.max_len - len(target_tokens)
            target_tokens = [0] * padding_len + target_tokens
            #
            mix_tokens = self.mix_train_dataset[user] + self.mix_val_dataset[user]+self.mix_test_dataset[user][:-1]
            mix_tokens = mix_tokens[-self.max_len:]
            mix_mask_len = self.max_len - len(mix_tokens)
            mix_tokens = [0] * mix_mask_len + mix_tokens
            #
            source_seq = []
            for i in self.mix_train_dataset[user]:
                if i > self.interval:
                    source_seq.append(i-self.interval)
            for i in self.mix_val_dataset[user]:
                if i > self.interval:
                    source_seq.append(i-self.interval)
            for i in self.mix_test_dataset[user]:
                if i>self.interval:
                    source_seq.append(i-self.interval)
            source_tokens = source_seq[-self.max_len:]
            source_mask_len = self.max_len - len(source_tokens)
            source_tokens = [0] * source_mask_len + source_tokens
        elif self.target_dataset=="game":
            user = self.users[index]
            target_seq = [ii-self.interval for ii in self.u2seq[user] + self.u2seq_add[user]]
            # seq = self.u2seq[user]
            answer = [ii-self.interval for ii in self.u2answer[user]]
            target_tokens = target_seq[-self.max_len:]
            padding_len = self.max_len - len(target_tokens)
            target_tokens = [0] * padding_len + target_tokens
            #
            mix_tokens = self.mix_train_dataset[user] + self.mix_val_dataset[user] + self.mix_test_dataset[user][:-1]
            mix_tokens = mix_tokens[-self.max_len:]
            mix_mask_len = self.max_len - len(mix_tokens)
            mix_tokens = [0] * mix_mask_len + mix_tokens
            #
            source_seq = []
            for i in self.mix_train_dataset[user]:
                if i <= self.interval:
                    source_seq.append(i)
            for i in self.mix_val_dataset[user]:
                if i <= self.interval:
                    source_seq.append(i)
            for i in self.mix_test_dataset[user]:
                if i <= self.interval:
                    source_seq.append(i)
            source_tokens = source_seq[-self.max_len:]
            source_mask_len = self.max_len - len(source_tokens)
            source_tokens = [0] * source_mask_len + source_tokens
        elif self.target_dataset=="douban_book":
            user = self.users[index]
            target_seq = self.u2seq[user] + self.u2seq_add[user]
            # seq = self.u2seq[user]
            answer = self.u2answer[user]
            target_tokens = target_seq[-self.max_len:]
            padding_len = self.max_len - len(target_tokens)
            target_tokens = [0] * padding_len + target_tokens
            #
            mix_tokens = self.mix_train_dataset[user] + self.mix_val_dataset[user]+self.mix_test_dataset[user][:-1]
            mix_tokens = mix_tokens[-self.max_len:]
            mix_mask_len = self.max_len - len(mix_tokens)
            mix_tokens = [0] * mix_mask_len + mix_tokens
            #
            source_seq = []
            for i in self.mix_train_dataset[user]:
                if i > self.interval:
                    source_seq.append(i-self.interval)
            for i in self.mix_val_dataset[user]:
                if i > self.interval:
                    source_seq.append(i-self.interval)
            for i in self.mix_test_dataset[user]:
                if i>self.interval:
                    source_seq.append(i-self.interval)
            source_tokens = source_seq[-self.max_len:]
            source_mask_len = self.max_len - len(source_tokens)
            source_tokens = [0] * source_mask_len + source_tokens
        elif self.target_dataset=="douban_music":
            user = self.users[index]
            target_seq = [ii-self.interval for ii in self.u2seq[user] + self.u2seq_add[user]]
            # seq = self.u2seq[user]
            answer = [ii-self.interval for ii in self.u2answer[user]]
            target_tokens = target_seq[-self.max_len:]
            padding_len = self.max_len - len(target_tokens)
            target_tokens = [0] * padding_len + target_tokens
            #
            mix_tokens = self.mix_train_dataset[user] + self.mix_val_dataset[user] + self.mix_test_dataset[user][:-1]
            mix_tokens = mix_tokens[-self.max_len:]
            mix_mask_len = self.max_len - len(mix_tokens)
            mix_tokens = [0] * mix_mask_len + mix_tokens
            #
            source_seq = []
            for i in self.mix_train_dataset[user]:
                if i <= self.interval:
                    source_seq.append(i)
            for i in self.mix_val_dataset[user]:
                if i <= self.interval:
                    source_seq.append(i)
            for i in self.mix_test_dataset[user]:
                if i <= self.interval:
                    source_seq.append(i)
            source_tokens = source_seq[-self.max_len:]
            source_mask_len = self.max_len - len(source_tokens)
            source_tokens = [0] * source_mask_len + source_tokens

        return torch.LongTensor(target_tokens), torch.LongTensor(source_tokens),torch.LongTensor(mix_tokens),torch.LongTensor(answer)


class Data_Test():
    def __init__(self, data_train, data_val, data_test,mix_train_dataset,mix_val_dataset,mix_test_dataset, args):
        self.batch_size = args.batch_size
        self.u2seq = data_train
        self.u2seq_add = data_val
        self.u2answer = data_test
        self.mix_train_dataset=mix_train_dataset
        self.mix_val_dataset=mix_val_dataset
        self.mix_test_dataset=mix_test_dataset
        self.max_len = args.max_len
        self.interval=args.interval
        self.target_dataset = args.target_dataset

    def get_pytorch_dataloaders(self):
        dataset = TestDataset(self.u2seq, self.u2seq_add, self.u2answer,self.mix_train_dataset,self.mix_val_dataset,
                              self.mix_test_dataset,self.max_len,self.interval,self.target_dataset)
        dataloader = data_utils.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, pin_memory=True)
        return dataloader

# TODO: merge evaluate functions for test and val set

def dcg(hit):
    log2 = torch.log2(torch.arange(1, hit.size()[-1] + 1) + 1).unsqueeze(0)
    rel = (hit/log2).sum(dim=-1)
    return rel

# evaluate on test set
def evaluate(model, test_dataloader, args):
    K_can = [1, 3, 5, 10, 20, 50, 100]
    total_NDCG = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    total_HT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    num_batch = 0.0
    for test_batch in test_dataloader:
        seq=test_batch[0]
        source_seqs=test_batch[1]
        mix_seqs=test_batch[2]
        label=test_batch[3]
        predictions = model.predict(seq,source_seqs,mix_seqs)
        for i in range(len(K_can)):
            K=K_can[i]
            _, topK = torch.topk(predictions,k=K,dim=-1)
            topK = topK.cpu().detach()
            # label=label.unsqueeze(-1)
            # print("label:",label)
            # print("topK:",topK)
            hit = label == topK
            # print("hit:",hit)
            # print("hit[:, :K].sum().item() :",hit[:, :K].sum().item() )
            hr = hit[:, :K].sum().item() / label.size()[0]
            total_HT[i]+=hr

            hit = (label == topK).int()
            max_dcg = dcg(torch.tensor([1] + [0] * (K - 1)))
            predict_dcg = dcg(hit[:, :K])
            ndcg=(predict_dcg / max_dcg).mean().item()
            total_NDCG[i]+=ndcg

        num_batch+=1

    #     for iii in range(len(seq)):
    #         predictions = model.predict(seq[iii].unsqueeze(0))
    #         predictions = predictions[0]  # - for 1st argsort DESC
    #         _, topK = predictions.topk(K, dim=-1, largest=True, sorted=True)
    #         topK = topK.cpu().detach().numpy()
    #         if label[iii][-1] in topK[:K]:
    #             t_hr+=1
    #             rank = np.argwhere(topK[:K] == label[iii][-1])
    #             t_ndcg += 1.0 / np.log2(rank.item() + 2)
    #         valid_u+=1
    # print("t_hr:",t_hr/valid_u)
    # print("t_ndcg:",t_ndcg/valid_u)
    # print("HT:", HT / num_batch)
    # print("NDCG:", NDCG / num_batch)
    for i in range(len(total_NDCG)):
        total_HT[i]=total_HT[i]/num_batch
        total_NDCG[i]=total_NDCG[i]/num_batch

    return total_NDCG, total_HT


# evaluate on val set
def evaluate_valid(model, valid_dataloader, args):
    num_batch = 0.0
    K_can = [1,3,5,10,20,50,100]
    total_NDCG=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    total_HT=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]

    for valid_batch in valid_dataloader:
        seq = valid_batch[0]
        source_seqs=valid_batch[1]
        mix_seqs=valid_batch[2]
        label = valid_batch[3]
        predictions = model.predict(seq,source_seqs,mix_seqs)
        for i in range(len(K_can)):
            K=K_can[i]
            _, topK = torch.topk(predictions, k=K, dim=-1)
            topK = topK.cpu().detach()
            # label = label.unsqueeze(-1)
            # print("label:",label)
            # print("topK:",topK)
            hit = label == topK
            # print("hit:",hit)
            # print("hit[:, :K].sum().item() :",hit[:, :K].sum().item() )
            hr = hit[:, :K].sum().item() / label.size()[0]
            total_HT[i] += hr

            hit = (label == topK).int()
            max_dcg = dcg(torch.tensor([1] + [0] * (K - 1)))
            predict_dcg = dcg(hit[:, :K])
            ndcg = (predict_dcg / max_dcg).mean().item()
            total_NDCG[i] += ndcg

        num_batch += 1

    for i in range(len(total_NDCG)):
        total_HT[i] = total_HT[i] / num_batch
        total_NDCG[i] = total_NDCG[i] / num_batch

    return total_NDCG, total_HT