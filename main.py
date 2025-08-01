import os
import time
import random


os.environ["CUDA_LAUNCH_BLOCKING"]="1"
import torch
print(torch.__version__)
import argparse


from model import DiffuSAS
from utils import *
import pickle

def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='douban_music_book',help="[game-toy,toy-game,douban_music_book,douban_book_music]")
parser.add_argument('--target_dataset', default="douban_book",help="[toy,game,douban_book,douban_music]")
parser.add_argument('--train_dir',default='./data')
parser.add_argument('--batch_size', default=512, type=int)
parser.add_argument('--test_batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--max_len', default=200, type=int)
parser.add_argument('--hidden_units', default=64, type=int)
parser.add_argument('--num_blocks', default=2, type=int)
parser.add_argument('--num_epochs', default=10, type=int)
parser.add_argument('--num_heads', default=1, type=int)
parser.add_argument('--dropout_rate', default=0.2, type=float)
parser.add_argument('--l2_emb', default=0.0, type=float)
parser.add_argument('--device', default='cpu', type=str)
parser.add_argument('--inference_only', default=False, type=str2bool)
parser.add_argument('--state_dict_path', default=None, type=str)
parser.add_argument('--interval', default=50448, type=int,help="amazon:37868,douban:50448")
parser.add_argument('--game_item_num', default=11735, type=int)
parser.add_argument('--toy_item_num', default=37868, type=int)
parser.add_argument('--douban_book_item_num', default=50448, type=int)
parser.add_argument('--douban_music_item_num', default=36504, type=int)
parser.add_argument('--random_seed', default=2025, type=int)
parser.add_argument('--timesteps', default=32, type=int)
parser.add_argument('--beta_start', default=0.001, type=float)
parser.add_argument('--beta_end', default=0.02, type=float)
parser.add_argument('--beta_sche', default="cosine", type=str)
parser.add_argument('--pretrain', default=True, type=str2bool)
parser.add_argument('--temperature', default=1, type=float)
parser.add_argument('--w', default=0.6, type=float)
parser.add_argument('--diffusion_loss_weight', default=0.1, type=float)

def main():
    args = parser.parse_args()
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("args.device:",args.device)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    path_data = './data/' + args.dataset + '/dataset.pkl'
    with open(path_data, 'rb') as f:
        data_raw = pickle.load(f)

    args.num_items=data_raw['smap']
    args.num_users=data_raw['umap']
    tra_data = Data_Train(data_raw['train'], data_raw['mix_train'],args)

    val_data = Data_Val(data_raw['train'], data_raw['val'], data_raw['mix_train'],data_raw['mix_val'],args)
    test_data = Data_Test(data_raw['train'], data_raw['val'], data_raw['test'], data_raw['mix_train'],data_raw['mix_val'],data_raw['mix_test'],args)
    tra_data_loader = tra_data.get_pytorch_dataloaders()
    args.train_lengths=len(tra_data_loader)
    val_data_loader = val_data.get_pytorch_dataloaders()
    test_data_loader = test_data.get_pytorch_dataloaders()
    model = DiffuSAS(args).to(args.device) # no ReLU activation in original SASRec implementation?
    # for name, param in model.named_parameters():
    #     try:
    #         torch.nn.init.xavier_normal_(param.data)
    #     except:
    #         pass # just ignore those failed init layers
    if args.pretrain:
        model.load_pretrain_embedding_weight()

    # this fails embedding init 'Embedding' object has no attribute 'dim'
    # model.apply(torch.nn.init.xavier_uniform_)

    model.train() # enable model training

    epoch_start_idx = 1


    if args.inference_only:
        model.eval()
        t_test = evaluate(model, test_data_loader, args)
        print('test (NDCG@10: %.4f, HR@10: %.4f)' % (t_test[0], t_test[1]))

    # ce_criterion = torch.nn.CrossEntropyLoss()
    # https://github.com/NVIDIA/pix2pixHD/issues/9 how could an old bug appear again...
    # bce_criterion = torch.nn.BCEWithLogitsLoss() # torch.nn.BCELoss()
    ce_criterion = torch.nn.CrossEntropyLoss()
    adam_optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))

    T = 0.0
    t0 = time.time()
    best_hr=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_ndcg=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_epoch=[0, 0, 0, 0, 0, 0, 0]
    K_can = [1, 3, 5, 10, 20, 50, 100]
    count_n=0
    traintime=0
    text_time=0
    for epoch in range(epoch_start_idx, args.num_epochs + 1):
        total_loss=0.
        num_batch=0.
        flag=True
        train_begin_time=time.time()
        for train_batch in tra_data_loader:
            target_seq=train_batch[0].to(args.device)
            source_seqs=train_batch[1].to(args.device)
            mix_seqs=train_batch[2].to(args.device)
            label=train_batch[3].to(args.device)
            scores,diffuloss=model(target_seq,source_seqs,mix_seqs)
            # loss=ce_criterion(scores,label.squeeze(-1))+args.diffusion_loss_weight*diffuloss
            loss=ce_criterion(scores,label.squeeze(-1))+0.1*diffuloss
            # loss=ce_criterion(scores,label.squeeze(-1))
            adam_optimizer.zero_grad()

            # for param in model.item_emb.parameters(): loss += args.l2_emb * torch.norm(param)

            loss.backward()
            adam_optimizer.step()
            total_loss += loss.item()
            num_batch+=1
        print("loss in epoch {} iteration {}: {}".format(epoch, epoch,
                                                         total_loss / num_batch))  # expected 0.4~0.6 after init few epochs

        if args.inference_only: break # just to decrease identition
        train_end_time=time.time()
        traintime+=train_end_time-train_begin_time
        print("traintime:", traintime)
        test_begin=time.time()
        if epoch % 1 == 0:
            model.eval()
            t1 = time.time() - t0
            T += t1
            print('Evaluating', end='')

            t_test = evaluate(model, test_data_loader, args)
            test_end=time.time()
            print(" test time:",test_end-test_begin)
            t_valid = evaluate_valid(model, val_data_loader, args)
            for i in range(len(t_test[0])):
                print("epoch:%d, time: %f(s), valid (NDCG@%d: %.4f, HR@%d: %.4f), test (NDCG@%d: %.4f, HR@%d: %.4f)"
                    % (epoch, T, K_can[i],t_valid[0][i], K_can[i],t_valid[1][i], K_can[i],t_test[0][i],K_can[i], t_test[1][i]))

            # f.write(str(t_valid) + ' ' + str(t_test) + '\n')
            # f.flush()
            t0 = time.time()
            model.train()
            for i in range(len(best_hr)):
                flag=False
                if t_test[0][i]>best_ndcg[i] and t_test[1][i]>best_hr[i]:
                # if t_test[1][i]>best_hr[i]:
                    best_ndcg[i] = t_test[0][i]
                    best_hr[i]=t_test[1][i]
                    best_epoch[i]=epoch

            # if t_test[0]>best_ndcg and t_test[1]>best_hr:
            #     best_ndcg=t_test[0]
            #     best_hr=t_test[1]
            #     best_epoch=epoch

        if flag:
            count_n+=1
        # if epoch == args.num_epochs:
        #     # folder = args.dataset + '_' + args.train_dir
        #     # fname = 'SASRec.epoch={}.lr={}.layer={}.head={}.hidden={}.maxlen={}.pth'
        #     # fname = fname.format(args.num_epochs, args.lr, args.num_blocks, args.num_heads, args.hidden_units, args.max_len)
        #     fname='./data/' + args.dataset+".pth"
        #     torch.save(model.state_dict(), fname)
        if count_n>=3:
            break
    for i in range(len(best_hr)):
        print("best_epoch:",best_epoch[i],"hr@",K_can[i],"ndcg@",K_can[i],"hr-v:",best_hr[i],"ndcg-v:",best_ndcg[i])
    f.close()
    print("Done")
    print("traintime:",traintime)
    # model.create_tsne_weight(test_data_loader)

if __name__ == '__main__':
    main()