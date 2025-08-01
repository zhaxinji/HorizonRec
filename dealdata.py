#!/user/bin/env python3
# -*- coding: utf-8 -*-
# filename="D:/trainSet/Sports_and_Outdoors.txt"
# w_filename="./data/Sports.txt"
#
# def get_data(filename):
#     user_items={}
#     with open(filename,"r") as f:
#         for line in f.readlines():
#             line=line.strip().split(" ")
#             uid=int(line[0])
#             items=[int(item) for item in line[1:]]
#             if uid not in user_items:
#                 user_items[uid]=items
#     return user_items
#
# train_data=get_data(filename)
# with open(w_filename,"w") as fw:
#     for user,items in train_data.items():
#         for i,item in enumerate(items):
#             tempdata = ""
#             uid = user
#             tempdata += str(uid)
#             tempdata+=" "
#             itemid=item
#             tempdata+=str(itemid)
#             tempdata+="\n"
#             fw.write(tempdata)
#     fw.close()
# import pickle
# filename="C:/Users/ManbaIEsther/Desktop/新建文件夹 (3)/新建文件夹/DiffMM-main/Datasets/tiktok/trnMat.pkl"
# w_filename="C:/Users/ManbaIEsther/Desktop/新建文件夹 (3)/新建文件夹/DiffMM-main/Datasets/tiktok/trnMat.txt"
# filename="C:/Users/ManbaIEsther/Desktop/新建文件夹 (3)/新建文件夹/DiffMM-main/Datasets/tiktok/valMat.pkl"
# w_filename="C:/Users/ManbaIEsther/Desktop/新建文件夹 (3)/新建文件夹/DiffMM-main/Datasets/tiktok/valid.txt"
# filename="C:/Users/ManbaIEsther/Desktop/新建文件夹 (3)/新建文件夹/DiffMM-main/Datasets/tiktok/tstMat.pkl"
# w_filename="C:/Users/ManbaIEsther/Desktop/新建文件夹 (3)/新建文件夹/DiffMM-main/Datasets/tiktok/test.txt"
# with open(filename,"rb") as f:
#     dataset=pickle.load(f)
# with open(w_filename,"w") as fw:
#     for i in range(len(dataset.row)):
#         tempdata = ""
#         uid = dataset.row[i]
#         tempdata += str(uid)
#         tempdata+=" "
#         itemid=dataset.col[i]
#         tempdata+=str(itemid)
#         tempdata+="\n"
#         fw.write(tempdata)
#     fw.close()
# filename="D:/trainSet/ml-100k/u.item"
# filename1="D:/trainSet/ml-100k/u.info"
# with open(filename,"r",encoding="ISO-8859-1") as fr:
#     for line in fr.readlines():
#         print("line:",line.split("|"))
# with open(filename1,"rb") as fr:
#     for line in fr.readlines():
#         print("line:",line)
import pandas as pd
from tqdm import tqdm
import random
path_file="D:/llm_dataset/BookCrossing/Book reviews"
rating = pd.read_csv(path_file+'/BX-Book-Ratings.csv', sep=';', encoding="latin-1")
users = pd.read_csv(path_file+'/BX-Users.csv', sep=';', encoding="latin-1")
books = pd.read_csv(path_file+'/BX_Books.csv', sep=';', encoding="latin-1", error_bad_lines=False)
print("books:",books)
rating = pd.merge(rating, books, on='ISBN', how='inner')
books.to_csv('book_item_mapping.csv', index=False)
user_dict = {}
item_id = {}
for index, row in tqdm(books.iterrows()):
    item_id[row['ISBN']] = index
for index, row in tqdm(rating.iterrows()):
    userid = row['User-ID']
    if not user_dict.__contains__(userid):
        user_dict[userid] = {
            'ISBN': [],
            'Book-Rating': [],
            'Book-Title': [],
            'Book-Author': [],
            'Year-Of-Publication': [],
        }
    user_dict[userid]['ISBN'].append(item_id[row['ISBN']])
    user_dict[userid]['Book-Rating'].append(float(row['Book-Rating']))
    user_dict[userid]['Book-Title'].append(row['Book-Title'])
    user_dict[userid]['Book-Author'].append(row['Book-Author'])
    user_dict[userid]['Year-Of-Publication'].append(row['Year-Of-Publication'])

new_user_dict = {}
mx=0
for key in user_dict.keys():
    mx = max(mx, len(user_dict[key]['ISBN']))
    if len(user_dict[key]['ISBN'])  <= 3:
        pass
    else:
        new_user_dict[key] = user_dict[key]
user_list = list(new_user_dict.keys())
random.shuffle(user_list)
train_user = user_list[:int(len(user_list) * 0.8)]
valid_usser = user_list[int(len(user_list) * 0.8):int(len(user_list) * 0.9)]
test_user = user_list[int(len(user_list) * 0.9):]
nrows = []
user_list=train_user
for user in user_list:
    item_id = user_dict[user]['ISBN']
    print("items_id:",item_id)
    rating = [int(_ > 5) for _ in user_dict[user]['Book-Rating']]
    random.seed(42)
    random.shuffle(item_id)
    random.seed(42)
    random.shuffle(rating)
    nrows.append([user, item_id[:-1][:10], rating[:-1][:10], item_id[-1], rating[-1]])
    print("nrows:",nrows)