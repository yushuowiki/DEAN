import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from copy import deepcopy
from torch.autograd import Variable

from Model.DEAN import DEAN

import random
import os
import sys
import time
import pickle
import argparse
print(os.getcwd())
# sys.path.append("/Users/tuhuiling/Desktop/OutDGI")

from preprocess import build_data, init_embeddings,gettrain_adj,get_graph,get_rel_adj,normalized_laplacian,get_iteration_batch

def parse_args():
	args = argparse.ArgumentParser()
	args.add_argument("-pre_emb", "--pretrained_emb", type=bool,
                      default=False, help="Use pretrained embeddings")
	# 
	args.add_argument("-data", "--data",
                      default="data/nations", help="data directory")
	args.add_argument("-emb_size", "--embedding_size", type=int,
					default=50, help="Size of embeddings (if pretrained not used)")
	args.add_argument("-out_dim", "--entity_out_dim", type=int, nargs='+',
                      default=[100, 200], help="Entity output embedding dimensions")
	args.add_argument("-drop_GAT", "--drop_GAT", type=float,
                      default=0.3, help="Dropout probability for SpGAT layer")
	args.add_argument("-alpha", "--alpha", type=float,
                      default=0.2, help="LeakyRelu alphs for SpGAT layer")
	args.add_argument("-h_gat", "--nheads_GAT", type=int, nargs='+',
                      default=[2, 2], help="Multihead attention SpGAT")
	# 
	args.add_argument("-gcn_out","--n_h",type=int,default=256,help="rel embeding/gcn_out")
	args.add_argument("-l", "--lr", type=float, default=1e-3)
	args.add_argument("-w_gat", "--weight_decay_DGI", type=float,
                      default=5e-6, help="L2 reglarization for rel infomax")
	args.add_argument("-margin", "--margin", type=float,
                      default=5, help="Margin used in hinge loss")
	
	args.add_argument("-e_g", "--epochs_dgi", type=int,
                      default=3600, help="Number of epochs")
	args.add_argument("-b_gat", "--batch_size_gat", type=int,
                      default=86835, help="Batch size for GAT")

	args.add_argument("-neg_s_gat", "--valid_invalid_ratio_gat", type=int,
                      default=2, help="Ratio of valid to invalid triples for GAT training")
	
	args = args.parse_args()
	
	return args

args = parse_args()
# initial_entity_emb,initial_relation_emb,entity_out_dim, relation_out_dim,
                #  drop_GAT, alpha, nheads_GAT,rel_adj,n_h，edge_list,edge_type,edge_embed
def load_data(args):
	'''
	return: initial_entembedding,initial_relembedding
	'''
	#
	train_data,test_data,entity2id,relation2id,rel_num=build_data(args.data,is_unweigted=False, directed=True)
	if args.pretrained_emb:
		entity_embeddings, relation_embeddings = init_embeddings(os.path.join(args.data, 'entity2vec.txt'),
																os.path.join(args.data, 'relation2vec.txt'))
		print("Initialised relations and entities from TransE")
	else:
		entity_embeddings = np.random.randn(
			len(entity2id), args.embedding_size)
		relation_embeddings = np.random.randn(
			len(relation2id), args.embedding_size)
		print("Initialised relations and entities randomly")
	
	return entity2id,relation2id,train_data,test_data,torch.FloatTensor(entity_embeddings),torch.FloatTensor(relation_embeddings),rel_num

entity2id,relation2id,train_data,test_data,entity_embeddings, relation_embeddings,rel_num = load_data(args)
entity_embeddings_copied = deepcopy(entity_embeddings)
relation_embeddings_copied = deepcopy(relation_embeddings)
rel_adj=np.zeros(shape=(rel_num,rel_num))

print("Initial entity dimensions {} , relation dimensions {}".format(
    entity_embeddings.size(), relation_embeddings.size()))
print("rel_num:",rel_num)
print("rel_adj shape:",rel_adj.shape)

'''

'''
def batch_gat_loss(gat_loss_func, train_indices, entity_embed, relation_embed):
    len_pos_triples = int(train_indices.shape[0] / (int(args.valid_invalid_ratio_gat) + 1))

    pos_triples = train_indices[:len_pos_triples]
    neg_triples = train_indices[len_pos_triples:]

    pos_triples = pos_triples.repeat(int(args.valid_invalid_ratio_gat), 1)

    source_embeds = entity_embed[pos_triples[:, 0]]
    relation_embeds = relation_embed[pos_triples[:, 1]]
    tail_embeds = entity_embed[pos_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    pos_norm = torch.norm(x, p=1, dim=1)

    source_embeds = entity_embed[neg_triples[:, 0]]
    relation_embeds = relation_embed[neg_triples[:, 1]]
    tail_embeds = entity_embed[neg_triples[:, 2]]

    x = source_embeds + relation_embeds - tail_embeds
    neg_norm = torch.norm(x, p=1, dim=1)

    y = -torch.ones(int(args.valid_invalid_ratio_gat) * len_pos_triples).cuda()

    loss = gat_loss_func(pos_norm, neg_norm, y)
    return loss

train_adj_matrix=gettrain_adj(train_data)

'''
get rel_adj/ from getgraph of KBGAT
'''

train_graph=get_graph(train_adj_matrix)
print(len(train_graph))
# print(train_graph)

norm_rel_adj=normalized_laplacian(get_rel_adj(train_graph,rel_adj))

# np.save(os.path.join(args.data,'rel_adj.npy'),norm_rel_adj)
'''
valid_triples_dict=train_triples+test_triples
'''
valid_triples_dict = {j: i for i, j in enumerate(
            train_data[0] +  test_data[0])}

CUDA = torch.cuda.is_available()

'''
Training step
'''

print("Defining model")
print("\nModel type -> GAT layer with {} heads used , Initital Embeddings training".format(args.nheads_GAT[0]))
norm_rel_adj=torch.from_numpy(norm_rel_adj).float()
norm_rel_adj=norm_rel_adj.cuda()
model_DEAN=DEAN(entity_embeddings, relation_embeddings, args.entity_out_dim, args.entity_out_dim,
                                args.drop_GAT, args.alpha, args.nheads_GAT,norm_rel_adj,args.n_h)

if CUDA:
	model_DEAN.cuda()
optimizer = torch.optim.Adam(
        model_DEAN.parameters(), lr=args.lr, weight_decay=args.weight_decay_DGI)

scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=500, gamma=0.5, last_epoch=-1)

gat_loss_func = nn.MarginRankingLoss(margin=args.margin)

epoch_losses = []   # losses of all epochs
print("Number of epochs {}".format(args.epochs_dgi))
b_xent = nn.BCEWithLogitsLoss()
for epoch in range(args.epochs_dgi):
	print("\nepoch-> ", epoch)
	random.shuffle(train_data[0])

	train_indices=np.array(list(train_data[0])).astype(np.int32)
	train_values = np.array(
			[[1]] * len(train_data[0])).astype(np.float32)

	model_DEAN.train()
	start_time = time.time()
	epoch_loss = []

	if len(train_indices) % args.batch_size_gat == 0:
			num_iters_per_epoch = len(
				train_indices) // args.batch_size_gat
	else:
			num_iters_per_epoch = (
				len(train_indices) // args.batch_size_gat) + 1
	
	for iters in range(num_iters_per_epoch):
		start_time_iter=time.time()

		# get batch_indices batch_inputs
		batch_indices,batch_values=get_iteration_batch(iters,args.batch_size_gat,train_indices,entity2id,train_values,args.valid_invalid_ratio_gat,valid_triples_dict)

		if CUDA:
				batch_indices = Variable(
					torch.LongTensor(batch_indices)).cuda()
				batch_values = Variable(torch.FloatTensor(batch_values)).cuda()

		else:
				batch_indices = Variable(torch.LongTensor(batch_indices))
				batch_values = Variable(torch.FloatTensor(batch_values))

		# forward pass
		entity_embed,relation_embed,logits=model_DEAN(batch_indices,train_adj_matrix)
		optimizer.zero_grad()
		loss1 = batch_gat_loss(gat_loss_func, batch_indices, entity_embed, relation_embed)
		lbl_1=torch.ones(rel_num)
		lbl_2=torch.zeros(rel_num)
		lbl = torch.cat((lbl_1, lbl_2), dim=0)
        # print(lbl.shape)
		if torch.cuda.is_available():
			lbl = lbl.cuda()
		loss2= b_xent(logits, lbl)
		loss=loss1+args.lamb*loss2
        # if epoch==0 and iters==0:
        #     print("loss item():",loss.item())
		loss.backward()
		optimizer.step()
		epoch_loss.append(loss.data.item())
		end_time_iter = time.time()

        # print("Iteration-> {0}  , Iteration_time-> {1:.4f} , Iteration_loss {2:.4f}".format(iters, end_time_iter - start_time_iter, loss.data.item()))
        # sys.exit(0)
		state={
                'state_dict'	: model_DEAN.state_dict(),
                'epoch'	: epoch+1,
    }
	torch.save(state,os.path.join(args.save_path, 'model.pth'))
	scheduler.step()
	print("Epoch {} , average loss {} , epoch_time {}".format(epoch, sum(epoch_loss) / len(epoch_loss), time.time() - start_time))
	epoch_losses.append(sum(epoch_loss) / len(epoch_loss))


	
		



	




