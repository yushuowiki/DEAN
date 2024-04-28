import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import time


from Model.gcn import GCN

from Model.SpKBGAT import SpGAT
from Model.readout import KGAvgReadout
from Model.discriminator import Discriminator

# spGAT 
import sys
sys.path.append("..")


CUDA = torch.cuda.is_available()

class  DEAN(nn.Module):
    def __init__(self,initial_entity_emb,initial_relation_emb,entity_out_dim, relation_out_dim,
                    drop_GAT, alpha, nheads_GAT,rel_adj,n_h):
        super(DEAN,self).__init__()
        # Entities
        self.num_nodes = initial_entity_emb.shape[0]
        self.entity_in_dim = initial_entity_emb.shape[1] #dimension of initEntEmb
        self.entity_out_dim_1 = entity_out_dim[0]
        self.nheads_GAT_1 = nheads_GAT[0]
        self.entity_out_dim_2 = entity_out_dim[1]
        self.nheads_GAT_2 = nheads_GAT[1]

        # relations
        self.num_relation = initial_relation_emb.shape[0]
        self.relation_dim = initial_relation_emb.shape[1] #dimension of initrelEmb

        self.relation_out_dim_1 = relation_out_dim[0]

        self.drop_GAT = drop_GAT
        self.alpha = alpha      # For leaky relu
          
        # edge(edge_list)、edge_emb、edge_type   from load_data
        # self.edge_list=edge_list
        # self.edge_type=edge_type

        self.entity_embeddings = nn.Parameter(initial_entity_emb)
        self.relation_embeddings = nn.Parameter(initial_relation_emb)


        self.W_entities=nn.Parameter(torch.zeros(
            size=(self.entity_in_dim, self.entity_out_dim_1 * self.nheads_GAT_1)))

        nn.init.xavier_uniform_(self.W_entities.data, gain=1.414)

        # Encoder for entemb relemb ,adopt kbgat 
        self.sparse_gat_1=SpGAT(self.num_nodes, self.entity_in_dim, self.entity_out_dim_1, self.relation_dim,
                                    self.drop_GAT, self.alpha, self.nheads_GAT_1)

        # Deep graph infomax for Rel mutal infomax
        '''
        rel_X rel_adj----DGI---->rel_emb (update rel)
        rel_X from KBGAT 
        '''
        self.rel_adj=rel_adj
        # n_in n_h 
        # n_in rel
        self.gcn = GCN(self.entity_out_dim_1 * self.nheads_GAT_1, n_h)
        self.read= KGAvgReadout()
        self.sigm=nn.Sigmoid()
        self.disc = Discriminator(n_h)

        self.final_relEmb=nn.Parameter(initial_relation_emb)

        # KBGAT batch-inputs adj
    def forward(self,batch_inputs,adj):
        # batch inputs
        edge_list = adj[0]
        edge_type = adj[1]
        if(CUDA):
            edge_list = edge_list.cuda()
            edge_type = edge_type.cuda()

        edge_embed=self.relation_embeddings[edge_type]

        out_entity_1, out_relation_1 = self.sparse_gat_1(self.entity_embeddings, self.relation_embeddings,
        edge_list, edge_type, edge_embed)

        mask_indices = torch.unique(batch_inputs[:, 2]).cuda()
        mask = torch.zeros(self.entity_embeddings.shape[0]).cuda()
        mask[mask_indices] = 1.0

        entities_upgraded = self.entity_embeddings.mm(self.W_entities)
        out_entity_1 = entities_upgraded + \
            mask.unsqueeze(-1).expand_as(out_entity_1) * out_entity_1
        out_entity_1 = F.normalize(out_entity_1, p=2, dim=1) #实体更新 结束
        # rel infomax part
        #
        # self.rel_adj=torch.from_numpy(self.rel_adj)
        print("out_relation_1:",out_relation_1.shape)
        h_1=self.gcn(out_relation_1,self.rel_adj)
        msk=None
        #

        c=self.read(out_relation_1,msk)
        c=self.sigm(c)

        idx = np.random.permutation(self.num_relation)
        shuf_relfts =out_relation_1[idx, :]

        h_2 = self.gcn(shuf_relfts, self.rel_adj)
        # print(c.shape, h_1.shape, h_2.shape)
        samp_bias1=None
        samp_bias2=None
        ret=self.disc(c,h_1,h_2,samp_bias1, samp_bias2)

        self.final_relEmb.data=h_1.data

        
        return out_entity_1,h_1,ret

