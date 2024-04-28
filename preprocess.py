import numpy as np
import os
import torch

# 
def init_embeddings(entity_file, relation_file):
    entity_emb, relation_emb = [], []

    with open(entity_file) as f:
        for line in f:
            entity_emb.append([float(val) for val in line.strip().split()])

    with open(relation_file) as f:
        for line in f:
            relation_emb.append([float(val) for val in line.strip().split()])

    return np.array(entity_emb, dtype=np.float32), np.array(relation_emb, dtype=np.float32)

def read_entity_from_id(data_path):
    entity2id = {}
    with open(os.path.join(data_path, 'entities.dict')) as fin:
        for line in fin:
            eid, entity = line.strip().split(" ")
            entity2id[entity] = int(eid)

    return entity2id

def read_relation_from_id(data_path):
    relation2id = {}
    with open(os.path.join(data_path, 'relations.dict')) as fin:
        relation2id = dict()
        for line in fin:
            rid, relation = line.strip().split(" ")
            relation2id[relation] = int(rid)
    return relation2id

# trainning data---format:(h,r,t,label)
def parse_line(line):
    line = line.strip().split()
    e1, relation, e2 ,label= line[0].strip(), line[1].strip(), line[2].strip(), line[3].strip()
    return e1, relation, e2

def load_datax(filename, entity2id, relation2id,is_unweigted=False, directed=True):
	with open(filename) as f:
		lines = f.readlines()
	# this is list for relation triples
	triples_data = []
	rows, cols, data = [], [], []
	unique_entities = set()
	for line in lines:
		e1, relation, e2 = parse_line(line)
		unique_entities.add(e1)
		unique_entities.add(e2)
		# 
		triples_data.append(
			(entity2id[e1], relation2id[relation], entity2id[e2]))
		if not directed:
				# Connecting source and tail entity
			rows.append(entity2id[e1])
			cols.append(entity2id[e2])
			if is_unweigted:
				data.append(1)
			else:
				data.append(relation2id[relation])
		
		rows.append(entity2id[e2])
		cols.append(entity2id[e1])
		if is_unweigted:
			data.append(1)
		else:
			data.append(relation2id[relation])
	print("number of unique_entities ->", len(unique_entities))
	
	return triples_data, (rows, cols, data)


def build_data(data_path,is_unweigted=False, directed=True):
	entity2id = read_entity_from_id(data_path)
	relation2id = read_relation_from_id(data_path)
    
	
	train_triples, train_adjacency_mat=load_datax(os.path.join(data_path, 
        'train.txt'), entity2id, relation2id,is_unweigted, directed)
	test_triples, test_adjacency_mat= load_datax(os.path.join(
        data_path, 'test.txt'), entity2id, relation2id, is_unweigted, directed)

	rel_num=len(relation2id)
	return (train_triples, train_adjacency_mat),(test_triples, test_adjacency_mat),entity2id, relation2id,rel_num

def gettrain_adj(train_data):
	train_triples=train_data[0]
	adj_indices = torch.LongTensor(
            [train_data[1][0], train_data[1][1]])  # rows and columns
	#
	adj_values = torch.LongTensor(train_data[1][2])
	# 
	train_adj_matrix = (adj_indices, adj_values) #tuple
	# print("train_adj_matrix shape:",train_adj_matrix.shape)
	return train_adj_matrix	

def get_graph(train_adj_matrix):
	graph={}
	print(train_adj_matrix[0].transpose(0, 1).shape)
	print(train_adj_matrix[1].unsqueeze(1).shape)
	all_tiples = torch.cat([train_adj_matrix[0].transpose(
            0, 1), train_adj_matrix[1].unsqueeze(1)], dim=1)
	# print("all_tiples:",all_tiples)
	for data in all_tiples:
			source = data[1].data.item()
			target = data[0].data.item()
			value = data[2].data.item()

			if(source not in graph.keys()):
				graph[source] = {}
				graph[source][target] = value
			else:
				graph[source][target] = value
	print("Graph created")	
	return graph

def get_rel_adj(train_graph,rel_adj):
	for head in train_graph.keys():
		rel_tail = train_graph[head]
		rels = list(set(rel_tail.values()))
		for i in range(len(rels)):
			for j in range(i+1, len(rels)):
				rel_adj[rels[i]][rels[j]] += 1
				rel_adj[rels[j]][rels[i]] += 1
    
	return rel_adj
def normalized_laplacian(adj_matrix):
    R = np.sum(adj_matrix, axis=1)
    R_sqrt =1/np.sqrt(R)
    D_sqrt = np.diag(R_sqrt)
    I = np.eye(adj_matrix.shape[0])
    return I - np.matmul(np.matmul(D_sqrt, adj_matrix), D_sqrt)

def get_iteration_batch(iter_num,batch_size_gat,train_indices,entity2id,train_values,invalid_valid_ratio,valid_triples_dict):
        if (iter_num + 1) * batch_size_gat <= len(train_indices):
            batch_indices = np.empty(
                (batch_size_gat * (invalid_valid_ratio + 1), 3)).astype(np.int32)
            batch_values = np.empty(
                (batch_size_gat * (invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(batch_size_gat * iter_num,
                            batch_size_gat * (iter_num + 1))

            batch_indices[:batch_size_gat,
                               :] = train_indices[indices, :]
            batch_values[:batch_size_gat,
                              :] = train_values[indices, :]

            last_index = batch_size_gat

            if invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(entity2id), last_index * invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(
                    batch_indices[:last_index, :], (invalid_valid_ratio, 1))
                batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(
                    batch_values[:last_index, :], (invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(invalid_valid_ratio // 2):
                        current_index = i * (invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], batch_indices[last_index + current_index, 1],
                               batch_indices[last_index + current_index, 2]) in valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(entity2id))
                        batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        batch_values[last_index + current_index, :] = [-1]

                    for j in range(invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (invalid_valid_ratio // 2) + \
                            (i * (invalid_valid_ratio // 2) + j)

                        while (batch_indices[last_index + current_index, 0], batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(entity2id))
                        batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        batch_values[last_index + current_index, :] = [-1]

                return batch_indices, batch_values

            return batch_indices, batch_values

        else:
            last_iter_size = len(train_indices) - \
                batch_size_gat * iter_num
            batch_indices = np.empty(
                (last_iter_size * (invalid_valid_ratio + 1), 3)).astype(np.int32)
            batch_values = np.empty(
                (last_iter_size * (invalid_valid_ratio + 1), 1)).astype(np.float32)

            indices = range(batch_size_gat * iter_num,
                            len(train_indices))
            batch_indices[:last_iter_size,
                               :] = train_indices[indices, :]
            batch_values[:last_iter_size,
                              :] = train_values[indices, :]

            last_index = last_iter_size

            if invalid_valid_ratio > 0:
                random_entities = np.random.randint(
                    0, len(entity2id), last_index * invalid_valid_ratio)

                # Precopying the same valid indices from 0 to batch_size to rest
                # of the indices
                batch_indices[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(
                    batch_indices[:last_index, :], (invalid_valid_ratio, 1))
                batch_values[last_index:(last_index * (invalid_valid_ratio + 1)), :] = np.tile(
                    batch_values[:last_index, :], (invalid_valid_ratio, 1))

                for i in range(last_index):
                    for j in range(invalid_valid_ratio // 2):
                        current_index = i * (invalid_valid_ratio // 2) + j

                        while (random_entities[current_index], batch_indices[last_index + current_index, 1],
                               batch_indices[last_index + current_index, 2]) in valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(entity2id))
                        batch_indices[last_index + current_index,
                                           0] = random_entities[current_index]
                        batch_values[last_index + current_index, :] = [-1]

                    for j in range(invalid_valid_ratio // 2):
                        current_index = last_index * \
                            (invalid_valid_ratio // 2) + \
                            (i * (invalid_valid_ratio // 2) + j)

                        while (batch_indices[last_index + current_index, 0], batch_indices[last_index + current_index, 1],
                               random_entities[current_index]) in valid_triples_dict.keys():
                            random_entities[current_index] = np.random.randint(
                                0, len(entity2id))
                        batch_indices[last_index + current_index,
                                           2] = random_entities[current_index]
                        batch_values[last_index + current_index, :] = [-1]

                return batch_indices, batch_values

            return batch_indices, batch_values
