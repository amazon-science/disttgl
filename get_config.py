import math

def get_config(data, tot_rank, minibatch_parallelism=1):
    sample_param = {
        'layer': 1, 
        'neighbor': [10],
        'strategy': 'recent',
        'prop_time': False,
        'history': 1,
        'duration': 0,
        'num_thread': 32
    }
    memory_param = {
        'type': 'node',
        'dim_time': 100,
        'deliver_to': 'self',
        'mail_combine': 'last',
        'memory_update': 'smart',
        'mailbox_size': 1,
        'combine_node_feature': True,
        'dim_out': 100
    }
    gnn_param = {
        'arch': 'transformer_attention',
        'layer': 1,
        'att_head': 2,
        'dim_time': 100,
        'dim_out': 100
    }
    epoch = 10 if data in ['GDELT', 'LINK'] else 100
    train_param = {
        'epoch': math.ceil(epoch / tot_rank * minibatch_parallelism),
        'batch_size': 600,
        'lr': 0.0001 * tot_rank,
        'dropout': 0.2,
        'att_dropout': 0.2
    }

    train_param['train_neg_samples'] = 1
    train_param['eval_neg_samples'] = 49

    if data in ['GDELT', 'LINK']:
        train_param['batch_size'] = 3200
        # train_param['lr'] = 0.00001

    return sample_param, memory_param, gnn_param, train_param