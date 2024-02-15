from .node_edge_net import NodeEdgeNet


def get_refine_net(config, net_type, node_hidden_dim, edge_hidden_dim, train_link):
    if net_type == 'node_edge_net':
        refine_net = NodeEdgeNet(
            node_dim=node_hidden_dim,
            edge_dim=edge_hidden_dim,
            num_blocks=config.num_blocks,
            cutoff=config.cutoff,
            use_gate=config.use_gate,
            update_pos=train_link,
            expansion_mode=config.get('expansion_mode', 'linear')
        )
    else:
        raise ValueError(net_type)
    return refine_net
