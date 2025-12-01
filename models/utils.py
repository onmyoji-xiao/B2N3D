def get_siamese_features(net, in_features, aggregator=None):
    """ Applies a network in a siamese way, to 'each' in_feature independently
    :param net: nn.Module, Feat-Dim to new-Feat-Dim
    :param in_features: B x  N-objects x Feat-Dim
    :param aggregator, (opt, None, torch.stack, or torch.cat)
    :return: B x N-objects x new-Feat-Dim
    """
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    for i in range(n_items):
        features = net(in_features[:, i])
        out_features.append(features)
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
    return out_features


def get_hybrid_features(net, in_features, clip_features, aggregator=None):
    independent_dim = 1
    n_items = in_features.size(independent_dim)
    out_features = []
    out_features2 = []
    for i in range(n_items):
        features, features2 = net(in_features[:, i], clip_features[:, i])
        out_features.append(features)
        out_features2.append(features2)
    if aggregator is not None:
        out_features = aggregator(out_features, dim=independent_dim)
        out_features2 = aggregator(out_features2, dim=independent_dim)
    return out_features, out_features2
