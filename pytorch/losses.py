import torch
import torch.nn.functional as F


def clip_bce(output_dict, target_dict):
    """Binary crossentropy loss.
    """
    return F.binary_cross_entropy(
        output_dict['clipwise_output'], target_dict['target'])


def get_loss_func(loss_type):
    if loss_type == 'clip_bce':
        return clip_bce
    elif loss_type == 'f1micro':
        return F1_loss_objective


def F1_loss_objective(binarized_output, y_true):
    # let's first convert binary vector prob into logits
    #     prob = torch.clamp(prob, 1.e-12, 0.9999999)

    #     average = 'macro'
    average = 'micro'
    epsilon = torch.tensor(1e-12)

    if average == 'micro':
        y_true = torch.flatten(y_true)
        binarized_output = torch.flatten(binarized_output)

    true_positives = torch.sum(y_true * binarized_output, dim=0)
    predicted_positives = torch.sum(binarized_output, dim=0)
    positives = torch.sum(y_true, dim=0)
    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (positives + epsilon)

    f1 = 2 * ((precision * recall) / (precision + recall + epsilon))
    #     return precision, recall, f1
    return - f1.mean()


def macro_F1_loss_objective(binarized_output, y_true):
    # let's first convert binary vector prob into logits
    #     prob = torch.clamp(prob, 1.e-12, 0.9999999)

    epsilon = torch.tensor(1e-12)
    true_positives = torch.sum(y_true * binarized_output, dim=0)
    predicted_positives = torch.sum(binarized_output, dim=0)
    positives = torch.sum(y_true, dim=0)
    precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (positives + epsilon)

    f1 = 2 * ((precision * recall) / (precision + recall + epsilon))
    #     return precision, recall, f1
    return - f1.mean()


def macro_recall_loss_objective(binarized_output, y_true):
    # let's first convert binary vector prob into logits
    #     prob = torch.clamp(prob, 1.e-12, 0.9999999)

    epsilon = torch.tensor(1e-12)
    true_positives = torch.sum(y_true * binarized_output, dim=0)
    #     predicted_positives = torch.sum(binarized_output, dim=0)
    positives = torch.sum(y_true, dim=0)
    #     precision = true_positives / (predicted_positives + epsilon)
    recall = true_positives / (positives + epsilon)

    nb_predicted_positives = torch.sum(binarized_output)
    nb_true_positives = torch.sum(y_true)
    penalty = 10. * (1. - nb_predicted_positives / nb_true_positives) ** 2

    #     f1 = 2 * ((precision * recall) / (precision + recall + epsilon))
    #     return precision, recall, f1
    print("recall mean: %.3f --- penalty: %.3f" % (recall.mean(), penalty))
    return - recall.mean() + penalty


def setAcc_loss_objective(binarized_output, y_true):
    # let's first convert binary vector prob into logits
    #     prob = torch.clamp(prob, 1.e-12, 0.9999999)

    #     average = 'macro'
    average = 'micro'
    epsilon = torch.tensor(1e-12)

    if average == 'micro':
        y_true = torch.flatten(y_true)
        binarized_output = torch.flatten(binarized_output)

    true_positives = torch.sum(y_true * binarized_output, dim=0)
    #     return precision, recall, f1
    return - true_positives.mean()


