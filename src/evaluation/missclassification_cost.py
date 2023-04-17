import torch
from src.utils import get_constants


def evaluate_mcc(pred, true):
    """
    Returns a single value showing the estimated cost of the model based on its predictions and the true labels.
    Used to evaluate model performance and select best model

    Input:
    pred: tensor of model predictions, where each element is a three-element row tensor [Fresh, Half-Fresh, Spoiled]
    true: tensor of actual labels, where each element is a three-element row tensor [Fresh, Half-Fresh, Spoiled]

    Output:
    Single float value showing the estimated cost of the model based on its errors
    """

    cost_matrix = get_constants('misclassification_cost_matrix')
    device = get_constants('device')

    cost_matrix = torch.tensor(matrix).to(device)
    pred = pred.to(device)
    true = true.to(device)

    results = []

    # Loop through each row of true and pred
    for i in range(len(true_labels)):
        # Extract the ith row of true and pred
        true_i = true_labels[i].unsqueeze(0)  # Add a dimension of size 1 to make it a row vector
        pred_i = pred_labels[i].unsqueeze(0)  # Add a dimension of size 1 to make it a row vector
        print(true_i, pred_i)
        # Compute the product in the specified order
        product = true_i.mm(tensor).mm(pred_i.t())  # .t() is used to transpose pred_i

        # Add the result to the list of results
        results.append(product)

    output = torch.cat(results, dim=0)
    final_cost = sum(output).item()

    return final_cost



