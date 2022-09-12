import torch
import os
import numpy as np

pred_dir = "exp/predictions_all"

if os.path.exists('prediction.csv'):
    print("removing 'prediction.csv'...")
    os.remove('prediction.csv')   
        
for client_id in range(1, 14):
    if client_id <= 8:
        task_type = "graphClassification"
    else:
        task_type = "graphRegression"
    print(f"client={client_id}, task={task_type}")
    y_preds = 0.
    n = 0
    counts = [0, 1, 2, 3, 4]
    for count in counts:
        file_name = f"client_{client_id}_{task_type}_{count}.pt"
        pred_path = os.path.join(pred_dir, file_name)
        if os.path.exists(pred_path):
            y_inds, y_probs = torch.load(pred_path)
            y_preds += y_probs
            n += 1
        
    assert n!= 0, "no files found!"
    print(f"{n} files checked!")
    y_preds /= n
    if "classification" in task_type.lower():
        y_preds = y_preds.argmax(-1)
        
    with open('prediction.csv', 'a') as file:
        for y_ind, y_pred in zip(y_inds,  y_preds):
            if 'classification' in task_type.lower():
                line = [client_id, y_ind] + [y_pred]
            else:
                line = [client_id, y_ind] + list(y_pred)
            file.write(','.join([str(_) for _ in line]) + '\n')
        
        