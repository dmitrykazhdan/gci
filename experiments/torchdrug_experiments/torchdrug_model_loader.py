import os.path
import tkinter
import matplotlib
matplotlib.use('TkAgg')

import torch
from torchdrug import tasks, core


def load_torchdrug_model(model_path, model, train_set, valid_set, test_set, dataset, n_epochs=None):

    if not os.path.exists(model_path):

        if n_epochs is None: n_epochs = 40

        task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                        criterion="bce", metric=("auprc", "auroc"))

        optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
        solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                             gpus=None, batch_size=128)
        solver.train(num_epoch=n_epochs)
        solver.evaluate("valid")
        solver.save(model_path)

    else:
        task = tasks.PropertyPrediction(model, task=dataset.tasks,
                                        criterion="bce", metric=("auprc", "auroc"))

        checkpoint = torch.load(model_path)["model"]
        task.load_state_dict(checkpoint, strict=False)

        optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
        solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                             gpus=None, batch_size=128)
        solver.evaluate("valid")
