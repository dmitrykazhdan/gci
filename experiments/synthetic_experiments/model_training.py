import torch

def run_train_loop(model, train_loader, test_loader, n_epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, n_epochs):
        train(model, train_loader, optimizer, criterion)
        train_acc = test(train_loader, model)
        test_acc = test(test_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')


def train(model, data_loader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(data_loader):
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()
    print(f'loss: {running_loss}')


def test(loader, model):
    model.eval()
    correct = 0
    for data in loader:
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct / loader.n_samples