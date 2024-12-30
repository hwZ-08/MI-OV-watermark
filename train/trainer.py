import torch
import torch.nn.functional as functional
import time
import random
from tqdm import tqdm

def train(net, criterion, optimizer, trainloader, device):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {'loss': train_loss / len(trainloader),
            'acc': correct / total,
            'time': time.time() - start_time}


def train_on_watermark(net, criterion, optimizer, trainloader, triggers, device, k=4):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    wmidx = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if wmidx + k <= len(triggers.images):
            xt = triggers.images[wmidx: wmidx + k]
            yt = triggers.targets[wmidx: wmidx + k]
            wmidx += k
        else:
            xt = triggers.images[wmidx:]
            yt = triggers.targets[wmidx:]
            wmidx = 0
        
        xt, yt = xt.to(device), yt.to(device)

        outputs = net(torch.cat([inputs, xt], dim=0))
        targets = torch.cat([targets, yt], dim=0)

        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {'loss': train_loss / len(trainloader),
            'acc': correct / total,
            'time': time.time() - start_time}


def train_with_customset(net, criterion, optimizer, trainloader, customset, device, k=4):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    indices = list(range(len(customset)))
    random.shuffle(indices)
    start_idx = 0

    for batch_idx, (inputs, targets) in tqdm(enumerate(trainloader), total=len(trainloader)):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        if start_idx + k <= len(customset):
            xt = [customset[i][0] for i in indices[start_idx: start_idx + k]]
            yt = [customset[i][1] for i in indices[start_idx: start_idx + k]]
            start_idx += k
        else:
            xt = [customset[i][0] for i in indices[start_idx:]]
            yt = [customset[i][1] for i in indices[start_idx:]]
            start_idx = 0

        xt = torch.stack(xt, dim=0)
        yt = torch.tensor(yt)
        xt, yt = xt.to(device), yt.to(device)

        outputs = net(torch.cat([inputs, xt], dim=0))
        targets = torch.cat([targets, yt], dim=0)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {'loss': train_loss / len(trainloader),
            'acc': correct / total,
            'time': time.time() - start_time}


def test(net, criterion, testloader, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return {'loss': test_loss / len(testloader),
            'acc': (correct / total),
            'time': time.time() - start_time}


def test_on_watermark(net, criterion, triggers, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    with torch.no_grad():
        xt, targets = triggers.images, triggers.targets
        batches = len(xt) // 100 + int(len(xt) % 100 > 0)
        for batch_idx in range(batches):
            curr_inputs = xt[batch_idx * 100: min((batch_idx + 1) * 100, len(xt))].to(device)
            curr_targets = targets[batch_idx * 100: min((batch_idx + 1) * 100, len(xt))].to(device)
            outputs = net(curr_inputs)
            loss = criterion(outputs, curr_targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += curr_targets.size(0)
            correct += predicted.eq(curr_targets).sum().item()

    return {'loss': test_loss / batches,
            'acc': (correct / total),
            'time': time.time() - start_time}
