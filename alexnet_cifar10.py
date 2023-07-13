from dataset.cifar10 import get_train_valid_loader, get_test_loader
from model.alexnet import *


# hyperparam
num_classes = 10
num_epochs = 20
batch_size = 128
learning_rate = 0.005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_loader, valid_loader = get_train_valid_loader(data_dir='./data', batch_size=batch_size,
                                                    augment=False, random_seed=1)
test_loader = get_test_loader(data_dir='./data',
                              batch_size=batch_size)
model = AlexNet(num_classes).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
# use adam or sgd optimizer
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.005)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)


# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
          .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            del images, labels, outputs

        print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))
