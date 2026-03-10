import os
import torch
import pandas as pd
import torch.nn.functional as F

from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINConv, global_mean_pool

from torch.nn import Sequential, Linear, ReLU

# ----------------------------
# Paths
# ----------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

DATA_DIR = os.path.join(REPO_ROOT, "data")
SUBMISSION_DIR = os.path.join(REPO_ROOT, "submissions")

os.makedirs(SUBMISSION_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ----------------------------
# Load MUTAG dataset
# ----------------------------

dataset = TUDataset(root=DATA_DIR, name="MUTAG")

# ----------------------------
# Load CSV splits
# ----------------------------

print("Loading CSV splits...")

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# ----------------------------
# Build train graphs
# ----------------------------

train_graphs = []

for _, row in train_df.iterrows():

    g = dataset[int(row.graph_index)]
    g.y = torch.tensor([int(row.label)])

    train_graphs.append(g)

# ----------------------------
# Build test graphs
# ----------------------------

test_graphs = []

for _, row in test_df.iterrows():

    g = dataset[int(row.graph_index)]
    test_graphs.append(g)

# ----------------------------
# DataLoaders
# ----------------------------

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

# ----------------------------
# Model
# ----------------------------

class GINModel(torch.nn.Module):

    def __init__(self, input_dim, num_classes):
        super().__init__()

        nn = Sequential(
            Linear(input_dim, 64),
            ReLU(),
            Linear(64, 64)
        )

        self.conv1 = GINConv(nn)
        self.lin = Linear(64, num_classes)

    def forward(self, data):

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return F.log_softmax(x, dim=1)

# ----------------------------
# Initialize model
# ----------------------------

model = GINModel(dataset.num_features, dataset.num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# Training
# ----------------------------

print("Training model...")

for epoch in range(50):

    model.train()
    total_loss = 0

    for data in train_loader:

        data = data.to(device)

        optimizer.zero_grad()

        out = model(data)

        loss = F.nll_loss(out, data.y)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1} | Loss {total_loss:.4f}")

# ----------------------------
# Prediction
# ----------------------------

print("Generating predictions...")

model.eval()

predictions = []

with torch.no_grad():

    for data in test_loader:

        data = data.to(device)

        out = model(data)

        pred = out.argmax(dim=1)

        predictions.extend(pred.tolist())

# ----------------------------
# Save submission
# ----------------------------

submission_path = os.path.join(SUBMISSION_DIR, "submission.csv")

pd.DataFrame({
    "graph_index": test_df.graph_index,
    "label": predictions
}).to_csv(submission_path, index=False)

print("Submission saved to:", submission_path)
