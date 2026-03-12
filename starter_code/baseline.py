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
SUBMISSIONS_DIR = os.path.join(REPO_ROOT, "submissions")

os.makedirs(SUBMISSIONS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----------------------------
# Load MUTAG dataset
# ----------------------------

dataset = TUDataset(root=DATA_DIR, name="MUTAG")

# ----------------------------
# Load CSV splits
# ----------------------------

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

# ----------------------------
# Perturbation Function
# ----------------------------

def perturb_graph(data, feature_shift=0.3, noise_std=0.05):
    data = data.clone()
    if data.x is not None:
        # Distribution shift
        shift = torch.full_like(data.x, feature_shift)
        data.x = data.x + shift
        # Gaussian noise
        noise = torch.randn_like(data.x) * noise_std
        data.x = data.x + noise
    return data

# ----------------------------
# Build graph lists
# ----------------------------

train_graphs = []
ideal_test_graphs = []
perturbed_test_graphs = []

for _, row in train_df.iterrows():

    g = dataset[int(row.graph_index)]
    g.y = torch.tensor([int(row.label)])

    train_graphs.append(g)

for _, row in test_df.iterrows():

    g = dataset[int(row.graph_index)]

    ideal_test_graphs.append(g)

    perturbed_test_graphs.append(perturb_graph(g))

# ----------------------------
# DataLoaders
# ----------------------------

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)

ideal_test_loader = DataLoader(ideal_test_graphs, batch_size=32)

perturbed_test_loader = DataLoader(perturbed_test_graphs, batch_size=32)

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
# Initialize Model
# ----------------------------

model = GINModel(dataset.num_features, dataset.num_classes).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ----------------------------
# Training
# ----------------------------

print("Training on IDEAL graphs...")

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
# Prediction Function
# ----------------------------

def predict(model, loader):

    model.eval()

    preds = []

    with torch.no_grad():

        for data in loader:

            data = data.to(device)

            out = model(data)

            preds.extend(out.argmax(dim=1).tolist())

    return preds

# ----------------------------
# Predictions
# ----------------------------

print("Generating IDEAL predictions...")
ideal_predictions = predict(model, ideal_test_loader)

print("Generating PERTURBED predictions...")
perturbed_predictions = predict(model, perturbed_test_loader)

# ----------------------------
# Save Submissions
# ----------------------------

ideal_path = os.path.join(SUBMISSIONS_DIR, "ideal_submission.csv")
perturbed_path = os.path.join(SUBMISSIONS_DIR, "perturbed_submission.csv")

pd.DataFrame({
    "graph_index": test_df.graph_index,
    "target": ideal_predictions
}).to_csv(ideal_path, index=False)

pd.DataFrame({
    "graph_index": test_df.graph_index,
    "target": perturbed_predictions
}).to_csv(perturbed_path, index=False)

print("Saved ideal submission:", ideal_path)
print("Saved perturbed submission:", perturbed_path)
