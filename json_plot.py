from matplotlib.cm import get_cmap
import plotly.graph_objects as go
import pandas as pd
import json

# Load the dataset
dataset_path = "plot_data.csv"  # Replace with the actual path to your dataset
df = pd.read_csv(dataset_path)

# Construct the full URL for images
base_url = "https://raw.githubusercontent.com/YoongiKim/CIFAR-10-images/refs/heads/master/train/"
df["image_url"] = base_url + df["url"] + ".jpg"

# Get the tab10 colormap from Matplotlib
tab10 = get_cmap("tab10")

# Map label indices to colors
num_classes = 10  # Number of distinct labels in the dataset
colors = [tab10(i / num_classes) for i in df["label_idx"]]

# Convert Matplotlib RGBA colors to Plotly-compatible hex colors
def rgba_to_hex(rgba):
    r, g, b, a = rgba[:4]
    return f"rgba({int(r * 255)}, {int(g * 255)}, {int(b * 255)}, {a})"

df["color"] = [rgba_to_hex(color) for color in colors]

# Precompute the condition for "incorrect learned human noisy labels"
df["high_both"] = (df["inclusion_prob"] > 0.6) & (df["exclusion_prob"] > 0.6) & (df["label_idx"] != df["noisy_label_idx"])

# Create the scatter plot
fig = go.Figure()

# Scatter plot for all points excluding high_both (Rest)
rest_df = df[~df["high_both"]]
fig.add_trace(
    go.Scatter(
        x=rest_df["embeddings2d_x"],
        y=rest_df["embeddings2d_y"],
        mode="markers",
        marker=dict(size=8, opacity=0.6, color=rest_df["color"]),
        hoverinfo="text",
        text=[
            f"Index: {i}<br>CIFAR-10: {row['label_string']}<br>CIFAR-10N: {row['noisy_label_string']}<br>"
            f"Incl. Prob: {row['inclusion_prob']:.2f}<br>Excl. Prob: {row['exclusion_prob']:.2f}"
            for i, row in df.loc[rest_df.index].iterrows()  # Use indices from the original DataFrame
        ],
        customdata=rest_df[
            ["image_url", "label_string", "noisy_label_string", "inclusion_prob", "exclusion_prob"]
        ].values.tolist(),
        name="Click Me!",
    )
)

# Scatter plot for "Incorrect learned human noisy labels"
high_both_df = df[df["high_both"]]
fig.add_trace(
    go.Scatter(
        x=high_both_df["embeddings2d_x"],
        y=high_both_df["embeddings2d_y"],
        mode="markers",
        marker=dict(size=8, opacity=0.6, color=high_both_df["color"], line=dict(color="black", width=2)),
        hoverinfo="text",
        text=[
            f"Index: {i}<br>CIFAR-10: {row['label_string']}<br>CIFAR-10N: {row['noisy_label_string']}<br>"
            f"Incl. Prob: {row['inclusion_prob']:.2f}<br>Excl. Prob: {row['exclusion_prob']:.2f}"
            for i, row in df.loc[high_both_df.index].iterrows()  # Use indices from the original DataFrame
        ],
        customdata=high_both_df[
            ["image_url", "label_string", "noisy_label_string", "inclusion_prob", "exclusion_prob"]
        ].values.tolist(),
        name="Incorrect learned human noisy labels",
    )
)

# Customize layout
fig.update_layout(
    plot_bgcolor="white",  # Background of the scatter plot
    paper_bgcolor="white",  # Background of the entire figure
    xaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        linecolor="black",  # Black border
        linewidth=2,  # Thickness of the border
        mirror=True,  # Apply the border to the top as well
    ),
    yaxis=dict(
        showgrid=False,
        showticklabels=False,
        zeroline=False,
        linecolor="black",  # Black border
        linewidth=2,  # Thickness of the border
        mirror=True,  # Apply the border to the right as well
    ),
    margin=dict(l=10, r=10, t=10, b=10),  # Adjust margins as needed
    legend=dict(
        itemsizing="constant",
        orientation="h",  # Horizontal legend
        x=0.5,  # Center the legend
        xanchor="center",
        y=-0.05,  # Position the legend below the plot
        yanchor="top",
        bgcolor="white",  # Ensure legend background remains white
    ),
)

# Save the figure to JSON
json_path = "plot.json"
with open(json_path, "w") as f:
    json.dump(fig.to_json(), f)

print(f"Plotly figure saved as {json_path}")
