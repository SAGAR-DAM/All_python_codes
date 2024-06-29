import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import plot
import numpy as np

# Define the grid and function
x, y = np.mgrid[-10:10:501j, -10:10:501j]

def f(x, y, n):
    r = x**2 + y**2
    val = np.sin(r) * np.cos(n * x)
    return val

# Create a figure
fig = make_subplots(rows=1, cols=1)

# Initial data
z = f(x, y, 1)

# Create the heatmap
heatmap = go.Heatmap(
    z=z,
    x=np.linspace(-10, 10, z.shape[1]),
    y=np.linspace(-10, 10, z.shape[0]),
    colorscale='jet',
    colorbar=dict(title='Value')
)

# Add heatmap to the figure
fig.add_trace(heatmap)

# Update layout for slider
fig.update_layout(
    title='Interactive Heatmap with Slider'+"\n"+"f_n(x,y)=sin(x^2+y^2)cos(nx)",
    sliders=[{
        'active': 0,
        'currentvalue': {'prefix': 'n: '},
        'pad': {"t": 50},
        'steps': [
            {
                'label': str(i),
                'method': 'update',
                'args': [{'z': [f(x, y, i)]}]
            } for i in range(0, 11)
        ]
    }],
    xaxis=dict(scaleanchor="y", scaleratio=1),
    yaxis=dict(scaleanchor="x", scaleratio=1)
)

# Save the plot as an HTML file
plot(fig, filename='D:\\Codes\\Test folder\\interactive_plot.html')
