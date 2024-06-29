import numpy as np
import plotly.graph_objects as go

# Define custom colorscale mimicking "jet" colormap
colors = [
    [0, "rgb(0,0,128)"],   # Dark blue
    [0.25, "rgb(0,0,255)"], # Blue
    [0.5, "rgb(0,255,255)"], # Cyan
    [0.75, "rgb(255,255,0)"], # Yellow
    [1, "rgb(255,0,0)"]   # Red
]

res = 51
# Generate data
x, y, z = np.mgrid[-1:1:res*1j, -1:1:res*1j, -1:1:res*1j]
r = np.sqrt(x**2+y**2+z**2)*15

f = (abs(r**2*np.exp(-r/2)*(2*z**2-x**2-y**2)))**2   # 3dz
#f = (abs(y*(3*x**2-y**2)*r**3*np.exp(-r/2)))**2    #4f
#f = (abs((3024 - 2016*r + 432*r**2 - 36*r**3 + r**4)*r**2*np.exp(-r/2)*(x**2-y**2)))**2

# Create the plot
fig = go.Figure(data=go.Volume(
    x=x.flatten(),
    y=y.flatten(),
    z=z.flatten(),
    value=f.flatten(),
    isomin=0,  # Lower bound of data range
    isomax=np.max(f),  # Upper bound of data range
    opacity=0.1,  # Opacity of the plot
    surface_count=25,  # Number of isosurfaces
    colorscale=colors  # Use custom colorscale
    ))
    
# Update plot layout including title and centering the plot
fig.update_layout(
    title="3D Density Plot",  # Update the title
    scene=dict(
        xaxis=dict(title='X'),
        yaxis=dict(title='Y'),
        zaxis=dict(title='Z'),
        aspectratio=dict(x=1, y=1, z=1),
        ),
    width=700,
    height=700,  # Adjust height to center the plot vertically
    margin=dict(autoexpand=True),  # Allow the plot to expand to fit the available space
)

# Show the plot
fig.show(renderer="browser")
