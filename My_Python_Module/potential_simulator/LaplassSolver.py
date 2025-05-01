# -*- coding: utf-8 -*-
"""
Created on Wed Feb  7 11:24:13 2024

@author: mrsag
"""
# import yt
import numpy as np
import matplotlib.pyplot as plt
# import glob
# from Curve_fitting_with_scipy import Gaussianfitting as Gf
from scipy.signal import fftconvolve
# from matplotlib.offsetbox import OffsetImage, AnnotationBbox
# import matplotlib.image as mpimg
# from matplotlib.legend_handler import HandlerBase
# from matplotlib.lines import Line2D
from matplotlib.path import Path
from skimage import io
from skimage.transform import resize
from copy import deepcopy
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from mayavi import mlab


# %%
class polygon_plate:
    def __init__(self, edges, potential=0, resolution=100):
        self.edges = edges  # List of (x, y) points
        self.potential = potential
        self.resolution = resolution
        self.mask = None

    def is_inside(self, x, y):
        # Ray casting algorithm for point-in-polygon
        n = len(self.edges)
        inside = False
        px, py = x, y
        for i in range(n):
            x0, y0 = self.edges[i]
            x1, y1 = self.edges[(i + 1) % n]
            if ((y0 > py) != (y1 > py)):
                x_intersect = (x1 - x0) * (py - y0) / (y1 - y0 + 1e-10) + x0
                if px < x_intersect:
                    inside = not inside
        return inside

    def generate_mask(self, xx, yy):
        mask = np.zeros_like(xx, dtype=bool)
        for i in range(xx.shape[0]):
            for j in range(xx.shape[1]):
                if self.is_inside(xx[i, j], yy[i, j]):
                    mask[i, j] = True
        self.mask = mask
        return mask


class Simulationbox2d:
    def __init__(self, resolution_x=300, resolution_y=300, box_x=1, box_y=1,
                 b_y0=0, b_x0=0, b_y1=0, b_x1=0,
                 potential_offset=0, match_boundary=True, potential_given_on_boundary=False):

        self.box_x = box_x
        self.box_y = box_y
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.dx = np.diff(np.linspace(0,self.box_x,self.resolution_x))[0]
        self.dy = np.diff(np.linspace(0,self.box_y,self.resolution_y))[0]
        self.yy, self.xx = np.mgrid[0:self.box_y:resolution_y*1j, 0:self.box_x:resolution_x*1j]
        self.potential = np.zeros_like(self.xx) + potential_offset
        self.solved = False

        def resample_boundary(boundary, target_length):
            if isinstance(boundary, (int, float)) and boundary == 0:
                return np.zeros(target_length)
            boundary = np.array(boundary)
            x_old = np.linspace(0, 1, len(boundary))
            x_new = np.linspace(0, 1, target_length)
            return np.interp(x_new, x_old, boundary)

        # Boundary setup
        self.b_y0 = resample_boundary(b_y0, resolution_x)  # bottom edge (y=0)
        self.b_y1 = resample_boundary(b_y1, resolution_x)  # top edge    (y=1)
        self.b_x0 = resample_boundary(b_x0, resolution_y)  # left edge   (x=0)

        delta_y0 = self.b_x0[0] - self.b_y0[0]
        delta_y1 = self.b_x0[-1] - self.b_y1[0]
        self.b_y0 += delta_y0
        self.b_y1 += delta_y1

        if match_boundary:
            self.b_x1 = np.linspace(self.b_y0[-1], self.b_y1[-1], resolution_y)
        else:
            self.b_x1 = resample_boundary(b_x1, resolution_y)

        # Apply boundary values
        self.potential[0, :] = self.b_y0         # y = 0 (bottom)
        self.potential[-1, :] = self.b_y1        # y = 1 (top)
        self.potential[:, 0] = self.b_x0         # x = 0 (left)
        self.potential[:, -1] = self.b_x1        # x = 1 (right)

        self.fixed_mask = np.zeros_like(self.potential, dtype=bool)
        if potential_given_on_boundary:
            self.fixed_mask[0, :] = True
            self.fixed_mask[-1, :] = True
            self.fixed_mask[:, 0] = True
            self.fixed_mask[:, -1] = True

    def add_polygon_plate(self, poly):
        mask = poly.generate_mask(self.xx, self.yy)
        self.potential[mask] = poly.potential
        self.fixed_mask[mask] = True


    def add_disk_plate(self, center, radius, potential=0):
        """
        Adds a circular plate (disk) to the simulation domain.

        Parameters:
        - center: (x, y) tuple for center of the disk
        - radius: radius of the disk (in simulation units)
        - potential: potential to assign to the disk
        """
        cx, cy = center
        distance = np.sqrt((self.xx - cx)**2 + (self.yy - cy)**2)
        mask = distance <= radius
        self.potential[mask] = potential
        self.fixed_mask[mask] = True


    def add_image(self, image, threshold=0.5, potential=0):
        """
        Adds a fixed potential region based on a grayscale image mask.

        Parameters:
        - image_path: path to the image file
        - threshold: normalized threshold (0–1) for deciding potential region
        - potential: potential value to assign above threshold
        """
        
        # Resize to match simulation resolution
        image=np.flip(image,axis=0)
        threshold /= np.max(image)
        resized_image = resize(image, (self.resolution_x, self.resolution_y), anti_aliasing=True)

        # Create mask where intensity > threshold
        mask = resized_image > threshold

        # Apply potential and fix those positions
        self.potential[mask] = potential
        self.fixed_mask[mask] = True

        
    def solve(self, max_iterations=10000, tolerance=1e-4):
        self.solved =True
        with tqdm(total=max_iterations, desc="Jacobi Iteration", colour='green', ncols=100, dynamic_ncols=False) as pbar:
            for it in range(max_iterations):
                potential_old = self.potential.copy()

                # Jacobi update
                self.potential[1:-1, 1:-1] = 0.25 * (
                    potential_old[1:-1, 2:] +
                    potential_old[1:-1, :-2] +
                    potential_old[2:, 1:-1] +
                    potential_old[:-2, 1:-1]
                )

                # Restore fixed values
                self.potential[self.fixed_mask] = potential_old[self.fixed_mask]

                # Check for convergence
                delta = np.abs(self.potential - potential_old).max()

                # Use tqdm.write to display iteration status without interrupting the bar
                # if it % 50 == 0:
                #     tqdm.write(f"Iteration {it}, max change: {delta:.2e}")

                # Update progress bar
                pbar.set_postfix_str(f"Δ={delta:.2e}")
                pbar.update(1)

                if delta < tolerance:
                    tqdm.write(f"Converged after {it} iterations")
                    break

            # Final message after the loop
            tqdm.write(f"final Max change in Itn({it+1}) - Itn({it}): {delta:.2e}")

    def electric_field(self):
        if(not self.solved):
            self.solve(max_iterations=1000)
        
        self.Ex = -np.diff(self.potential,axis=1)/self.dx
        self.Ey = -np.diff(self.potential,axis=0)/self.dy

    def plot_potential(self,imshow=True,contourplot=True,linecontourplot=True):
        if(not self.solved):
            self.solve(max_iterations=1000)
        
        if(imshow):
            plt.imshow(self.potential, origin='lower', cmap="jet", extent=[0, self.box_x, 0, self.box_y])
        if(contourplot):
            contour = plt.contourf(self.xx, self.yy, self.potential, levels=200, cmap="jet")
        if(linecontourplot):
            contours = plt.contour(self.xx, self.yy, self.potential, levels=50, colors='black', linewidths=0.4, linestyles='solid')

        plt.title("Potential distribution")
        plt.show()

    def plot_electric_field(self,stepx=10,stepy=10,scale=50,remove_singularity=0,
                            cmap="jet",grid=False,cut_edge=False,
                            colorbar=True,xlim=None,ylim=None):
        self.electric_field()
        magnitude = np.sqrt((self.Ex[0:-1,:])**2 + (self.Ey[:,0:-1])**2)

        if(remove_singularity!=0):
            for i in range(remove_singularity):
                max_index = np.argmax(magnitude)
                multi_idx = np.unravel_index(max_index, magnitude.shape)
                (self.Ex[0:-1,:])[multi_idx[0],multi_idx[1]]=0
                (self.Ey[:,0:-1])[multi_idx[0],multi_idx[1]]=0
                magnitude[multi_idx[0],multi_idx[1]]=0
        if(not cut_edge):
            magnitude = np.sqrt((self.Ex[0:-1,:][::stepx,::stepy])**2 + (self.Ey[:,0:-1][::stepx,::stepy])**2)
            plt.quiver(self.xx[0:-1,0:-1][::stepx,::stepy], self.yy[0:-1,:-1][::stepx,::stepy], 
                    self.Ex[0:-1,:][::stepx,::stepy]*self.dx, self.Ey[:,0:-1][::stepx,::stepy]*self.dy, magnitude,
                    cmap=cmap,scale=50, scale_units='xy', angles='xy')
        if(cut_edge):
            magnitude = np.sqrt((self.Ex[2:-3,2:-2][::stepx,::stepy])**2 + (self.Ey[2:-2,2:-3][::stepx,::stepy])**2)
            plt.quiver(self.xx[2:-3,2:-3][::stepx,::stepy], self.yy[2:-3,2:-3][::stepx,::stepy], 
                    self.Ex[2:-3,2:-2][::stepx,::stepy]*self.dx, self.Ey[2:-2,2:-3][::stepx,::stepy]*self.dy, magnitude,
                    cmap=cmap,scale=scale, scale_units='xy', angles='xy')
        if(colorbar):
            plt.colorbar()
            
        plt.title('Electric field distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.axis('equal')
        if(xlim != None):
            plt.xlim(xlim)
        if(ylim != None):
            plt.ylim(ylim)
            
        plt.grid(grid)
        plt.show()

        self.electric_field()
        
    def plot_Ex_Ey_E_separately(self,plot_Ex=False,plot_Ey=False,plot_mod_E=False,cmap="jet",logscale=False,colorbar=False):
        self.electric_field()
        magnitude = np.sqrt((self.Ex[0:-1,:])**2 + (self.Ey[:,0:-1])**2)
        
        if(plot_Ex):
            if(logscale):
                plt.imshow(np.log(np.abs(self.Ex)),cmap=cmap,extent=[0, self.box_x, 0, self.box_y])
                plt.title("log(|Ex|)")
            if(not logscale):
                plt.imshow(self.Ex,cmap=cmap,extent=[0, self.box_x, 0, self.box_y])
                plt.title("Ex")
            if(colorbar):
                plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
                
        if(plot_Ey):
            if(logscale):
                plt.imshow(np.log(np.abs(self.Ey)),cmap=cmap,extent=[0, self.box_x, 0, self.box_y])
                plt.title("log(|Ey|)")
            if(not logscale):
                plt.imshow(self.Ey,cmap=cmap,extent=[0, self.box_x, 0, self.box_y])
                plt.title("Ey")
            if(colorbar):
                plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()
                
        if(plot_mod_E):
            if(logscale):
                plt.imshow(np.log(magnitude),cmap=cmap,extent=[0, self.box_x, 0, self.box_y])
                plt.title("log(|E|)")
            if(not logscale):
                plt.imshow(magnitude,cmap=cmap,extent=[0, self.box_x, 0, self.box_y])
                plt.title("|E|")
            if(colorbar):
                plt.colorbar()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.show()


# %%


class SimulationBox3D:
    def __init__(self, resolution_x=50, resolution_y=50, resolution_z=50,
                 box_x=1.0, box_y=1.0, box_z=1.0, potential_offset=0):
        self.resolution_x = resolution_x
        self.resolution_y = resolution_y
        self.resolution_z = resolution_z

        self.box_x = box_x
        self.box_y = box_y
        self.box_z = box_z

        self.dx = np.diff(np.linspace(0,box_x,resolution_x))[0]
        self.dy = np.diff(np.linspace(0,box_y,resolution_y))[0]
        self.dz = np.diff(np.linspace(0,box_z,resolution_z))[0]

        # Create a proper Cartesian grid
        self.x, self.y, self.z = np.meshgrid(
            np.linspace(0, box_x, resolution_x),
            np.linspace(0, box_y, resolution_y),
            np.linspace(0, box_z, resolution_z),
            indexing='ij'
        )
        self.solved = False
        self.potential = np.full_like(self.x, potential_offset, dtype=float)
        self.fixed_mask = np.zeros_like(self.potential, dtype=bool)

    def add_sphere(self, center, radius, potential=0):
        cx, cy, cz = center
        mask = ((self.x - cx)**2 + (self.y - cy)**2 + (self.z - cz)**2) <= radius**2
        self.potential[mask] = potential
        self.fixed_mask[mask] = True

    def add_box(self, x_bounds, y_bounds, z_bounds, potential=0):
        x0, x1 = x_bounds
        y0, y1 = y_bounds
        z0, z1 = z_bounds
        mask = ((self.x >= x0) & (self.x <= x1) &
                (self.y >= y0) & (self.y <= y1) &
                (self.z >= z0) & (self.z <= z1))
        self.potential[mask] = potential
        self.fixed_mask[mask] = True

    def add_cylinder(self, base_center, radius, height, axis='z', potential=0):
        cx, cy, cz = base_center
        if axis == 'z':
            mask = (((self.x - cx)**2 + (self.y - cy)**2 <= radius**2) &
                    (self.z >= cz) & (self.z <= cz + height))
        elif axis == 'x':
            mask = (((self.y - cy)**2 + (self.z - cz)**2 <= radius**2) &
                    (self.x >= cx) & (self.x <= cx + height))
        elif axis == 'y':
            mask = (((self.x - cx)**2 + (self.z - cz)**2 <= radius**2) &
                    (self.y >= cy) & (self.y <= cy + height))
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        self.potential[mask] = potential
        self.fixed_mask[mask] = True

    def add_hollow_pipe(self, base_center, radius, thickness, height, axis='z', potential=0):
        cx, cy, cz = base_center
        if axis == 'z':
            mask = (((self.x - cx)**2 + (self.y - cy)**2 <= (radius+thickness/2)**2) &
                    ((self.x - cx)**2 + (self.y - cy)**2 >= (radius-thickness/2)**2)
                    (self.z >= cz) & (self.z <= cz + height))
        elif axis == 'x':
            mask = (((self.y - cy)**2 + (self.z - cz)**2 <= (radius+thickness/2)**2) &
                    ((self.y - cy)**2 + (self.z - cz)**2 >= (radius-thickness/2)**2) &
                    (self.x >= cx) & (self.x <= cx + height))
        elif axis == 'y':
            mask = (((self.x - cx)**2 + (self.z - cz)**2 <= (radius+thickness/2)**2) &
                    ((self.x - cx)**2 + (self.z - cz)**2 >= (radius-thickness/2)**2)
                    (self.y >= cy) & (self.y <= cy + height))
        else:
            raise ValueError("axis must be 'x', 'y', or 'z'")
        self.potential[mask] = potential
        self.fixed_mask[mask] = True

    def add_ellipsoid(self, center, radii, potential=0):
        cx, cy, cz = center
        rx, ry, rz = radii
        mask = (((self.x - cx)/rx)**2 + ((self.y - cy)/ry)**2 + ((self.z - cz)/rz)**2 <= 1)
        self.potential[mask] = potential
        self.fixed_mask[mask] = True

    def add_hyperboloid(self, center, coeffs, waist=1, axis="z", potential=0):
        cx, cy, cz = center
        a, b, c = coeffs
        if axis=="x":
            mask = (-((self.x - cx)/a)**2 + ((self.y - cy)/b)**2 + ((self.z - cz)/c)**2 <= waist**2)
        if axis=="y":
            mask = (((self.x - cx)/a)**2 - ((self.y - cy)/b)**2 + ((self.z - cz)/c)**2 <= waist**2)
        if axis=="z":
            mask = (((self.x - cx)/a)**2 + ((self.y - cy)/b)**2 - ((self.z - cz)/c)**2 <= waist**2)

        self.potential[mask] = potential
        self.fixed_mask[mask] = True

    def add_plane(self, coefficients, thickness=0.01, potential=0):
        A, B, C, D = coefficients
        norm = np.sqrt(A**2 + B**2 + C**2)
        distance = (A * self.x + B * self.y + C * self.z + D) / norm
        mask = np.abs(distance) <= thickness / 2
        self.potential[mask] = potential
        self.fixed_mask[mask] = True

    def solve(self, max_iter=1000, tol=1e-4, method='jacobi', verbose=False):
        V = self.potential.copy()
        self.solved=True
        with tqdm(total=max_iter, desc="Solver Iteration", colour='green', ncols=100, dynamic_ncols=False) as pbar:
            for it in range(max_iter):
                V_old = V.copy()
                
                if method == 'jacobi':
                    V_new = V.copy()
                    V_new[1:-1, 1:-1, 1:-1] = 1/6 * (
                        V[2:, 1:-1, 1:-1] + V[:-2, 1:-1, 1:-1] +
                        V[1:-1, 2:, 1:-1] + V[1:-1, :-2, 1:-1] +
                        V[1:-1, 1:-1, 2:] + V[1:-1, 1:-1, :-2]
                    )
                    mask = ~self.fixed_mask[1:-1, 1:-1, 1:-1]
                    V[1:-1, 1:-1, 1:-1][mask] = V_new[1:-1, 1:-1, 1:-1][mask]
                elif method == 'gauss-seidel':
                    for i in range(1, self.resolution_x - 1):
                        for j in range(1, self.resolution_y - 1):
                            for k in range(1, self.resolution_z - 1):
                                if not self.fixed_mask[i, j, k]:
                                    V[i, j, k] = 1/6 * (V[i+1, j, k] + V[i-1, j, k] +
                                                    V[i, j+1, k] + V[i, j-1, k] +
                                                    V[i, j, k+1] + V[i, j, k-1])
                else:
                    raise ValueError("Unknown method")

                diff = np.abs(V - V_old).max()
                
                # Update progress bar
                pbar.set_postfix_str(f"Δ={diff:.2e}")
                pbar.update(1)

                if diff < tol:
                    if verbose:
                        tqdm.write(f"Converged at iteration {it}, max change: {diff:.2e}")
                    break
        
        self.potential = V
        tqdm.write(f"Final Max change in iteration {it+1}: {diff:.2e}")

    def plot_potential_isosurface(self,contours=50,opacity=0.4,cmap="jet"):
        if (not self.solved):
            self.solve(max_iter=300)
            
        nx, ny, nz = self.resolution_x, self.resolution_y, self.resolution_z
        lx, ly, lz = self.box_x, self.box_y, self.box_z

        # Create coordinate grid
        x, y, z = np.mgrid[
            0:lx:nx*1j,
            0:ly:ny*1j,
            0:lz:nz*1j
        ]

        mlab.figure(bgcolor=(1, 1, 1), size=(800, 600))
        mlab.contour3d(x, y, z, self.potential, contours=contours, opacity=opacity, colormap=cmap)
        axes = mlab.axes(xlabel='X', ylabel='Y', zlabel='Z',color=(1.0,0.0,0.0))
        mlab.colorbar(title="potential", orientation='vertical')
        axes.title_text_property.color = (1.0, 0.0, 0.0)  # red title text (if titles used)
        axes.label_text_property.color = (1.0, 0.0, 0.0)  # blue label text
        mlab.title("3D Isosurface of Potential")
        mlab.show()

    def plot_potential_density(self):
        if (not self.solved):
            self.solve(max_iter=300)
        # Create physical axes
        x = np.linspace(0, self.self_x, self.resolution_x)
        y = np.linspace(0, self.self_y, self.resolution_y)
        z = np.linspace(0, self.self_z, self.resolution_z)
        x, y, z = np.meshgrid(x, y, z, indexing='ij')  # Match shape with potential

        # Setup the scalar field for volumetric rendering
        src = mlab.pipeline.scalar_field(x, y, z, self.potential)

        # Optional: Adjust vmin/vmax to emphasize low potential regions (e.g., near zero)
        vmin = np.min(self.potential)
        vmax = np.max(self.potential)
        mlab.pipeline.volume(src, vmin=vmin, vmax=vmax)
        # Add axis, title, colorbar
        mlab.axes(xlabel='X', ylabel='Y', zlabel='Z')
        mlab.colorbar(title='Potential', orientation='vertical')
        mlab.title('3D Volume Rendering of Potential')
        mlab.view(azimuth=45, elevation=60, distance='auto')

        mlab.show()


    def electric_field(self):
        if(not self.solved):
            self.solve(max_iter=1000)
        
        self.Ex = -np.diff(self.potential,axis=0)/self.dx
        self.Ey = -np.diff(self.potential,axis=1)/self.dy
        self.Ez = -np.diff(self.potential,axis=2)/self.dz



    def plot_electric_field_3d(self, stepx=5, stepy=5, stepz=5, scale_factor=1.0,
                            remove_singularity=0, cut_edge=False, color_by_magnitude=True):
        self.electric_field()

        # Align Ex, Ey, Ez to a common grid: (nx-1, ny-1, nz-1)
        self.Ex = self.Ex[:, :-1, :-1]
        self.Ey = self.Ey[:-1, :, :-1]
        self.Ez = self.Ez[:-1, :-1, :]

          # Magnitude
        magnitude = np.sqrt(self.Ex**2 + self.Ey**2 + self.Ez**2)

        # Remove singularities (large spikes)
        if remove_singularity > 0:
            for _ in range(remove_singularity):
                idx = np.argmax(magnitude)
                i, j, k = np.unravel_index(idx, magnitude.shape)
                self.Ex[i, j, k] = 0
                self.Ey[i, j, k] = 0
                self.Ez[i, j, k] = 0
                magnitude[i, j, k] = 0

        # Aligned coordinates
        X = self.x[:-1, :-1, :-1][::stepx, ::stepy, ::stepz]
        Y = self.y[:-1, :-1, :-1][::stepx, ::stepy, ::stepz]
        Z = self.z[:-1, :-1, :-1][::stepx, ::stepy, ::stepz]
        
        
        # Downsample fields
        self.Ex = self.Ex[::stepx, ::stepy, ::stepz]
        self.Ey = self.Ey[::stepx, ::stepy, ::stepz]
        self.Ez = self.Ez[::stepx, ::stepy, ::stepz]

        # Plot with Mayavi
        mlab.figure(bgcolor=(1, 1, 1))
        mlab.quiver3d(X, Y, Z, self.Ex, self.Ey, self.Ez,
                    mode='arrow',
                    colormap='jet')
        axes = mlab.axes(xlabel='X', ylabel='Y', zlabel='Z',color=(1.0,0.0,0.0))
        axes.title_text_property.color = (1.0, 0.0, 0.0)  # red title text (if titles used)
        axes.label_text_property.color = (1.0, 0.0, 0.0)  # blue label text
        if color_by_magnitude:
            mlab.colorbar(title='|E|', orientation='vertical')
        mlab.title('3D Electric Field')
        mlab.show()

        self.electric_field()




