import subprocess
import numpy as np
import tkinter as tk
from tkinter import scrolledtext, messagebox, simpledialog

def compute_eigenvalues(matrix):
    input_data = f"{len(matrix)} {len(matrix[0])}\n"  # Rows and columns
    input_data += "\n".join([" ".join(map(str, row)) for row in matrix])

    try:
        result = subprocess.run(["D:\\C++ codes\\c++ for python\\eigenvalue.exe"], input=input_data.encode(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode('utf-8').strip()

        # Check if there's an error in the C++ executable output
        if result.stderr:
            raise RuntimeError(result.stderr.decode('utf-8'))

        coefficients = list(map(float, output.split()))
        eigenvalues = np.roots(coefficients)
        return eigenvalues

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {e}")
        return []

def on_compute():
    try:
        # Retrieve matrix from entry fields
        matrix = []
        for row_widgets in matrix_entries:
            row = []
            for entry in row_widgets:
                value = entry.get().strip()
                if value:
                    row.append(float(value))
                else:
                    row.append(0.0)  # Default to 0 if empty
            matrix.append(row)
        
        matrix = np.array(matrix)

        eigenvalues = compute_eigenvalues(matrix)

        # Display eigenvalues
        text_eigenvalues.config(state=tk.NORMAL)
        text_eigenvalues.delete(1.0, tk.END)
        text_eigenvalues.insert(tk.END, "Eigenvalues:\n====================\n")
        for eig in eigenvalues:
            text_eigenvalues.insert(tk.END, f"{eig}\n")
        text_eigenvalues.config(state=tk.DISABLED)

    except Exception as e:
        messagebox.showerror("Error", f"Invalid input or error: {e}")

def create_matrix_entries(dim):
    global matrix_entries, button_compute, button_start_over
    matrix_entries = []

    # Clear any existing widgets in frame_matrix
    for widget in frame_matrix.winfo_children():
        widget.destroy()

    for i in range(dim):
        row_entries = []
        for j in range(dim):
            entry = tk.Entry(frame_matrix, width=5, bg='lightyellow')  # Change entry background color
            entry.grid(row=i, column=j, padx=5, pady=5)
            row_entries.append(entry)
        matrix_entries.append(row_entries)

    # Create compute button
    button_compute = tk.Button(frame_matrix, text="Compute Eigenvalues", command=on_compute, bg='lightblue', fg='black')  # Change button colors
    button_compute.grid(row=dim, columnspan=dim, pady=10)

    # Create start over button
    button_start_over = tk.Button(frame_matrix, text="Start Over", command=on_start_over, bg='lightblue', fg='black')  # Change button colors
    button_start_over.grid(row=dim+1, columnspan=dim, pady=10)

def on_start_over():
    global dim
    # Clear the eigenvalues display
    text_eigenvalues.config(state=tk.NORMAL)
    text_eigenvalues.delete(1.0, tk.END)
    text_eigenvalues.config(state=tk.DISABLED)
    
    dim = simpledialog.askinteger("Input", "Enter matrix dimension:", minvalue=1, maxvalue=20)
    if dim is not None:
        create_matrix_entries(dim)

# Set up the GUI
root = tk.Tk()
root.title("Sagar's Eigenvalue Calculator")

# Input frame
frame_matrix = tk.Frame(root, bg='lightgray')  # Change frame background color
frame_matrix.grid(row=0, column=0, padx=10, pady=10)

# Output frame (make it bigger by specifying width and height)
frame_output = tk.Frame(root, width=400, height=300, bg='lightgray')  # Change frame background color
frame_output.grid(row=0, column=1, padx=20, pady=20)

# Configure grid to make frame_output resize properly
frame_output.grid_propagate(False)

# Prompt user for matrix dimension
dim = simpledialog.askinteger("Input", "Enter matrix dimension:", minvalue=1, maxvalue=20)
if dim is None:
    root.destroy()
else:
    # Create matrix entry fields
    create_matrix_entries(dim)

    label_eigenvalues = tk.Label(frame_output, text="Eigenvalues:", bg='lightgray', fg='black')  # Change label colors
    label_eigenvalues.pack()

    text_eigenvalues = scrolledtext.ScrolledText(frame_output, width=48, height=15, state=tk.DISABLED, bg='white', fg='black')  # Change scrolled text colors
    text_eigenvalues.pack(expand=True, fill=tk.BOTH)

    root.mainloop()
