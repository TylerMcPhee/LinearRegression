import numpy as np
import tkinter as tk
from tkinter import messagebox
from numpy import genfromtxt

# Original linear regression functions
def compute_error_for_line_given_points(b, m, points):
    total_error = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return new_b, new_m

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for i in range(num_iterations):
        b, m = step_gradient(b, m, np.array(points), learning_rate)
    return b, m

# Tkinter GUI
class LinearRegressionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Linear Regression")

        self.learning_rate_label = tk.Label(root, text="Learning Rate:")
        self.learning_rate_label.pack()
        self.learning_rate_entry = tk.Entry(root)
        self.learning_rate_entry.pack()

        self.iterations_label = tk.Label(root, text="Iterations:")
        self.iterations_label.pack()
        self.iterations_entry = tk.Entry(root)
        self.iterations_entry.pack()

        self.run_button = tk.Button(root, text="Run Regression", command=self.run_regression)
        self.run_button.pack()

        self.results_label = tk.Label(root, text="")
        self.results_label.pack()

    def run_regression(self):
        try:
            learning_rate = float(self.learning_rate_entry.get())
            iterations = int(self.iterations_entry.get())

            # Load the data directly from LinearData.csv
            self.points = genfromtxt('LinearData.csv', delimiter=',')
            if self.points is None:
                messagebox.showerror("Error", "Failed to load data from LinearData.csv.")
                return

            initial_b = 0
            initial_m = 0

            # Run the linear regression
            b, m = gradient_descent_runner(self.points, initial_b, initial_m, learning_rate, iterations)

            # Show the result
            error = compute_error_for_line_given_points(b, m, self.points)
            self.results_label.config(text=f"Final b: {b:.4f}, Final m: {m:.4f}, Error: {error:.4f}")

            # Log results to a text file
            with open("regression_results.txt", "w") as f:
                f.write(f"Final b: {b:.4f}, Final m: {m:.4f}, Error: {error:.4f}\n")
                f.write("Regression line equation: y = {:.4f}x + {:.4f}\n".format(m, b))

            messagebox.showinfo("Results", f"Results saved to 'regression_results.txt'")

        except ValueError:
            messagebox.showerror("Error", "Invalid input for learning rate or iterations.")
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {e}")

# Initialize the Tkinter root and start the GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = LinearRegressionGUI(root)
    root.mainloop()
