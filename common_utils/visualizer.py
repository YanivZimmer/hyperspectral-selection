import numpy as np
import matplotlib.pyplot as plt
import os.path
from PIL import Image
import uuid
import json


class Visualizer:
    def gates_progression_image(self, dataset, matrix, fold, lam, one_gates):
        plt.clf()
        plt.imshow(matrix, cmap='gray_r', vmin=0, vmax=1)
        plt.axis('off')  # Turn off axis labels and ticks
        plt.savefig(f'{dataset}/grayscale_image_gates_{one_gates}_fold_{fold}_lam_{lam}_guid_{str(uuid.uuid4())}.png')

    def gates_progression_image_pil(self, dataset, matrix, fold, lam, one_gates):
        # Scale the matrix values to the range [0, 255]
        scaled_matrix = ((1-matrix) * 255).astype(np.uint8)

        # Create a Pillow image from the matrix
        image = Image.fromarray(scaled_matrix, mode='L')

        # Save or display the image
        image.save(f'{dataset}/grayscale_image_gates_{one_gates}_fold_{fold}_lam_{lam}_guid_{str(uuid.uuid4())}.png')

    def write_last_gates(self, dataset, gates, one_gates):
        dir = f'{dataset}/gates'
        data = {one_gates: gates}
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(f'{dir}/final_gates_{one_gates}_guid_{str(uuid.uuid4())}.json', "w") as f:
            json.dump(data, f, ensure_ascii=False)

    def save_acc_plot(self, dataset, train, val, algo_name, fold, lam, optimizer):
        x_values = np.arange(len(train))
        plt.clf()
        plt.plot(x_values, train, label=f'Train_{fold}')
        plt.plot(x_values, val, label=f'Test_{fold}')
        title=f'acc_plot_{algo_name}_lam_{lam}_fold_{fold}_opt{optimizer.__class__.__name__}'
        # Add labels and title
        plt.xlabel('Index')
        plt.ylabel('Values')
        plt.title(title)
        # Add a legend
        plt.legend()
        # Save the plot
        plt.savefig(f'{dataset}/{title}_{str(uuid.uuid4())}.png')