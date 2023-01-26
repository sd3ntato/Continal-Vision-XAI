from avalanche.benchmarks.classic import SplitMNIST
from avalanche.evaluation import Metric
import matplotlib.pyplot as plt
import shap
import torch
import numpy as np
import imageio
import os


def prepare_shap_samples_MNIST():
    benchmark = SplitMNIST(
        n_experiences=10, fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )

    backgrounds = [None] * 10
    test_images = [None] * 10

    for i, experience in enumerate(benchmark.test_stream):
        backgrounds[i] = torch.stack([experience.dataset[i][0] for i in range(10)])
        test_images[i] = experience.dataset[11][0]

    background = torch.cat(backgrounds)
    test_images = torch.stack(test_images)

    return background, test_images


def save_gif(experiment_name, duration=0.5, n_experiences=10):
    images = []
    i = 0
    for i in range(len(os.listdir(f"./{experiment_name}_imgs/"))):
        print(f"appending image {i}")
        images.append(imageio.imread(f"./{experiment_name}_imgs/plot_{i}.png"))
        i += 1

    print(f"saving {len(images)} images in a gif")
    imageio.mimsave(f"./{experiment_name}.gif", images, format="GIF", duration=duration)


def save_model(model):
    torch.save(model.state_dict(), "./models/model1")


def save_shap_image_plot(e, test_images, i, n_epochs, gif_name):
    shap_values = e.shap_values(test_images)
    shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values]
    test_numpy = np.swapaxes(np.swapaxes(test_images.numpy(), 1, -1), 1, 2)

    shap.plots.image(shap_numpy, -test_numpy, show=False)
    fig = plt.gcf()
    fig.suptitle(f"experience:{i//n_epochs}, epoch:{i%n_epochs}", fontsize="xx-large")
    plt.savefig(f"{gif_name}_imgs/plot_{i}.png")


# a standalone metric implementation
class ShapMetric(Metric[float]):
    def __init__(self, background, test_images, gif_name, model=None, n_epochs=None):
        super(ShapMetric, self).__init__()
        self.background = background
        self.test_images = test_images
        self.model = model
        self.e = shap.DeepExplainer(self.model, self.background)
        self.gif_name = gif_name
        self.n_epochs = n_epochs
        self.i = 0
        pass

    def update(self):
        pass

    def result(self) -> float:
        return 0

    def reset(self):
        pass

    def after_training_epoch(self, strategy):
        save_shap_image_plot(
            self.e, self.test_images, self.i, self.n_epochs, self.gif_name
        )
        self.i += 1
        pass


# to load the model:

# model.load_state_dict(torch.load('./models/model1'))
# model.eval()
