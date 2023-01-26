from avalanche.benchmarks.classic import SplitMNIST
import os

from utils import save_gif

"""
trains the network with the selected configuration and saves 
the explanations made troughout the various trainning steps 
"""
def exp0(n_experiences, strategy, gif_name):

    benchmark = SplitMNIST(
        n_experiences=n_experiences,
        fixed_class_order=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    )

    # prepare the path in wich to save the explanation plots: 
    # if there is no folder create one
    # if there is a folder empty it 
    if not os.path.exists(f"./{gif_name}_imgs/"):
        os.makedirs(f"./{gif_name}_imgs/")
    else:
        for filename in os.listdir(f"./{gif_name}_imgs/"):
            os.remove(f"./{gif_name}_imgs/{filename}")

    # if the gif already exists, remove it
    if os.path.exists(f"{gif_name}.gif"):
        os.remove(f"{gif_name}.gif")

    print("Starting experiment...")

    for experience in benchmark.train_stream:
        print("Start of experience: ", experience.current_experience)
        print("Current Classes: ", experience.classes_in_this_experience)
        print("number of samples:", len(experience.dataset))

        # images are automatically saved saved during training through 
        # the ShapMetric object that was previously attached to the strategy
        strategy.train(experience)

    save_gif(experiment_name=gif_name, duration=0.8, n_experiences=n_experiences)
