import itertools
import logging
from pathlib import Path
from typing import List, Literal, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pymimir as mi
import torch
import wandb
from matplotlib.animation import FFMpegWriter, FuncAnimation
from torchrl.record import WandbLogger


def entropy(tensor):
    # Compute the entropy for each decision
    # we normalize such that the entropy is in [0,1]
    return torch.distributions.Categorical(probs=tensor).entropy() / np.log(len(tensor))


class RLExperiment:

    def __init__(
        self,
        run_name: str | Path,
        blocks_instance: Literal["small", "medium", "large"] = "medium",
        logger: Optional[WandbLogger] = None,
    ):
        plt.set_loglevel("warning")  # avoid info logging for animations
        self.logger = logger
        self.blocks_instance = blocks_instance
        self.run_name = run_name

        self.blocks_domain = mi.DomainParser(
            "../test/pddl_instances/blocks/domain.pddl"
        ).parse()
        self.blocks_problem = mi.ProblemParser(
            f"../test/pddl_instances/blocks/{blocks_instance}.pddl"
        ).parse(self.blocks_domain)
        self.space = mi.StateSpace.new(
            self.blocks_problem, mi.GroundedSuccessorGenerator(self.blocks_problem)
        )

        self.out_dir = (
            Path(f"../out/{blocks_instance}/{run_name}/")
            if logger is None
            else Path(logger.log_dir)
        )
        self.values = torch.load(
            self.out_dir / "values.pt", map_location=torch.device("cpu")
        )
        self.total_iterations = len(self.values)
        self.distances = [
            self.space.get_distance_to_goal_state(s) for s in self.space.get_states()
        ]
        self.optimal_values = self._calculate_optimal_discounted_distances()
        self.sqavg_values = (self.values - self.optimal_values).square()
        self.mse_loss = torch.mean(self.sqavg_values, dim=1)
        self.l1_values = torch.abs(self.values - self.optimal_values)

        self.one_before_goal = next(
            i
            for i, s in enumerate(self.space.get_states())
            if self.space.get_distance_to_goal_state(s) == 1
        )
        self.initial_state_idx = self.space.get_states().index(
            self.space.get_initial_state()
        )
        graph, goal_indices = self._construct_graph()
        self.graph: nx.DiGraph = graph
        self.goal_indices: set[int] = goal_indices
        self.pos = nx.spring_layout(graph, seed=42, iterations=2000)
        if (self.out_dir / "probs.pt").exists():
            self.probs_list_of_nested: List[torch.nested.Tensor] = torch.load(
                self.out_dir / "probs.pt", map_location=torch.device("cpu")
            )
        else:
            self.probs_list_of_nested = None

        loss_files = self.out_dir.glob("loss_*")
        self.losses: dict[str, torch.Tensor] = {
            file.stem: torch.load(file, map_location=torch.device("cpu"))
            for file in loss_files
        }
        # The indices of the best actions for every state
        self.best_actions_indices: dict[mi.State, set[int]] = {
            s: self._idx_of_best_transition(self.space.get_forward_transitions(s))
            for s in self.space.get_states()
        }
        self.best_actions_by_state_idx: dict[int, set[int]] = {
            i: self.best_actions_indices[state]
            for (i, state) in enumerate(self.space.get_states())
        }
        self.action_indices: list[list[int]] = torch.load(
            self.out_dir / "actions.pt", map_location=torch.device("cpu")
        )

    def _state(self, idx: int) -> mi.State:
        return self.space.get_states()[idx]

    def has_probs(self) -> bool:
        return self.probs_list_of_nested is not None

    def _idx_of_best_transition(self, transitions) -> set[int]:
        dis_to_goal = [
            self.space.get_distance_to_goal_state(t.target) for t in transitions
        ]
        best_d = min(dis_to_goal)
        best_indices = set(i for i, d in enumerate(dis_to_goal) if d == best_d)
        return best_indices

    def _calculate_optimal_discounted_distances(self, gamma=0.9):

        distances = torch.tensor(
            [self.space.get_distance_to_goal_state(s) for s in self.space.get_states()],
            dtype=torch.int,
        )
        optimal_values = -(1 - gamma**distances) / (1 - gamma)
        return optimal_values

    def _construct_graph(self):
        graph = nx.DiGraph()
        goal_indices = set()
        for idx, state in enumerate(self.space.get_states()):
            if self.space.is_goal_state(state):
                goal_indices.add(idx)
            graph.add_node(state.__repr__(), idx=idx)
        for state in self.space.get_states():
            for t in self.space.get_forward_transitions(state):
                graph.add_edge(
                    t.source.__repr__(),
                    t.target.__repr__(),
                    action=t.action.schema.name,
                )
        return graph, goal_indices

    def plot_optimal_distances_histogram(self):
        """Plots the distribution of goal distances in the state space."""
        unique_distances = len(set(self.distances))
        plt.xlabel("Distance to goal state")
        plt.ylabel("Number of states")
        plt.hist(
            self.distances, bins=np.arange(unique_distances + 1) - 0.5, rwidth=0.95
        )
        plt.xticks(range(unique_distances))
        plt.show()

    # LOSSES

    def plot_l1_loss_for_initial_and_one_before_goal(self, start_it=0):
        fig, (ax_init, ax_one_before) = plt.subplots(1, 2, figsize=(12, 5))

        ax_init.scatter(
            torch.arange(start_it, self.total_iterations),
            self.l1_values[start_it:, self.initial_state_idx],
            s=0.4,
        )
        ax_init.set_title("L1 for initial state")

        ax_one_before.scatter(
            torch.arange(start_it, self.total_iterations),
            self.l1_values[start_it:, self.one_before_goal],
            s=0.4,
            c="tab:orange",
        )
        ax_one_before.set_title("L1 for last state before goal")

        fig.suptitle("L1 loss torch.abs(values - optimal_values)")
        plt.show()

    def plot_predicted_values_vs_optimal_values(self, start_it=0):
        plt.scatter(
            torch.arange(start_it, self.total_iterations),
            self.mse_loss[start_it:],
            s=0.4,
        )
        plt.title("MSE over time")
        plt.show()

    def plot_logged_losses(self):
        fig, ax = plt.subplots()
        ax.set_title("Loss over time")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Loss")
        for name, loss in self.losses.items():
            ax.plot(torch.arange(len(loss)), loss, label=name)
        plt.legend(loc="upper right")
        plt.show()

    # EXTRAS
    def plot_epsilon(self, epsilon_file_name="epsilon.pt"):
        if not (self.out_dir / epsilon_file_name).exists():
            return
        epsilon_steps: torch.Tensor = torch.load(
            self.out_dir / "epsilon.pt", map_location=torch.device("cpu")
        )
        epsilon_per_iteration_step = epsilon_steps.count_nonzero(dim=1).view(-1)
        # visualize the number of random steps over time as bar chart
        plt.bar(range(self.total_iterations), epsilon_per_iteration_step)
        # plt.bar(range(total_iterations), [epsilon_over_time[i]*124 for i in range(total_iterations)], alpha=0.5)
        plt.title("Number of random steps over time")
        plt.show()

    def plot_done_sample(self):
        done_samples = torch.load(
            self.out_dir / "done_samples.pt", map_location=torch.device("cpu")
        )
        accumulated_done_samples = torch.tensor(done_samples, dtype=torch.int).cumsum(
            dim=0
        )
        # %%
        plt.plot(range(self.total_iterations), accumulated_done_samples)
        plt.title("Number of encountered goal states accumulated over time")
        plt.show()

    def plot_policy_precision_over_time(self):
        if not (self.out_dir / "actions.pt").exists():
            return

        def number_of_optimal_actions(timestep):
            return sum(
                selected_action_idx in self.best_actions_by_state_idx[state_idx]
                for state_idx, selected_action_idx in enumerate(
                    self.action_indices[timestep]
                )
            )

        num_selected_best_action = [
            number_of_optimal_actions(time) for time in range(self.total_iterations)
        ]
        fraction_best_action = [
            num_opt_actions / self.space.num_states()
            for num_opt_actions in num_selected_best_action
        ]

        fig, ax = plt.subplots()
        ax.scatter(range(self.total_iterations), fraction_best_action, s=0.4)
        ax.set_title("Policy precision over time (1.0 is optimal)")
        ax.set_xlabel("Time step")
        ax.set_ylabel("Fraction of best actions selected")
        plt.show()

    def plot_policy_certainty(self):
        if not self.has_probs():
            return

        plt.scatter(
            range(self.total_iterations),
            [
                entropy(probs_at_t[self.one_before_goal])
                for probs_at_t in self.probs_list_of_nested
            ],
            s=0.4,
            label="Last state before goal",
        )
        plt.scatter(
            range(self.total_iterations),
            [
                entropy(probs_at_t[self.initial_state_idx])
                for probs_at_t in self.probs_list_of_nested
            ],
            s=0.4,
            c="tab:orange",
            label="Initial state",
        )
        plt.title("Certainty of policy (0 is 100% certain, 1 is uniform distribution)")
        plt.xlabel("Time step")
        plt.ylabel("Entropy for initial state")
        plt.legend(loc="upper right")
        plt.show()

    def plot_gradients(self, gradient_name, state_index: Optional[int] = None):
        gradients = dict()
        gradients_dir = self.out_dir / "gradients"
        for file in gradients_dir.iterdir():
            name = file.stem
            gradients[name] = torch.load(file, map_location=torch.device("cpu"))
        gradients.keys()

        if state_index is None:
            fig, ax = plt.subplots()
            ax.set_title("Gradients over time")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Gradient")
            ax.plot(
                torch.arange(0, len(gradients[gradient_name])),
                gradients[gradient_name].squeeze(),
            )
            plt.show()
        else:
            fig, ax = plt.subplots()
            ax.set_title(f"Gradients over time for weight: {state_index}")
            ax.set_xlabel("Time step")
            ax.set_ylabel("Gradient")
            ax.scatter(
                torch.arange(0, len(gradients[gradient_name])),
                gradients[gradient_name].squeeze()[:, state_index],
                s=0.4,
            )
            plt.show()

    # ANIMATIONS
    def _update_graph_plot(self, time, scaled_cmap, axis, value_tensor):
        axis.clear()
        # Normalize the values
        # Map node colors
        node_colors = [
            scaled_cmap.to_rgba(value_tensor[time, attr["idx"]].item())
            for _, attr in self.graph.nodes.data()
        ]

        # Draw the graph
        nx.draw_networkx(
            self.graph,
            self.pos,
            node_color=node_colors,
            nodelist=[n for n in self.graph.nodes],
            with_labels=False,
            node_size=100,
            arrowstyle="-",  # no arrows
            ax=axis,
        )

        axis.text(
            0.95,
            0.95,
            f"Time step: {time}",
            horizontalalignment="right",
            verticalalignment="top",
            transform=axis.transAxes,
            fontsize=12,
            bbox=dict(facecolor="white", alpha=0.5),
        )

    def _update_histogram(self, iteration, axis, bins, value_tensor):
        axis.clear()
        axis.hist(value_tensor[iteration], bins=bins)
        axis.hist(self.optimal_values.abs(), bins=bins, alpha=0.2, color="tab:orange")
        axis.set_title("Histogram of predicted state values")
        axis.set_xticks(bins - 0.5)
        axis.tick_params(axis="x", labelrotation=90)
        axis.set_xlabel("Discounted optimal values")
        axis.set_ylabel("Number of states with predicted absolute value")
        axis.set_ylim(bottom=0, top=55)

    def plot_graph_values_with_hist(self):
        fig, (value_axis, hist_axis) = plt.subplots(figsize=(16, 7), ncols=2)
        max_value = 12
        clamped_abs_values = torch.clamp(self.values.abs(), max=max_value)

        cmap = plt.cm.cool
        vmin = clamped_abs_values.min().item()
        vmax = clamped_abs_values.max().item()
        normalized_cmap = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        sm = matplotlib.cm.ScalarMappable(norm=normalized_cmap, cmap=cmap)

        fig.colorbar(
            sm,
            ax=value_axis,
            orientation="vertical",
            label=f"Predicted state values (clamped to {max_value})",
        )

        bins = torch.arange(0, max_value + 0.5, 0.5)

        def animate_values_and_hist(time):
            self._update_graph_plot(
                time, scaled_cmap=sm, axis=value_axis, value_tensor=clamped_abs_values
            )
            self._update_histogram(
                time, hist_axis, bins, value_tensor=clamped_abs_values
            )

        ani = FuncAnimation(
            fig,
            animate_values_and_hist,
            frames=range(0, self.total_iterations, 10),
            repeat=False,
        )

        ani.save(
            self.out_dir / "graph_values_with_hist.mp4", writer=FFMpegWriter(fps=6)
        )
        if self.logger is None:
            return
        wandb_video = wandb.Video(
            str(self.out_dir / "graph_values_with_hist.mp4"), fps=6, format="mp4"
        )
        self.logger.experiment.log({"graph_values_with_hist": wandb_video})

    def plot_graph_with_probs(self):
        if not self.has_probs():
            logging.warning("Trying to plot probabilities with probs.pt missing")
            return

        edge_list: List[List[Tuple[str, str]]] = [
            [
                (t.source.__repr__(), t.target.__repr__())
                for t in self.space.get_forward_transitions(s)
            ]
            for s in self.space.get_states()
            # Watch out goal has edges too which are not in probs_list
        ]
        edge_list = list(itertools.chain.from_iterable(edge_list))

        def update_edge_plot(time, axis):
            axis.clear()

            # Normalize the values
            # Map node colors
            probs_list = [
                [tensor.item() for tensor in trans_prob]
                for trans_prob in self.probs_list_of_nested[time]
            ]
            probs_list = list(itertools.chain.from_iterable(probs_list))

            # Draw the graph
            nx.draw_networkx_edges(
                self.graph,
                self.pos,
                ax=axis,
                node_size=100,
                edgelist=edge_list,
                alpha=probs_list,
                connectionstyle="arc3,rad=0.2",
            )

            nx.draw_networkx_nodes(
                self.graph,
                self.pos,
                ax=axis,
                node_size=90,
                node_color="none",
                linewidths=0.3,
                edgecolors="black",
            )

            axis.text(
                0.95,
                0.95,
                f"Time step: {time}",
                horizontalalignment="right",
                verticalalignment="top",
                transform=axis.transAxes,
                fontsize=12,
                bbox=dict(facecolor="white", alpha=0.5),
            )

        def update_nodes_and_edges(time, scaled_cmap, node_axis, edge_axis):
            self._update_graph_plot(
                time,
                scaled_cmap=scaled_cmap,
                axis=node_axis,
                value_tensor=self.l1_values,
            )
            update_edge_plot(time, edge_axis)

        fig, (ax_graph, ax_edges) = plt.subplots(figsize=(20, 10), ncols=2)

        fig.suptitle("L1 loss per state and policy probability over time")

        cmap = plt.cm.cool
        vmin = self.l1_values.min().item()
        vmax = self.l1_values.max().item()
        normalized_cmap = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

        sm = matplotlib.cm.ScalarMappable(norm=normalized_cmap, cmap=cmap)

        fig.colorbar(
            sm,
            ax=ax_graph,
            orientation="vertical",
            label="Normalized state value error",
        )
        # Create the animation
        ani = FuncAnimation(
            fig,
            lambda time: update_nodes_and_edges(time, sm, ax_graph, ax_edges),
            frames=range(0, self.total_iterations, 10),
            repeat=False,
        )
        # Save the animation as a GIF
        ani.save(self.out_dir / "graph_with_probs.mp4", writer=FFMpegWriter(fps=6))

        if self.logger is None:
            return
        wandb_video = wandb.Video(
            str(self.out_dir / "graph_with_probs.mp4"), fps=6, format="mp4"
        )
        self.logger.experiment.log({"graph_with_probs": wandb_video})
