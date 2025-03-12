import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd
import time

# Set a pleasing style for the plots
sns.set(style="whitegrid")

st.title("Probability Distribution Generator, CLT Simulator, and Ant Foraging")

# Create three tabs
tabs = st.tabs(["Probability Distributions", "Central Limit Theorem", "Ant Foraging Simulation"])

###############################################################################
# Tab 1: Probability Distributions
###############################################################################
with tabs[0]:
    st.header("Explore Probability Distributions")
    distribution = st.selectbox("Select Distribution", ("Geometric", "Binomial", "Poisson", "Exponential", "Normal"))
    sample_size = st.number_input("Sample Size", min_value=100, value=10000, step=100)

    if distribution == "Geometric":
        p = st.slider("Probability of Success (p)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
        data = np.random.geometric(p, sample_size)
        k = np.arange(1, np.max(data) + 1)
        pmf = (1 - p) ** (k - 1) * p

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=np.arange(0.5, np.max(data) + 1.5, 1), density=True, alpha=0.6, color='g', label="Empirical")
        ax.plot(k, pmf, 'bo', ms=8, label="Theoretical PMF")
        ax.vlines(k, 0, pmf, colors='b', lw=2)
        ax.set_xlabel("Number of Trials Until First Success")
        ax.set_ylabel("Probability")
        ax.set_title(f"Geometric Distribution (p = {p})")
        ax.legend()

    st.pyplot(fig)

###############################################################################
# Tab 2: Central Limit Theorem (CLT) Simulation
###############################################################################
with tabs[1]:
    st.header("Central Limit Theorem Simulation")
    base_dist = st.selectbox("Select Base Distribution for CLT", ("Uniform (0,1)", "Exponential", "Poisson", "Geometric"))
    num_summands = st.number_input("Number of Summands", min_value=1, value=30, step=1)
    sample_size_clt = st.number_input("Sample Size for CLT Simulation", min_value=100, value=10000, step=100)

    base_data = np.random.uniform(0, 1, (sample_size_clt, num_summands))
    summed_data = np.sum(base_data, axis=1)
    normalized_data = (summed_data - np.mean(summed_data)) / np.std(summed_data)

    fig_clt, ax_clt = plt.subplots(figsize=(8, 4))
    ax_clt.hist(normalized_data, bins=50, density=True, alpha=0.6, color='olive', label="Empirical")
    ax_clt.set_title("Central Limit Theorem Simulation")
    ax_clt.legend()
    st.pyplot(fig_clt)

###############################################################################
# Tab 3: Ant Foraging Simulation with Animation
###############################################################################
with tabs[2]:
    st.header("Ant Foraging Simulation")

    # User input for simulation parameters
    grid_size = st.slider("Grid Size", min_value=10, max_value=100, value=50, step=5)
    num_ants = st.slider("Number of Ants", min_value=1, max_value=50, value=20, step=1)
    num_food = st.slider("Number of Food Items", min_value=1, max_value=50, value=10, step=1)
    num_steps = st.slider("Number of Simulation Steps", min_value=50, max_value=500, value=200, step=50)

    # Buttons for initializing and running the simulation
    if st.button("Initialize Simulation"):
        st.session_state['ants_positions'] = np.full((num_ants, 2), grid_size // 2)
        st.session_state['ants_carrying'] = np.zeros(num_ants, dtype=int)
        st.session_state['food_positions'] = np.random.randint(0, grid_size, (num_food, 2))
        st.session_state['collected_food_positions'] = []
        st.session_state['running'] = False  # Stop any ongoing simulation
        st.session_state['step'] = 0  # Reset step counter

        st.success("Simulation Initialized! Click 'Run Simulation' to start.")

    if st.button("Run Simulation"):
        if 'ants_positions' not in st.session_state:
            st.warning("Please initialize the simulation first.")
        else:
            st.session_state['running'] = True  # Start simulation
            moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)]
            nest = np.array([grid_size // 2, grid_size // 2])

            # Create a placeholder for the animated figure
            plot_placeholder = st.empty()

            for step in range(num_steps):
                if not st.session_state['running']:  # Stop simulation if user interrupts
                    break

                for i in range(num_ants):
                    pos = st.session_state['ants_positions'][i].copy()

                    # If carrying food, return to nest
                    if st.session_state['ants_carrying'][i] == 1:
                        direction = np.sign(nest - pos)
                        st.session_state['ants_positions'][i] += direction
                        if np.array_equal(st.session_state['ants_positions'][i], nest):
                            st.session_state['ants_carrying'][i] = 0  # Drop food at the nest
                            st.session_state['collected_food_positions'].append(nest)
                    else:
                        # Check if ant is at a food location
                        food_idx = next((j for j, f in enumerate(st.session_state['food_positions']) if np.array_equal(pos, f)), None)
                        if food_idx is not None:
                            st.session_state['ants_carrying'][i] = 1
                            st.session_state['food_positions'] = np.delete(st.session_state['food_positions'], food_idx, axis=0)
                        else:
                            # Move randomly
                            move = moves[np.random.randint(len(moves))]
                            st.session_state['ants_positions'][i] = np.clip(st.session_state['ants_positions'][i] + move, 0, grid_size - 1)

                # Update the same figure instead of creating a new one each time
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.scatter(nest[0], nest[1], c='red', marker='*', s=200, label="Nest")
                ax.scatter(st.session_state['ants_positions'][:, 0], st.session_state['ants_positions'][:, 1], c='blue', label="Ants")

                # Display food locations
                if len(st.session_state['food_positions']) > 0:
                    ax.scatter(st.session_state['food_positions'][:, 0], st.session_state['food_positions'][:, 1], c='green', marker='o', s=100, label="Remaining Food")
                if len(st.session_state['collected_food_positions']) > 0:
                    collected_food_positions = np.array(st.session_state['collected_food_positions'])
                    ax.scatter(collected_food_positions[:, 0], collected_food_positions[:, 1], c='orange', marker='o', s=100, label="Collected Food")

                ax.legend()
                ax.set_xlim(0, grid_size)
                ax.set_ylim(0, grid_size)
                ax.set_title(f"Step {step+1}/{num_steps}")

                # Update the plot in the same placeholder
                plot_placeholder.pyplot(fig)

                time.sleep(0.1)  # Pause for smooth animation

            st.success("Simulation Complete!")
