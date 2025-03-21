import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

# --- Stochastic Opinion Update Function ---
def update_opinion_stochastic(o_prev, o_curr, resistance):
    """
    Update the opinion of the next speaker using a stochastic weighted update.
    
    The new opinion is computed as:
      o_new = w * o_prev + (1 - w) * o_curr,
    where w is sampled uniformly from [0, 1 - resistance].
    """
    if resistance >= 1:
        w = 0.0
    else:
        w = np.random.uniform(0, 1 - resistance)
    return w * o_prev + (1 - w) * o_curr

# --- Function to Generate Speaker Transition Matrix ---
def generate_transition_matrix(model, num_speakers):
    if model == "Random":
        P = np.random.rand(num_speakers, num_speakers)
        P = P / P.sum(axis=1, keepdims=True)
    elif model == "Disconnected Factions":
        k = num_speakers // 2
        PA = np.random.rand(k, k)
        PA = PA / PA.sum(axis=1, keepdims=True)
        PB = np.random.rand(num_speakers - k, num_speakers - k)
        PB = PB / PB.sum(axis=1, keepdims=True)
        P_top = np.hstack((PA, np.zeros((k, num_speakers - k))))
        P_bot = np.hstack((np.zeros((num_speakers - k, k)), PB))
        P = np.vstack((P_top, P_bot))
    elif model == "Hierarchical Influence":
        base = [0.5]
        for i in range(1, num_speakers):
            base.append(0.5 ** (i + 1))
        row = np.array(base)
        row = row / row.sum()
        P = np.tile(row, (num_speakers, 1))
    else:
        P = np.random.rand(num_speakers, num_speakers)
        P = P / P.sum(axis=1, keepdims=True)
    return P

# --- Streamlit App UI ---
st.title("Coupled Speaker and Opinion Dynamics Simulation")

st.sidebar.header("Simulation Parameters")
num_speakers = st.sidebar.slider("Number of Speakers", min_value=2, max_value=20, value=5, step=1)
num_iter = st.sidebar.number_input("Number of Iterations", min_value=100, max_value=10000, value=500, step=100)
resistance = st.sidebar.slider("Persuasion Resistance (0 = open-minded, 1 = total resistance)", 
                                min_value=0.0, max_value=1.0, value=0.3, step=0.01)
st.sidebar.write(f"Persuasion Resistance: {resistance:.2f}")

model_option = st.sidebar.selectbox(
    "Select Conversation Model",
    options=["Random", "Disconnected Factions", "Hierarchical Influence"]
)

# Generate the speaker transition matrix based on the selected model
P = generate_transition_matrix(model_option, num_speakers)
st.sidebar.subheader("Speaker Transition Matrix")
st.sidebar.write(P)

# --- Initialize Opinions ---
opinions = np.random.rand(num_speakers)  # opinions in [0,1]
opinion_history = [opinions.copy()]

# Choose an initial speaker at random.
current_speaker = np.random.choice(num_speakers)
speaker_history = [current_speaker]

# --- Run the Simulation ---
for _ in range(num_iter):
    next_speaker = np.random.choice(num_speakers, p=P[current_speaker])
    opinions[next_speaker] = update_opinion_stochastic(opinions[current_speaker], opinions[next_speaker], resistance)
    opinion_history.append(opinions.copy())
    speaker_history.append(next_speaker)
    current_speaker = next_speaker

# Convert opinion history to numpy array for plotting.
opinion_history = np.array(opinion_history)

# Define a color mapping for speakers using matplotlib's default cycle.
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
speaker_colors = {i: colors[i % len(colors)] for i in range(num_speakers)}

# --- Plot the Sequence of Speakers ---
fig1, ax1 = plt.subplots(figsize=(10, 4))
for t in range(1, len(speaker_history)):
    # Add 1 to speaker index for display (i.e., speakers labeled 1 to num_speakers)
    ax1.scatter(t, speaker_history[t] + 1, color=speaker_colors[speaker_history[t-1]], s=50)
ax1.set_xlabel("Time Step")
ax1.set_ylabel("Speaker")
ax1.set_title("Sequence of Speakers (Colored by Previous Speaker)")
ax1.set_yticks(np.arange(1, num_speakers + 1))
st.pyplot(fig1)

# --- Plot the Evolution of Opinions ---
fig2, ax2 = plt.subplots(figsize=(10, 6))
time_steps = np.arange(opinion_history.shape[0])
for i in range(num_speakers):
    ax2.plot(time_steps, opinion_history[:, i], label=f"Speaker {i+1}", color=speaker_colors[i])
ax2.set_xlabel("Time Step")
ax2.set_ylabel("Opinion")
ax2.set_title("Evolution of Opinions Over Time")
ax2.legend(loc="upper right")
st.pyplot(fig2)
