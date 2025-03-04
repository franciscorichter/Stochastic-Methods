import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import pandas as pd

# Set a pleasing style for the plots
sns.set(style="whitegrid")

st.title("Probability Distribution Generator and CLT Simulator")

# Create two tabs: one for distributions and one for CLT simulation
tabs = st.tabs(["Probability Distributions", "Central Limit Theorem"])

###############################################################################
# Tab 1: Probability Distributions
###############################################################################
with tabs[0]:
    st.header("Explore Probability Distributions")
    distribution = st.selectbox("Select Distribution", 
                                  ("Geometric", "Binomial", "Poisson", "Exponential", "Normal"))
    sample_size = st.number_input("Sample Size", min_value=100, value=10000, step=100)

    if distribution == "Geometric":
        p = st.slider("Probability of Success (p)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
        data = np.random.geometric(p, sample_size)
        k = np.arange(1, np.max(data) + 1)
        pmf = (1 - p) ** (k - 1) * p

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=np.arange(0.5, np.max(data) + 1.5, 1), density=True,
                alpha=0.6, color='g', label="Empirical")
        ax.plot(k, pmf, 'bo', ms=8, label="Theoretical PMF")
        ax.vlines(k, 0, pmf, colors='b', lw=2)
        ax.set_xlabel("Number of Trials Until First Success")
        ax.set_ylabel("Probability")
        ax.set_title(f"Geometric Distribution (p = {p})")
        ax.legend()

    elif distribution == "Binomial":
        n = st.number_input("Number of Trials (n)", min_value=1, value=10, step=1)
        p = st.slider("Probability of Success (p)", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
        data = np.random.binomial(n, p, sample_size)
        k = np.arange(0, n + 1)
        pmf = stats.binom.pmf(k, n, p)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=np.arange(-0.5, n + 1.5, 1), density=True,
                alpha=0.6, color='orange', label="Empirical")
        ax.plot(k, pmf, 'ro', ms=8, label="Theoretical PMF")
        ax.vlines(k, 0, pmf, colors='r', lw=2)
        ax.set_xlabel("Number of Successes")
        ax.set_ylabel("Probability")
        ax.set_title(f"Binomial Distribution (n = {n}, p = {p})")
        ax.legend()

    elif distribution == "Poisson":
        lmbda = st.number_input("Lambda (λ)", min_value=0.1, value=3.0, step=0.1)
        data = np.random.poisson(lmbda, sample_size)
        k = np.arange(0, np.max(data) + 1)
        pmf = stats.poisson.pmf(k, lmbda)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=np.arange(-0.5, np.max(data) + 1.5, 1), density=True,
                alpha=0.6, color='purple', label="Empirical")
        ax.plot(k, pmf, 'go', ms=8, label="Theoretical PMF")
        ax.vlines(k, 0, pmf, colors='g', lw=2)
        ax.set_xlabel("Number of Events")
        ax.set_ylabel("Probability")
        ax.set_title(f"Poisson Distribution (λ = {lmbda})")
        ax.legend()

    elif distribution == "Exponential":
        lmbda = st.number_input("Lambda (λ)", min_value=0.01, value=0.1, step=0.01)
        data = np.random.exponential(1/lmbda, sample_size)
        x = np.linspace(0, np.percentile(data, 99), 1000)
        pdf = lmbda * np.exp(-lmbda * x)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=50, density=True, alpha=0.6, color='skyblue', label="Empirical")
        ax.plot(x, pdf, 'r-', lw=2, label="Theoretical PDF")
        ax.set_xlabel("Time Between Events")
        ax.set_ylabel("Density")
        ax.set_title(f"Exponential Distribution (λ = {lmbda})")
        ax.legend()

    elif distribution == "Normal":
        mu = st.number_input("Mean (μ)", value=0.0, step=0.1)
        sigma = st.number_input("Standard Deviation (σ)", min_value=0.1, value=1.0, step=0.1)
        data = np.random.normal(mu, sigma, sample_size)
        x = np.linspace(np.min(data), np.max(data), 1000)
        pdf = stats.norm.pdf(x, mu, sigma)

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(data, bins=50, density=True, alpha=0.6, color='gray', label="Empirical")
        ax.plot(x, pdf, 'b-', lw=2, label="Theoretical PDF")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        ax.set_title(f"Normal Distribution (μ = {mu}, σ = {sigma})")
        ax.legend()

    st.pyplot(fig)
    
    st.markdown("### Download Generated Data")
    df = pd.DataFrame(data, columns=["Data"])
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name='generated_data.csv',
        mime='text/csv',
    )

###############################################################################
# Tab 2: Central Limit Theorem (CLT) Simulation
###############################################################################
with tabs[1]:
    st.header("Central Limit Theorem Simulation")
    st.markdown(
        "The Central Limit Theorem (CLT) states that the sum (or average) of a large number of independent, identically distributed (i.i.d.) random variables with finite mean and variance converges in distribution to a normal distribution. "
        "In this tab, you can choose a base random variable and simulate the CLT by summing a specified number of independent samples from that distribution, then normalizing the sum."
    )
    
    base_dist = st.selectbox("Select Base Distribution for CLT", 
                               ("Uniform (0,1)", "Exponential", "Poisson", "Geometric"))
    num_summands = st.number_input("Number of Summands", min_value=1, value=30, step=1)
    sample_size_clt = st.number_input("Sample Size for CLT Simulation", min_value=100, value=10000, step=100)
    
    # Set up the base distribution parameters and generate data accordingly
    if base_dist == "Uniform (0,1)":
        mean_val = 0.5
        var_val = 1/12
        base_data = np.random.uniform(0, 1, (sample_size_clt, num_summands))
    elif base_dist == "Exponential":
        lambda_exp = st.number_input("Lambda (λ) for Exponential", min_value=0.01, value=1.0, step=0.1)
        mean_val = 1 / lambda_exp
        var_val = 1 / (lambda_exp**2)
        base_data = np.random.exponential(mean_val, (sample_size_clt, num_summands))
    elif base_dist == "Poisson":
        lambda_pois = st.number_input("Lambda (λ) for Poisson", min_value=0.1, value=3.0, step=0.1)
        mean_val = lambda_pois
        var_val = lambda_pois
        base_data = np.random.poisson(lambda_pois, (sample_size_clt, num_summands))
    elif base_dist == "Geometric":
        p_geo = st.slider("Probability (p) for Geometric", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
        mean_val = 1 / p_geo
        var_val = (1 - p_geo) / (p_geo**2)
        base_data = np.random.geometric(p_geo, (sample_size_clt, num_summands))
    
    # Compute the sum of the base data along the summands axis
    summed_data = np.sum(base_data, axis=1)
    # Expected sum and variance of the sum
    expected_sum = num_summands * mean_val
    variance_sum = num_summands * var_val
    # Normalize the sum (CLT): subtract mean and divide by sqrt(variance)
    normalized_data = (summed_data - expected_sum) / np.sqrt(variance_sum)
    
    x_clt = np.linspace(np.min(normalized_data), np.max(normalized_data), 1000)
    pdf_clt = stats.norm.pdf(x_clt, 0, 1)
    
    fig_clt, ax_clt = plt.subplots(figsize=(8, 4))
    ax_clt.hist(normalized_data, bins=50, density=True, alpha=0.6, color='olive', label="Empirical")
    ax_clt.plot(x_clt, pdf_clt, 'k-', lw=2, label="Standard Normal PDF")
    ax_clt.set_xlabel("Normalized Sum")
    ax_clt.set_ylabel("Density")
    ax_clt.set_title("Central Limit Theorem Simulation")
    ax_clt.legend()
    st.pyplot(fig_clt)
    
    st.markdown("### Download CLT Data")
    df_clt = pd.DataFrame(normalized_data, columns=["Normalized Sum"])
    csv_clt = df_clt.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download CLT data as CSV",
        data=csv_clt,
        file_name='clt_data.csv',
        mime='text/csv',
    )
