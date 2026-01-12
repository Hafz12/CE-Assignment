import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Delhi Metro MOACO Optimization",
    layout="wide"
)

# =====================================================
# Load ACO Results
# =====================================================
pareto = joblib.load("aco_pareto.pkl")
convergence = joblib.load("aco_convergence.pkl")
runtime = joblib.load("aco_runtime.pkl")

pareto_df = pd.DataFrame(pareto, columns=["Total Distance", "Total Fare"])

# =====================================================
# Title & Overview
# =====================================================
st.title("🐜 Delhi Metro Multi-Objective Optimization (ACO)")

st.markdown("""
This dashboard presents results obtained using  
**Multi-Objective Ant Colony Optimization (MOACO)**  
for optimizing metro routes based on **distance and fare**.

The algorithm identifies a set of **Pareto-optimal solutions**  
that represent trade-offs between competing objectives.
""")

# =====================================================
# Sidebar Controls
# =====================================================
st.sidebar.header("⚙️ ACO Controls")

show_convergence = st.sidebar.checkbox("Show Convergence Curve", True)
show_runtime = st.sidebar.checkbox("Show Computational Efficiency", True)

solution_idx = st.sidebar.slider(
    "Select Pareto Solution",
    0,
    len(pareto_df) - 1,
    0
)

# =====================================================
# Pareto Front Visualization
# =====================================================
st.subheader("📌 Pareto Front (Distance vs Fare)")

fig, ax = plt.subplots()
ax.scatter(pareto_df["Total Distance"], pareto_df["Total Fare"])
ax.scatter(
    pareto_df.iloc[solution_idx]["Total Distance"],
    pareto_df.iloc[solution_idx]["Total Fare"],
    color="red",
    label="Selected Solution"
)
ax.set_xlabel("Total Distance (Minimize)")
ax.set_ylabel("Total Fare (Minimize)")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.caption("""
Each ant constructs candidate solutions probabilistically.
Pheromone reinforcement guides the colony toward **non-dominated solutions**.
""")

# =====================================================
# Selected Solution Metrics
# =====================================================
st.subheader("🔍 Selected Pareto Solution")

c1, c2 = st.columns(2)

c1.metric(
    "Total Distance",
    f"{pareto_df.iloc[solution_idx]['Total Distance']:.2f} km"
)
c2.metric(
    "Total Fare",
    f"₹{pareto_df.iloc[solution_idx]['Total Fare']:.2f}"
)

# =====================================================
# Convergence Analysis
# =====================================================
if show_convergence:
    st.subheader("📈 Convergence Analysis")

    fig2, ax2 = plt.subplots()
    ax2.plot(convergence)
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Number of Pareto Solutions")
    ax2.set_title("ACO Pareto Front Growth")
    ax2.grid(True)

    st.pyplot(fig2)

    st.markdown("""
    **Interpretation**:
    - Early iterations show rapid exploration.
    - Pheromone accumulation stabilizes solution quality.
    - Later iterations refine trade-offs rather than discover new extremes.
    """)

# =====================================================
# Computational Efficiency
# =====================================================
if show_runtime:
    st.subheader("⏱ Computational Efficiency")

    st.metric("Total Execution Time", f"{runtime:.3f} seconds")

    st.markdown("""
    MOACO demonstrates moderate computational cost due to:
    - Multiple ants per iteration
    - Pheromone evaporation and reinforcement
    - Pareto archive maintenance
    """)

# =====================================================
# Extended Multi-Objective Analysis
# =====================================================
st.subheader("🧠 Extended Multi-Objective Analysis")

st.markdown("""
### Objective Justification
- **Distance minimization** improves travel efficiency
- **Fare minimization** improves affordability
- These objectives are inherently conflicting

### How ACO Balances Objectives
- Uses **separate pheromone trails** per objective
- Probabilistic solution construction encourages exploration
- Pareto dominance ensures solution diversity

### Effect on Solution Quality
- Produces a **well-distributed Pareto front**
- Avoids bias introduced by weighted-sum approaches
- Enables flexible decision-making
""")

# =====================================================
# Strengths & Limitations
# =====================================================
st.subheader("⚖️ Strengths vs Limitations of ACO")

colS, colL = st.columns(2)

with colS:
    st.markdown("""
    **Strengths**
    - Naturally suited for combinatorial problems
    - Strong exploration capability
    - Robust to local optima
    - Intuitive biological inspiration
    """)

with colL:
    st.markdown("""
    **Limitations**
    - Slower convergence than PSO
    - Risk of pheromone stagnation
    - Sensitive to evaporation and colony size
    """)

# =====================================================
# Comparison with Other Algorithms
# =====================================================
st.subheader("🔄 Comparison with Other Evolutionary Algorithms")

st.markdown("""
Compared to **PSO**:
- ACO offers better exploration
- PSO converges faster but may cluster solutions

Compared to **GA (NSGA-II)**:
- ACO is simpler to implement
- GA provides stronger diversity control

ACO is particularly effective for **discrete route selection problems**.
""")

# =====================================================
# Conclusion
# =====================================================
st.subheader("✅ Conclusion")

st.markdown("""
The MOACO dashboard demonstrates how swarm intelligence  
can effectively address **multi-objective optimization** in urban transportation.

By visualizing Pareto fronts, convergence behavior, and efficiency metrics,  
this system enhances interpretability and supports informed decision-making.
""")
