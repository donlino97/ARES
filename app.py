import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="ARES Framework",
    layout="wide"
)

st.title("🔐 ARES Framework – Security Domain Classification")
st.markdown(
    """
    Evaluate and rank security domains based on:
    - Risk factors
    - Vulnerability
    - Relevance
    - Utility
    - Interdependencies
    """
)

# ---------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------
def compute_risk_weights(risk_params):
    R = {
        o: (
            p["lambda"]
            * p["iota"]
            * p["epsilon"]
            * p["assurance"]
            * p["localization"]
        )
        for o, p in risk_params.items()
    }

    total = sum(R.values())

    if total == 0:
        return {o: 1 / len(R) for o in R}

    return {o: R[o] / total for o in R}


def compute_base_eta(topics, objectives, r, u, v, beta):
    return {
        (t, o): (
            beta * r.get((t, o), 0)
            + (1 - beta) * v.get(t, 0) * u.get((t, o), 0)
        )
        for t in topics
        for o in objectives
    }


def compute_adjusted_eta(topics, objectives, eta, D, alpha):
    idx = {t: i for i, t in enumerate(topics)}
    n = len(topics)

    M = np.eye(n)

    for (ti, tj), val in D.items():
        if ti in idx and tj in idx:
            M[idx[ti], idx[tj]] += alpha * val

    eta_matrix = np.array(
        [
            [eta[(t, o)] for o in objectives]
            for t in topics
        ]
    )

    eta_prime = M @ eta_matrix

    return {
        (topics[i], objectives[j]): eta_prime[i, j]
        for i in range(len(topics))
        for j in range(len(objectives))
    }


def compute_scores(topics, objectives, eta_prime, weights):
    return {
        t: sum(
            weights[o] * eta_prime[(t, o)]
            for o in objectives
        )
        for t in topics
    }


def classify(score):
    if score > 0.75:
        return "🔴 Critical"
    elif score > 0.60:
        return "🟠 High"
    elif score > 0.45:
        return "🟡 Medium"
    else:
        return "🟢 Low"


# ---------------------------------------------------
# PARAMETERS
# ---------------------------------------------------
st.sidebar.header("⚙️ Framework Parameters")

beta = st.sidebar.slider(
    "β (Relevance vs Vulnerability)",
    0.0,
    1.0,
    0.7,
    0.01
)

alpha = st.sidebar.slider(
    "α (Interdependency Influence)",
    0.0,
    1.0,
    0.2,
    0.01
)

objectives = [
    "Availability",
    "Physical",
    "Asset Management",
    "CIA"
]

# ---------------------------------------------------
# RISK PARAMETERS
# ---------------------------------------------------
st.sidebar.header("⚠️ Risk Parameters")

risk_params = {}

for o in objectives:
    st.sidebar.subheader(o)

    risk_params[o] = {
        "lambda": st.sidebar.slider(
            f"{o} λ",
            0.0,
            1.0,
            0.5,
            key=f"{o}_lambda"
        ),
        "iota": st.sidebar.slider(
            f"{o} ι",
            0.0,
            1.0,
            0.5,
            key=f"{o}_iota"
        ),
        "epsilon": st.sidebar.slider(
            f"{o} ε",
            0.0,
            1.0,
            0.5,
            key=f"{o}_epsilon"
        ),
        "assurance": st.sidebar.slider(
            f"{o} Assurance",
            0.0,
            1.0,
            0.5,
            key=f"{o}_assurance"
        ),
        "localization": st.sidebar.slider(
            f"{o} Localization",
            0.0,
            1.0,
            0.5,
            key=f"{o}_localization"
        )
    }

# ---------------------------------------------------
# SECURITY DOMAINS
# ---------------------------------------------------
topics = [
    "Network Security",
    "Identity & Access Management",
    "Cryptography",
    "Security Operations",
    "Security Awareness",
    "Critical Infrastructure",
    "Threat Intelligence",
    "Risk Analysis",
    "Physical Security",
    "Supply Chain Security"
]

# ---------------------------------------------------
# TOPIC INPUTS
# ---------------------------------------------------
st.header("📊 Security Domain Inputs")

r = {}
u = {}
v = {}

for topic in topics:

    st.subheader(topic)

    v[topic] = st.slider(
        f"{topic} Vulnerability",
        0.0,
        1.0,
        0.5,
        key=f"v_{topic}"
    )

    cols = st.columns(len(objectives))

    for i, objective in enumerate(objectives):

        with cols[i]:

            r[(topic, objective)] = st.slider(
                f"{objective} Relevance",
                0.0,
                1.0,
                0.5,
                key=f"r_{topic}_{objective}"
            )

            u[(topic, objective)] = st.slider(
                f"{objective} Utility",
                0.0,
                1.0,
                0.5,
                key=f"u_{topic}_{objective}"
            )

# ---------------------------------------------------
# INTERDEPENDENCY MATRIX
# ---------------------------------------------------
st.header("🔗 Interdependency Matrix (D)")

st.markdown(
    """
    Positive values = supporting influence  
    Negative values = inhibiting influence
    """
)

default_matrix = pd.DataFrame(
    np.zeros((len(topics), len(topics))),
    index=topics,
    columns=topics
)

D_df = st.data_editor(
    default_matrix,
    use_container_width=True,
    num_rows="fixed"
)

D = {}

for source in topics:
    for target in topics:

        value = float(D_df.loc[source, target])

        if value != 0:
            D[(source, target)] = value

# ---------------------------------------------------
# COMPUTATION
# ---------------------------------------------------
weights = compute_risk_weights(risk_params)

eta = compute_base_eta(
    topics,
    objectives,
    r,
    u,
    v,
    beta
)

eta_prime = compute_adjusted_eta(
    topics,
    objectives,
    eta,
    D,
    alpha
)

scores = compute_scores(
    topics,
    objectives,
    eta_prime,
    weights
)

ranking = sorted(
    scores.items(),
    key=lambda x: x[1],
    reverse=True
)

# ---------------------------------------------------
# RESULTS
# ---------------------------------------------------
st.header("🏆 Ranked Security Domains")

ranking_df = pd.DataFrame(
    ranking,
    columns=["Security Domain", "Score"]
)

ranking_df["Classification"] = ranking_df["Score"].apply(classify)

st.dataframe(
    ranking_df,
    use_container_width=True
)

# ---------------------------------------------------
# SCORE CHART
# ---------------------------------------------------
st.header("📈 Domain Scores")

chart_df = ranking_df.set_index(
    "Security Domain"
)[["Score"]]

st.bar_chart(chart_df)

# ---------------------------------------------------
# RISK WEIGHTS
# ---------------------------------------------------
st.header("⚖️ Objective Risk Weights")

weights_df = pd.DataFrame(
    {
        "Objective": list(weights.keys()),
        "Weight": list(weights.values())
    }
)

st.dataframe(
    weights_df,
    use_container_width=True
)

# ---------------------------------------------------
# NETWORK GRAPH
# ---------------------------------------------------
st.header("🌐 Security Dependency Graph")

G = nx.DiGraph()

for topic in topics:
    G.add_node(
        topic,
        score=scores[topic]
    )

for (src, dst), weight in D.items():
    G.add_edge(
        src,
        dst,
        weight=weight
    )

fig, ax = plt.subplots(
    figsize=(12, 9)
)

pos = nx.spring_layout(
    G,
    seed=42,
    k=2
)

node_sizes = [
    max(scores[node] * 5000, 500)
    for node in G.nodes()
]

edge_colors = [
    "green" if data["weight"] > 0 else "red"
    for _, _, data in G.edges(data=True)
]

edge_widths = [
    abs(data["weight"]) * 4
    for _, _, data in G.edges(data=True)
]

nx.draw_networkx_nodes(
    G,
    pos,
    node_size=node_sizes,
    node_color="skyblue",
    alpha=0.9,
    ax=ax
)

nx.draw_networkx_labels(
    G,
    pos,
    font_size=8,
    ax=ax
)

if len(G.edges()) > 0:
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_colors,
        width=edge_widths,
        arrows=True,
        arrowsize=20,
        ax=ax
    )

ax.set_title(
    "Security Domain Dependency Network",
    fontsize=14
)

ax.axis("off")

st.pyplot(fig)

# ---------------------------------------------------
# CLASSIFICATION SUMMARY
# ---------------------------------------------------
st.header("📌 Classification Summary")

critical = ranking_df[
    ranking_df["Classification"] == "🔴 Critical"
]

high = ranking_df[
    ranking_df["Classification"] == "🟠 High"
]

medium = ranking_df[
    ranking_df["Classification"] == "🟡 Medium"
]

low = ranking_df[
    ranking_df["Classification"] == "🟢 Low"
]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Critical", len(critical))

with c2:
    st.metric("High", len(high))

with c3:
    st.metric("Medium", len(medium))

with c4:
    st.metric("Low", len(low))

st.success("ARES Framework evaluation completed.")






# ===================================================

# ===================================================
# MCDA COMPARISON: ARES vs TOPSIS vs AHP
# ===================================================

from scipy.stats import spearmanr
import numpy as np

st.header("📊 MCDA Comparison Dashboard")

# ---------------------------------------------------
# BENCHMARK PERFORMANCE MATRIX
# ---------------------------------------------------

# Traditional MCDA methods only use the relevance
# assessments provided by experts.
#
# ARES remains the only method using:
# - Vulnerability
# - Utility
# - Interdependency propagation
# - Dynamic risk weighting

performance = {
    (t, o): r[(t, o)]
    for t in topics
    for o in objectives
}

# ---------------------------------------------------
# TOPSIS
# ---------------------------------------------------

def compute_topsis(
    topics,
    objectives,
    performance,
    weights
):
    X = np.array([
        [
            performance[(t, o)]
            for o in objectives
        ]
        for t in topics
    ])

    # Vector normalization
    norm = np.sqrt(
        (X ** 2).sum(axis=0)
    )

    norm[norm == 0] = 1

    X_norm = X / norm

    # Weighted matrix
    w = np.array([
        weights[o]
        for o in objectives
    ])

    X_weighted = X_norm * w

    # Ideal solutions
    ideal_best = X_weighted.max(axis=0)
    ideal_worst = X_weighted.min(axis=0)

    # Distances
    d_best = np.sqrt(
        ((X_weighted - ideal_best) ** 2).sum(axis=1)
    )

    d_worst = np.sqrt(
        ((X_weighted - ideal_worst) ** 2).sum(axis=1)
    )

    scores = d_worst / (d_best + d_worst + 1e-9)

    return {
        topics[i]: float(scores[i])
        for i in range(len(topics))
    }

# ---------------------------------------------------
# AHP
# ---------------------------------------------------

def compute_ahp_weights(pairwise_matrix):
    matrix = np.array(
        pairwise_matrix,
        dtype=float
    )

    eigenvalues, eigenvectors = np.linalg.eig(
        matrix
    )

    max_index = np.argmax(
        eigenvalues.real
    )

    weights = eigenvectors[
        :,
        max_index
    ].real

    weights = weights / weights.sum()

    return weights


def compute_ahp_scores(
    topics,
    objectives,
    performance,
    criteria_weights
):
    scores = {}

    for t in topics:
        score = 0

        for i, o in enumerate(objectives):
            score += (
                criteria_weights[i]
                * performance[(t, o)]
            )

        scores[t] = float(score)

    return scores

# ---------------------------------------------------
# AHP PAIRWISE MATRIX
# ---------------------------------------------------

# Example expert judgment matrix
# Replace later with real expert evaluations

pairwise_matrix = [
    [1,   3,   5,   7],
    [1/3, 1,   3,   5],
    [1/5, 1/3, 1,   3],
    [1/7, 1/5, 1/3, 1]
]

ahp_weights = compute_ahp_weights(
    pairwise_matrix
)

# ---------------------------------------------------
# COMPUTE METHODS
# ---------------------------------------------------

ares_scores = scores

topsis_scores = compute_topsis(
    topics,
    objectives,
    performance,
    weights
)

ahp_scores = compute_ahp_scores(
    topics,
    objectives,
    performance,
    ahp_weights
)

# ---------------------------------------------------
# RANKING FUNCTION
# ---------------------------------------------------

def rank(scores_dict):
    return sorted(
        scores_dict.items(),
        key=lambda x: x[1],
        reverse=True
    )

ares_rank = rank(ares_scores)
topsis_rank = rank(topsis_scores)
ahp_rank = rank(ahp_scores)

# ---------------------------------------------------
# DISPLAY
# ---------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "ARES",
    "TOPSIS",
    "AHP",
    "Benchmark"
])

with tab1:
    st.subheader("ARES Ranking")

    st.dataframe(
        pd.DataFrame(
            ares_rank,
            columns=["Domain", "Score"]
        ),
        use_container_width=True
    )

with tab2:
    st.subheader("TOPSIS Ranking")

    st.dataframe(
        pd.DataFrame(
            topsis_rank,
            columns=["Domain", "Score"]
        ),
        use_container_width=True
    )

with tab3:
    st.subheader("AHP Ranking")

    st.dataframe(
        pd.DataFrame(
            ahp_rank,
            columns=["Domain", "Score"]
        ),
        use_container_width=True
    )

with tab4:
    st.subheader("Method Comparison")

    comparison_df = pd.DataFrame({
        "ARES": dict(ares_scores),
        "TOPSIS": dict(topsis_scores),
        "AHP": dict(ahp_scores)
    })

    comparison_df = comparison_df.sort_values(
        by="ARES",
        ascending=False
    )

    st.dataframe(
        comparison_df,
        use_container_width=True
    )

    st.bar_chart(comparison_df)

    # Ranking correlation
    def corr(s1, s2):
        return spearmanr(
            [s1[t] for t in topics],
            [s2[t] for t in topics]
        )[0]

    st.subheader("Ranking Correlations")

    st.write(
        f"ARES vs TOPSIS: {corr(ares_scores, topsis_scores):.3f}"
    )

    st.write(
        f"ARES vs AHP: {corr(ares_scores, ahp_scores):.3f}"
    )

    st.write(
        f"TOPSIS vs AHP: {corr(topsis_scores, ahp_scores):.3f}"
    )

