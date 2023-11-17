import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

import pymc as pm
import pytensor.tensor as pt


def logisitic_curve(x, L, k, x0, exp=np.exp):
    return L / (1 + exp(-k * (x - x0)))


def plot_logistic_curve(k, x0, *, ax: plt.Axes):
    x = np.linspace(0, 100, 1000)
    y = logisitic_curve(x, 100, k, x0)
    ax.plot(x, y, label=f"k={k}, x0={x0}")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)


if __name__ == "__main__":
    df = pd.DataFrame.from_records(
        [
            ("running", 5, 60 * 50),
            ("running", 5, 60 * 45),
            ("running", 45, 60 * 15),
            ("running", 50, 60 * 10),
            ("running", 75, 60 * 10 + 20),
            ("running", 75, 60 * 10 - 5),
            ("running", 100, 60 * 7),
            ("running", 100, 60 * 6 + 49),
        ],
        columns=["exercise", "effort", "time"],
    )

    xx = np.linspace(0, 100, 50)
    coords = {"effort": xx, "exercises": ["running"]}
    with pm.Model(coords=coords) as model:
        # Priors
        k = pm.HalfNormal("k", sigma=0.01)
        x0 = pm.Beta("x0", 3, 3) * 100
        L = pm.HalfNormal("L", sigma=10)

        sigma = pm.HalfNormal("sigma", sigma=1)

        # Likelihood
        mu = logisitic_curve(df["effort"].to_numpy(), L, -k, x0)
        nu = pm.HalfNormal("nu", sigma=3)
        y = pm.StudentT(
            "y", nu=nu, mu=mu, sigma=sigma, observed=np.log(df["time"].to_numpy())
        )

        mu_curve = logisitic_curve(xx, L, -k, x0)
        pm.Deterministic(
            "curve", pt.exp(logisitic_curve(xx, L, -k, x0)), dims=("effort",)
        )

        # Sample
        idata = pm.sample(1000, tune=1000, chains=4)
        ppc = pm.sample_posterior_predictive(idata, var_names=["y", "curve"])

    df_conf = (
        ppc.posterior_predictive.curve.to_dataframe()
        .groupby(level=-1)
        .describe()
        .droplevel(0, axis=1)
        .assign(
            lower=lambda row: row["mean"] - 2 * row["std"],
            upper=lambda row: row["mean"] + 2 * row["std"],
        )
        .loc[:, ["lower", "mean", "upper"]]
    )
    ax = plt.gca()
    ax.fill_between(
        df_conf.index,
        df_conf["lower"],
        df_conf["upper"],
        color="C0",
        alpha=0.2,
        label="95% confidence",
    )
    ax.plot(df_conf.index, df_conf["mean"], color="black", linestyle="--", label="mean")
    ax.scatter(df["effort"], df["time"], color="C1", label="data", alpha=1)
    # ax.set_yscale("log")
    ax.legend()
    ylabels = ax.get_yticks()
    ax.set(ylim=(0, None), ylabel="time (minutes)", xlabel="effort (percent)")
    ax.set_yticks(ylabels, [round(y / 60, 1) for y in ylabels])
    plt.show()
    # fig, ax = plt.subplots()
    # plot_logistic_curve(-1, 50, ax=ax)
    # plot_logistic_curve(- 1 / 10, 50, ax=ax)
    # plot_logistic_curve(- 1 / 100, 50, ax=ax)
    # ax.legend()
    # plt.show()
