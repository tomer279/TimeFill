"""Continuous-time diffusion imputation package (M1 toy path).

Packages
--------
data
    Schema, config loading, synthetic generators, and masking.
sde
    VP-SDE forward process and forward-noising plots.
models
    Conditional Equinox score network (temporal TCN).
train
    Conditional DSM loss, Optax train loop, checkpoints (WP4).
sample
    Diffrax reverse-SDE imputation API (WP5).
eval
    Masked MAE/RMSE/CRPS/coverage and M1 eval plots (WP6).
plot_io
    Atomic Matplotlib figure save helper (Windows-safe overwrite).
"""
