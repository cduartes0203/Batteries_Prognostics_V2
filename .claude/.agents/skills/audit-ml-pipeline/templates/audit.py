# %% [markdown]
# # Audit — <NN>_<short_name>: <what this experiment tested>
#
# Read-only review of the stored report: checks and metrics for this
# experiment.

# %%
import skore

from <pkg> import PROJECT_ROOT

# %% [markdown]
# ## Open the project

# %%
project = skore.Project(
    name="<project-name>",
    mode="local",
    workspace=str(PROJECT_ROOT / "reports"),
)
project

# %% [markdown]
# ## List the available reports

# %%
summary = project.summarize()
summary

# %% [markdown]
# ## Load the report

# %%
REPORT_ID = "skore:report:<type-singular>:<N>"

report = project.get(REPORT_ID)
report

# %% [markdown]
# ## Persisted report
#
# <REPORT_LOCATOR>

# %% [markdown]
# ## Checks summary
#
# Automated checks for this report: issues and tips that affect how
# to read the metrics below.

# %%
checks = report.checks.summarize()
checks

# %% [markdown]
# ## Metrics summary
#
# Headline scores for this experiment (task-appropriate defaults;
# cross-validation reports include mean ± std across folds).

# %%
metrics = report.metrics.summarize().frame(verbose_name=True, flat_index=False)
metrics

# %% [markdown]
# ## Core audit complete
#
# Checks and metrics above are the first pass. Further views stay
# read-only on this same report.
