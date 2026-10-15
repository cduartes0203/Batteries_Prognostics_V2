# RFLO (Murray, 2019) vs. `Testing/DRTLO_GD_R2.py`

**Sources compared**

- **Paper:** J. M. Murray, *Local online learning in recurrent networks with random feedback*, eLife 8:e43299 (2019), [rflo.pdf](rflo.pdf). Main text eqs. (1)–(7), Appendix 1 eqs. (10)–(17).
- **Authors' code:** [RFLO.ipynb](RFLO.ipynb), class `RNN.run_trial`.
- **Your code:** [Testing/DRTLO_GD_R2.py](../Testing/DRTLO_GD_R2.py), class `RTLO`.

This is the class R07 uses: [eGDRNN_GD_R07.ipynb](../eGDRNN_GD_R07.ipynb) imports `Functions/AutoCloud.py`, which imports `Functions/DataCloud.py`, which imports `Testing/DRTLO_GD_R2.py`. Every granule creates one `RTLO` object (`DataCloud.__init__`).

---

## 1. RFLO as defined in the paper

### 1.1 Model (single recurrent layer, no bias)

$$
u(t) = W\,h(t-1) + W^{in}\,x(t), \qquad
h(t) = h(t-1) + \frac{1}{\tau}\Big[-h(t-1) + \phi\big(u(t)\big)\Big], \qquad
y(t) = W^{out} h(t)
$$

Loss: $\varepsilon(t) = y^*(t) - y(t)$, $\;L = \frac{1}{2T}\sum_t \lVert \varepsilon(t) \rVert^2$, with $\phi = \tanh$.

### 1.2 Exact gradient (RTRL), Appendix 1 eqs. (10)–(11)

$$
\Delta W_{ab} = \frac{\eta_2}{T}\sum_t \sum_j \big[(W^{out})^{\top}\varepsilon(t)\big]_j\, P^j_{ab}(t)
$$

$$
P^j_{ab}(t) = \Big(1-\tfrac{1}{\tau}\Big)P^j_{ab}(t-1)
\;+\; \underbrace{\tfrac{1}{\tau}\sum_k \phi'\big(u_j(t)\big) W_{jk} P^k_{ab}(t-1)}_{\text{non-local term}}
\;+\; \tfrac{1}{\tau}\,\delta_{ja}\,\phi'\big(u_a(t)\big)\,h_b(t-1)
$$

### 1.3 The two RFLO approximations

1. **Locality:** drop the non-local term. The rank-3 tensor $P^j_{ab}$ then collapses to a rank-2 *eligibility trace* $p_{ab}$.
2. **Random feedback:** replace $(W^{out})^{\top}$ with a fixed random matrix $B$ (feedback alignment, Lillicrap et al. 2016).

The resulting rule is the paper's eqs. (5)–(6):

$$
\delta W^{out}_{ab} = \eta_1\,\varepsilon_a(t)\,h_b(t), \qquad
\delta W_{ab} = \eta_2\,[B\varepsilon(t)]_a\,p_{ab}(t), \qquad
\delta W^{in}_{ab} = \eta_3\,[B\varepsilon(t)]_a\,q_{ab}(t)
$$

$$
p_{ab}(t) = \Big(1-\tfrac{1}{\tau}\Big)p_{ab}(t-1) + \tfrac{1}{\tau}\,\phi'\big(u_a(t)\big)\,h_b(t-1), \qquad
q_{ab}(t) = \Big(1-\tfrac{1}{\tau}\Big)q_{ab}(t-1) + \tfrac{1}{\tau}\,\phi'\big(u_a(t)\big)\,x_b(t)
$$

Cost: $\mathcal{O}(N^2)$ per time step, against $\mathcal{O}(N^4)$ for RTRL.

> **Index inconsistency in the paper.** The equations above follow from the model in §1.1 (eq. 13 of the paper: $u(t) = W h(t-1) + W^{in}x(t)$). The paper itself writes the trace three different ways:
>
> - eq. (6): $\phi'(u_a(t))\,x_b(t-1)$
> - eq. (17): $\phi'(u_a(t-1))\,h_b(t-1)$ and $\phi'(u_a(t-1))\,x_b(t-1)$
> - eq. (11): $\phi'(u_a(t))\,x_b(t-1)$
>
> Differentiating eq. (13) gives $\partial u_a(t)/\partial W^{in}_{ab} = x_b(t)$ and $\partial u_a(t)/\partial W_{ab} = h_b(t-1)$. So the forms above are the correct ones, and the paper's variants are notation slips. **Cite the derived form in your manuscript, not eq. (6) or eq. (17) verbatim.**

### 1.4 The paper's own ablation (Fig. 2d)

Murray tests the two approximations separately:

- **Local only** (locality approximation, symmetric feedback $W^{out\top}$): performs like full RFLO.
- **RF only** (exact non-local gradient, random feedback $B$): performs like BPTT.

So almost all of the gap between RFLO and BPTT comes from **dropping the non-local term**, not from the random feedback. This matters for §5 below.

---

## 2. Paper vs. the authors' notebook (`RFLO.ipynb`)

| Item | Paper (eqs. 1, 5, 6; Methods) | `RFLO.ipynb` | Verdict |
|---|---|---|---|
| Dynamics | $h(t)=h(t-1)+\frac{1}{\tau}[-h(t-1)+\phi(u(t))]$ | `h[tt+1] = h[tt] + (-h[tt] + f(u[tt+1]))/tau_m` | Same |
| Total input | $u(t)=Wh(t-1)+W^{in}x(t)$ | `u[tt+1] = w_rec@h[tt] + w_in@x[tt+1]` | Same |
| Output / error | $y=W^{out}h,\;\varepsilon=y^*-y$ | `y[tt+1] = w_out@h[tt+1]`; `err = y_ - y` | Same |
| $\phi'$ | $1-\tanh^2$ | `1/cosh(10*tanh(x/10))**2` (overflow-safe sech²) | Equivalent |
| Recurrent trace $p$ | $\phi'(u_a(t))\,h_b(t-1)$ | `outer(df(u[tt+1]), h[tt])` | **Correct** |
| Input trace $q$ | $\phi'(u_a(t))\,x_b(t)$ (derived) | `outer(df(u[tt+1]), x[tt])`, i.e. $x(t-1)$ | **One-step lag** (copies eq. 6). The RTRL branch uses `x[tt+1]`, which is correct. No effect in the demo: $x\equiv0$ and $\eta_{in}=0$. |
| $W^{out}$ update | $\eta_1\varepsilon h^{\top}$ | `eta1/t_max*outer(err, h)` | Same, scaled by $1/T$ |
| $W$, $W^{in}$ updates | $\eta[B\varepsilon]_a p_{ab}$ | `eta2*outer(b@err, ones)*p/t_max` | Same, scaled by $1/T$ |
| Update timing | online, or batched per trial | `online_learning` flag (default: per trial) | Same |
| Init $W$ | $\mathcal{N}(0, g^2/N)$, $g=1.5$ | `1.5*randn/sqrt(N)` | Same |
| Init $W^{in}$ | $\mathcal{U}[-1,1]$ | `0.1*(rand-1)`, i.e. $\mathcal{U}[-0.1,0]$ | Differs (sign-biased, small) |
| Init $W^{out}$ | $\mathcal{U}[-1/\sqrt N, 1/\sqrt N]$ | `0.1*(2*rand-1)/sqrt(N)` | 10× smaller |
| Feedback $B$ | $\mathcal{N}(0,1)$ | `randn/sqrt(n_out)` | Same for $n_{out}=1$ |
| Learning-rate order | $\eta_1$ = out, $\eta_2$ = rec, $\eta_3$ = in | list is `[eta_in, eta_rec, eta_out]` | Convention only |
| Bias | none | none | Same |

**Conclusion:** the notebook implements RFLO faithfully. The only formula deviation is the one-step lag in the input trace $q$, inherited from the paper's eq. (6), and it is inactive in the demo.

---

## 3. What `DRTLO_GD_R2.py` implements

A **stack of $L$ leaky recurrent layers**, trained online, one sample per `fit()` call. Notation: layer $l=1..L$; $h^0(t) \equiv x(t)$; $\tilde v = [v;\,1]$ appends the bias input.

$$
u^l(t) = W^l\,h^l(t-1) + U^l\,\tilde h^{\,l-1}(t), \qquad
h^l(t) = \Big(1-\tfrac{1}{\tau}\Big)h^l(t-1) + \tfrac{1}{\tau}\,\phi\big(u^l(t)\big), \qquad
y(t) = W^{out} h^L(t)
$$

In the code: $W^l$ = `wR[l]`, $U^l$ = `wI[l]`, $W^{out}$ = `wO`, $B^l$ = `BS[l]`; $\eta_1,\eta_2,\eta_3$ = `ηS` (out, rec, in), the same order as the paper.

### 3.1 Side-by-side

| Step | Murray RFLO (1 layer) | `DRTLO_GD_R2.fit` (layer $l$) | Lines | Verdict |
|---|---|---|---|---|
| Total input | $u=Wh(t-1)+W^{in}x(t)$ | $u^l=W^l h^l(t-1)+U^l\tilde h^{l-1}(t)$ | 102 | Same (adds bias, stacks layers) |
| State | $h(t)=(1-\frac1\tau)h(t-1)+\frac1\tau\phi(u(t))$ | identical | 103 | Same |
| Output | $y=W^{out}h(t)$ | $y=W^{out}h^L(t)$ | 109 | Same |
| Error | $\varepsilon=y^*-y$ | `eS = yR - yP` | 110 | Same |
| Recurrent trace | $\frac1\tau\phi'(u_a(\mathbf{t}))h_b(t-1)+(1-\frac1\tau)p$ | $\frac1\tau\phi'(u^l_a(\mathbf{t-1}))h^l_b(t-1)+(1-\frac1\tau)p$ | 115 | **Wrong time index** (E1) |
| Input trace | $\frac1\tau\phi'(u_a(\mathbf{t}))x_b(t)+(1-\frac1\tau)q$ | $\frac1\tau\phi'(u^l_a(\mathbf{t-1}))\tilde h^{l-1}_b(t)+(1-\frac1\tau)q$ | 116 | **Wrong time index** (E1) |
| Error routed to the hidden layer | $B\varepsilon$, with $B\in\mathbb{R}^{N\times n_{out}}$ | $B^l\varepsilon$, a separate random $B^l$ for **every** layer | 60, 118–119 | Still random feedback (E2). Deep version = Direct Feedback Alignment (E3) |
| $W$ update | $\eta_2[B\varepsilon]_a p_{ab}$ | `η2*outer(BS[l]@eS, ones)*pS[l]` | 118, 122 | Same form |
| $W^{in}$ update | $\eta_3[B\varepsilon]_a q_{ab}$ | `η3*outer(BS[l]@eS, ones)*qS[l]` | 119, 121 | Same form |
| $W^{out}$ update | $\eta_1\varepsilon h^{\top}$ (exact gradient) | `η1*outer(eS, h_L)` | 130–131 | Same |
| Update timing | per step or per trial (÷T) | every step, no ÷T | — | Fine for a data stream |
| Sign | gradient descent | `+=` with $\varepsilon=y^*-y$ | — | Correct |
| Init $W$ | $\mathcal N(0,1.5^2/N)$ | Xavier-uniform, variance $\approx 1/N$ (i.e. $g\approx1$) | 54 | Differs (E6) |
| Init $B$ | $\mathcal N(0,1)$ | Xavier-uniform, small | 60 | Differs (E6) |
| Seeds | random per network | `np.random.seed(42)` plus fixed `sd` per matrix | 6, 48–60 | Every granule is identical at birth (E4) |

**Everything except the trace timing (E1) matches RFLO's update form exactly.** The deep extension (feedback sent to each layer) goes beyond what the paper covers; see E3.

---

## 4. Findings

### E1 — Eligibility traces use $\phi'(u(t-1))$ instead of $\phi'(u(t))$ · **bug** · lines 115–116

`self.uP[l]` still holds the previous step's input when the traces are updated. `new_uP[l]`, the current one, has already been computed at line 104 but is not used.

- In $p$, both factors are lagged. This is internally consistent with the paper's eq. (17), but it does not match the derivative of the model.
- In $q$, $\phi'$ is lagged while the input is current, so **the two factors refer to different time steps**. This matches none of the paper's variants.
- On the very first call, `uP = 0`, so $\phi'(0)=1$ is used regardless of the actual input.

**Numerical check.** With the real `RTLO` class (1 layer, 6 units, τ = 10), I compared the term the code adds to $p$ against a central finite difference of $h(t)$ with respect to $W_{ab}$, holding $h(t-1)$ fixed:

| | value | relative error |
|---|---|---|
| finite difference (true local derivative) | −8.867e−03 | — |
| code, $\phi'(u(t-1))$ | −4.063e−03 | **54 %** |
| corrected, $\phi'(u(t))$ | −8.867e−03 | 0.00 % |

**Fix:** use `dActivation(new_uP[l], self.act)` in lines 115–116.

### E2 — The random feedback was *not* removed · **contradicts the intended design** · lines 60, 118–119

`self.BS[l]` is a fixed random matrix (never updated), and it multiplies the error in both hidden-layer updates. So the file is still **random-feedback** learning, despite the "GD" in its name.

The version that sends the error back through the real weights lives in `DRTLO_QN_R1.py` / `DRTLO_QN_R2.py` (`wO.T @ eS`, then `wI.T @ du`). R07 does not use those files.

### E3 — The deep extension is Direct Feedback Alignment, which Murray's paper does not cover · **theory gap, not a bug**

Each hidden layer receives $B^l\varepsilon$ directly from the output. That is *Direct Feedback Alignment* (Nøkland, 2016) combined with RFLO's eligibility traces.

Murray's convergence argument (Appendix 2: the loss decreases when $W^{out}$ aligns with $B$) is derived for **one** linear layer only. For $L>1$:

- the lower layers' feedback $B^l$ has to align with the *effective* path from layer $l$ to the output, and nothing in the paper guarantees that;
- the manuscript has to state this explicitly and cite Nøkland (2016), or switch to option (a) in §5.

This links directly to Reviewer 1, point 3 (no convergence or stability analysis) and Reviewer 4, point 1 (novelty).

### E4 — Fixed seeds and a global `np.random.seed(42)` · **design issue** · lines 6, 48–60

- Every `RTLO` starts with **identical** weights and identical feedback matrices, because the seeds 39–42 are hard-coded. A new granule therefore starts as an untrained clone of the very first granule's initial state. That weakens the "granule-specific model" argument and matches the identical weights seen in R07 cells 9–10.
- Calling `np.random.seed(42)` inside the constructor resets NumPy's **global** generator each time a granule is created. That silently affects any other code that draws from `np.random`.

Suggested fix: pass a `numpy.random.Generator` (or a per-granule seed) into the constructor and never reseed the global generator.

### E5 — The full weight history is stored at every step · **memory and complexity** · lines 124–125, 132

`wR_hist`, `wI_hist` and `wO_hist` append flattened copies of every matrix on every `fit`. Memory grows as $\mathcal O(\text{steps}\times\text{parameters}\times\text{granules})$. That inflates run time during the Optuna search, and it would distort any computational-cost figure reported in answer to Reviewer 1, point 1. Make it optional (e.g. `store_hist=False`).

### E6 — Initialisation differs from the paper · **document it** · lines 48–60

- **Recurrent weights:** Xavier-uniform gives variance ≈ $1/N$ ($g≈1$, edge of chaos), against Murray's $g=1.5$ (chaotic regime, richer dynamics).
- **Feedback:** Xavier-uniform $B$ is small, against Murray's $\mathcal N(0,1)$. A smaller $B$ scales down the hidden-layer learning, so it is effectively coupled to $\eta_2,\eta_3$.

Neither is wrong, but the paper should state the choice. Both affect how fast feedback alignment happens.

### E7 — Dead state · **cleanup** · various lines

`j`, `start`, `ref`, `ρ`, `rR`/`rP`/`rL`/`rU`, `rRsum`, `yP_hist`, `uP2`, `εY…`, `rulR…` are never used. `fit()` also sets `self.n` (line 92), which only matters for prediction. Removing these would make the class easier to describe in the paper, which helps with Reviewer 1, point 7 (reproducibility).

### What is coherent

- Within a time step, the forward pass in `fit` is identical to the one in `PredictSingle`/`PredictIntr`. AutoCloud calls `Restore()` before `Adapt()`, so the state used for the prediction and the state used for learning are the same. **No look-ahead.**
- Trace shapes, including the bias column, are right for every layer: `qS[0]` is $N_1\times(n_I+1)$ and `qS[l]` is $N_l\times(N_{l-1}+1)$.
- The trace at time $t$ is updated before it is multiplied by $\varepsilon(t)$, as in eq. (5).
- Per step, the cost is $\mathcal O\big(\sum_l N_l^2 + N_l N_{l-1}\big)$, the same order as RFLO and BPTT.

### Integration notes (in AutoCloud, but they affect the RNN)

- `AutoCloud.Adapt` calls `fit` only for granules with $\alpha_i>0$. Inactive granules keep a **frozen** hidden state $h(t)$ and frozen traces $p, q$. When such a granule is reactivated, its state belongs to some earlier time. Consider advancing $h$ for every granule (a forward pass is cheap) and training only the active ones.
- `AutoCloud.MergeCloudsRNN` is unreachable while `wtaG=True`, and broken if it ever runs (as reported earlier).

---

## 5. What option (a) — "remove the random feedback" — would look like

Goal: keep RFLO's local traces over time, but replace each random $B^l$ with the **true error signal within the current time step**.

Define the learning signal $\boldsymbol{\ell}^l(t) \approx \partial L(t)/\partial h^l(t)$, keeping only paths that stay inside time step $t$:

$$
\boldsymbol{\ell}^L(t) = (W^{out})^{\top}\varepsilon(t)
$$

$$
\boldsymbol{\ell}^{\,l}(t) = \frac{1}{\tau}\,\big(\bar U^{\,l+1}\big)^{\top}\Big[\phi'\big(u^{l+1}(t)\big)\odot \boldsymbol{\ell}^{\,l+1}(t)\Big], \qquad l = L-1,\dots,1
$$

Here $\bar U^{\,l+1}$ is `wI[l+1][:, :-1]`: the bias column is excluded because the bias does not depend on $h^l$. The updates keep exactly the same form; only $B^l\varepsilon$ is replaced by $\boldsymbol\ell^l$:

$$
\delta W^l_{ab} = \eta_2\,\ell^l_a(t)\,p^l_{ab}(t), \qquad
\delta U^l_{ab} = \eta_3\,\ell^l_a(t)\,q^l_{ab}(t), \qquad
\delta W^{out}_{ab} = \eta_1\,\varepsilon_a(t)\,h^L_b(t)
$$

with $p^l, q^l$ computed with $\phi'(u^l(t))$ (fix E1 first).

| | Current (`DRTLO_GD_R2`) | Option (a) |
|---|---|---|
| Error to layer $l$ | $B^l\varepsilon$ (fixed random) | $\boldsymbol\ell^l$ (true, within one step) |
| Feedback matrices to tune / seed | $L$ | none |
| Theory | Murray App. 2 (1 layer) + DFA (heuristic) | Murray's *"Local only"* variant (Fig. 2d), extended across layers; same family as *e-prop* (Bellec et al., 2020) |
| Remaining approximation | locality **and** random feedback | locality only |
| Cost per step | $\mathcal O(\sum N_l^2)$ | $\mathcal O(\sum N_l^2)$ (one extra matrix–vector product per layer) |
| Online / causal | yes | yes |

**Expected effect:** according to Murray's Fig. 2d, removing the random feedback alone gives a **small** accuracy gain, because locality causes most of the gap to BPTT. The main benefits of (a) are therefore:

- **Fewer arbitrary components:** no random $B^l$ and no seeds to justify.
- **A cleaner theory statement:** an exact gradient with respect to the current state, truncated in time. This is easier to defend against Reviewer 1, point 3 and Reviewer 4, point 1.
- **A ready-made ablation** (RFLO-DFA vs. RFLO-symmetric), which answers Reviewer 2, point 7 and Reviewer 4, point 2.

---

## 6. Summary

| # | Finding | Type | Where |
|---|---|---|---|
| E1 | Traces use $\phi'(u(t-1))$; must be $\phi'(u(t))$ (54 % error in the check) | **Bug** | `DRTLO_GD_R2.py:115-116` |
| E2 | Random feedback still present; the gradient version is only in the QN files | Intent mismatch | `:60`, `:118-119` |
| E3 | Deep extension = DFA, outside Murray's analysis | Theory gap | design |
| E4 | Identical init for every granule; global reseed | Design | `:6`, `:48-60` |
| E5 | Weight history stored every step | Cost | `:124-125`, `:132` |
| E6 | Init differs from the paper ($g$, $B$ scale) | Document | `:48-60` |
| E7 | Unused attributes | Cleanup | various |
| — | Paper eqs. (6)/(11)/(17) disagree on time indices; use the derived form | Writing | rflo.pdf |
| — | Authors' notebook lags $x$ in $q$ (inactive in their demo) | Reference code | RFLO.ipynb |

No code was changed.

---

## 7. `Testing/eRNN_GD_R1.py` (used by `eGDRNN_GD_OPT_R6.ipynb`)

**Import chain:** [eGDRNN_GD_OPT_R6.ipynb](../eGDRNN_GD_OPT_R6.ipynb) imports `Functions/AutoCloud_V3.py`, which imports `Functions/DataCloud_V3.py`, which imports [Testing/eRNN_GD_R1.py](../Testing/eRNN_GD_R1.py) (class `eRNN`). The tuning in that notebook goes through `Functions/TedaOptimize_GD_R1.py`.

### 7.1 Learning rule: identical to `DRTLO_GD_R2`

A `diff` of the two `fit()` methods shows that the only change is renaming `n_layers` to `nL`. The two classes therefore share the **same learning rule**, so every finding in §4 applies unchanged:

| Finding | `DRTLO_GD_R2` | `eRNN_GD_R1` |
|---|---|---|
| E1 trace timing $\phi'(u(t-1))$ | lines 115–116 | **same**, lines 139–140 |
| E2 random feedback $B^l$ | lines 60, 118–119 | **same**, lines 84, 142–143 |
| E3 deep version = DFA (outside Murray's theory) | yes | **same** |
| E5 weight history stored every step | lines 124–125, 132 | **same**, lines 148–149, 156 |
| E6 initialisation differs from paper | Xavier-uniform | **same** |
| E7 dead state | yes | **same** (`j`, `start`, `ref`, `rR…`, `yP_hist`, `uP2`, …) |
| E4 seeds | global `np.random.seed(42)` in the constructor + fixed seeds | **partly improved:** a seed parameter `sd`, and no reseed in the constructor. But `XavierUniform` still reseeds the **global** generator (line 4), and `AutoCloud_V3` hard-codes `self.sd = 42` (line 20), so every granule is still identical at birth. |

### 7.2 What is new in `eRNN_GD_R1` and its V3 integration

| # | Change | Verdict |
|---|---|---|
| F1 | **New granules inherit their weights by reference.** `AutoCloud_V3.DefRnnParams` (called at line 660 for every new granule) passes `ReturnParameters()` straight into `ReceiveParameters()`, with no copy. | **Serious bug.** See below. |
| F2 | Merging is reachable (R6 uses `wtaG=False`). `MergeCloudsRNN` (V3) averages every item layer by layer. | Works, unlike the V1 version, which crashes. But it is a **plain mean**, not the typicality-weighted mean of the manuscript (eq. 37). It also averages $h$, $p$, $q$ while `uP` is not passed on. |
| F3 | Default `ηS=None` gives `0.1*np.ones(nL)`, i.e. one value **per layer**, while `fit` unpacks exactly 3 values. | **Crash** whenever `nL ≠ 3` and no `ηS` is given (`ValueError: not enough values to unpack`). Latent, because AutoCloud always passes 3 values. |
| F4 | The constructor converts $N\to10^{-N}$ **in place** on the caller's object. | A list passed in gets overwritten (`[1,4,3]` becomes `[0.1, 1e-4, 1e-3]`). An **integer NumPy array gets truncated to `[0 0 0]`**, which silently disables learning. Latent in R6, because AutoCloud_V3 converts first and passes floats. The conversion is also duplicated between AutoCloud and eRNN. |
| F5 | `XavierUniform`, `Activation`, `dActivation` are copied into the file rather than imported from `Utils_RTLO`. | Same code duplicated. Cleanup. |
| F6 | `ReturnParameters` order changed (`wO` moved to position 9) compared with `DRTLO_GD_R2`. | Consistent within V3, but incompatible with V1 — another reason not to mix the versions. |

#### F1 in detail: granules share one network

`ReturnParameters()` returns the object's **own** lists and arrays. `fit()` then changes them in place: `self.wI[l] += …`, `self.wR[l] += …`, `self.wO += …`, and `self.pS[l] = …` writes into the shared list. Only `hP` is rebound to a new list, so the hidden state is the one thing that separates afterwards.

Check, run with the real class: build granule A, train it 3 steps, create granule B with `B.ReceiveParameters(A.ReturnParameters())`, then train **only B** for one step.

| Quantity of A | Changed by `B.fit`? | Same object as B's? |
|---|---|---|
| `wO` | **yes** | **yes** |
| `wR[0]` | **yes** | **yes** (same list) |
| `pS[0]` | **yes** | **yes** (same list) |
| `hP` | no | no (rebound in `fit`) |

Consequences in R6:

- Every granule created by `DefRnnParams` shares the weights and traces of its parent (G2 shares with G1, G3 with G2, and so on). The "one RNN per granule" structure collapses into **one shared network with several hidden states**.
- With `wtaG=False`, several granules are active at once, and `Adapt` calls `fit` once for each. The shared weights are therefore updated several times per sample (an effective learning rate × the number of active granules), and the shared traces get decayed and accumulated several times per step.
- Only granules produced by a merge escape: `MergeCloudsRNN` builds new averaged arrays.
- The granule-level interpretability claims in the paper (granule ↔ degradation stage ↔ its own predictor) do not hold for results produced this way.

**Fix:** deep-copy in `DefRnnParams`, e.g. `cloud.rnn.ReceiveParameters(copy.deepcopy(params))`, or make `ReturnParameters` return copies. Warm-starting a new granule from an existing one is a **good idea**: it answers E4, the cold start of new granules in R07. It just has to be a copy. Also consider copying from the most typical granule at creation time rather than always from `c[-2]`.

### 7.3 Overall ranking of the RFLO implementations

| Implementation | Learning rule vs Murray | Random feedback | Layers | Main problems |
|---|---|---|---|---|
| `RFLO.ipynb` (authors) | faithful | yes, $B$ | 1 | $q$ uses $x(t-1)$ (inactive in their demo) |
| `DRTLO_GD_R2.py` (R07) | same form; E1 timing bug | yes, $B^l$ per layer (DFA) | $L$ | E1, E3, E4, E5 |
| `eRNN_GD_R1.py` (R6) | **identical** to DRTLO_GD_R2 (E1) | yes, $B^l$ per layer (DFA) | $L$ | E1, E3, E5 **plus F1 (shared weights between granules)**, F3/F4 latent |

**Assessment:** `eRNN_GD_R1` is a refactor of `DRTLO_GD_R2` (seed parameter, internal $N\to10^{-N}$ conversion, merge-friendly `ReturnParameters`). It is **not** a change to the learning algorithm, and in particular it does **not** remove the random feedback. In isolation it is as coherent as `DRTLO_GD_R2`. Inside AutoCloud_V3, however, F1 makes R6's results unreliable as a multi-granule model. R6's design choices are closer to the manuscript (weighted aggregation with `wtaG=False`, merging enabled, warm-started granules), so it is the better base **once F1 and E1 are fixed**.
