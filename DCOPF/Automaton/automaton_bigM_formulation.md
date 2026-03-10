# Automaton Line-Tripping — Big-M MILP Formulation
## DC OPF with PTDF Flow Model

---

## 1. Intuition

A protection relay (automaton) monitors the **natural line flow** — the flow
that physics would impose if no thermal limit existed.  When that natural flow
exceeds the thermal rating, the relay **deterministically** opens the breaker:

```
Natural flow P̃ₗ = PTDF · Pnet            (what physics imposes)
                                           ↓
          |P̃ₗ| ≤ Pˡᵐᵃˣ  →  zₗ = 1  (stay connected, P̃ₗ realised)
          |P̃ₗ| > Pˡᵐᵃˣ  →  zₗ = 0  (trip, actual flow Pₗ = 0)
```

The challenge: `zₗ` is **not** a free decision of the optimiser — it is
*determined* by the physics.  Big-M constraints encode this determinism as
linear inequalities over binary variables.

---

## 2. Sets and Parameters

| Symbol | Description |
|--------|-------------|
| `𝒩` | Set of buses, `𝒩 = {B1, B2}` |
| `𝒢` | Set of generators, `𝒢 = {G1, G2}` |
| `ℒ` | Set of lines, `ℒ = {L12}` |
| `b(g)` | Bus at which generator `g` is connected |
| `Pᵍᵐⁱⁿ, Pᵍᵐᵃˣ` | Min / max output of generator `g` (MW) |
| `Pᵈ_b` | Active load at bus `b` (MW) |
| `PTDFₗ,ᵍ` | Sensitivity of flow on line `l` to a unit injection at bus `b(g)` (MW/MW) |
| `Pˡᵐᵃˣ` | Thermal rating of line `l` (MW) |
| `cᵍ` | Linear generation cost of generator `g` ($/MWh) |
| `M` | Big-M constant (chosen as a valid upper bound; see §6) |

---

## 3. Decision Variables

| Variable | Domain | Description |
|----------|--------|-------------|
| `Pᵍ` | `[Pᵍᵐⁱⁿ, Pᵍᵐᵃˣ]` | Generator output (MW) |
| `Pₗ` | `ℝ` | **Actual** line flow (MW) — zero when tripped |
| `P̃ₗ` | `ℝ` | **Natural** (unconstrained) line flow (MW) |
| `zₗ` | `{0, 1}` | Line status: `1` = connected, `0` = tripped |
| `βₗ` | `{0, 1}` | Overload direction: `1` = positive overload, `0` = negative |

---

## 4. Objective

Minimise total generation cost:

$$\min \sum_{g \in \mathcal{G}} c_g \cdot P_g$$

---

## 5. Constraints

### 5.1 Natural Flow Definition (PTDF)

The natural flow is what physics imposes unconditionally,
independent of the line status:

$$\tilde{P}_l = \sum_{g \in \mathcal{G}} \text{PTDF}_{l,g} \cdot \bigl(P_g - P^d_{b(g)}\bigr)
\qquad \forall l \in \mathcal{L}$$

*For the two-bus system:* `PTDF_{L12, G1} = 0` (slack), `PTDF_{L12, G2} = −1`, so:

$$\tilde{P}_{12} = -\,(P_{G2} - P^d_{B2}) = 250 - P_{G2}$$

---

### 5.2 Nodal Power Balance

The balance equation uses the **actual** flow `Pₗ`, which is zero when the
line is tripped:

$$\sum_{g \in \mathcal{G}_b} P_g
  - \!\!\sum_{l:\,\text{from}(l)=b}\!\! P_l
  + \!\!\sum_{l:\,\text{to}(l)=b}\!\! P_l
  = P^d_b
\qquad \forall b \in \mathcal{N}$$

*For the two-bus system:*

$$P_{G1} - P_{12} = 0 \qquad \text{(Bus 1)}$$
$$P_{G2} + P_{12} = 250 \qquad \text{(Bus 2)}$$

> **Note:** When `z_{L12} = 0`, constraint §5.4 forces `P₁₂ = 0`, so each bus
> must satisfy its own balance using local generation only.

---

### 5.3 Actual Flow Equals Natural Flow When Connected

When the line is connected (`zₗ = 1`), the actual and natural flows must agree.
When tripped (`zₗ = 0`), the constraint is relaxed by Big-M:

$$\tilde{P}_l - M(1 - z_l) \;\leq\; P_l \;\leq\; \tilde{P}_l + M(1 - z_l)
\qquad \forall l \in \mathcal{L}$$

$$\boxed{
z_l = 1 \;\Rightarrow\; P_l = \tilde{P}_l
\qquad
z_l = 0 \;\Rightarrow\; P_l \text{ unconstrained by this pair}
}$$

---

### 5.4 Zero Flow When Tripped

$$-P^{\max}_l \cdot z_l \;\leq\; P_l \;\leq\; P^{\max}_l \cdot z_l
\qquad \forall l \in \mathcal{L}$$

$$\boxed{
z_l = 0 \;\Rightarrow\; P_l = 0
\qquad
z_l = 1 \;\Rightarrow\; -P^{\max}_l \leq P_l \leq P^{\max}_l
}$$

> Together, §5.3 and §5.4 implement: "if connected, flow = PTDF result bounded
> by thermal limit; if tripped, flow = 0."

---

### 5.5 Automaton Trip Logic — Forward Implication

**"Trip if and only if the natural flow would exceed the thermal limit"**

#### 5.5.1 Forward: Overload forces trip  
*(if* `|P̃ₗ| > Pˡᵐᵃˣ` *then* `zₗ = 0`*, equivalently:* `zₗ = 1` *→* `|P̃ₗ| ≤ Pˡᵐᵃˣ`*)*

$$\tilde{P}_l \;\leq\; P^{\max}_l + M(1 - z_l) \qquad \forall l \in \mathcal{L} \tag{A1+}$$
$$\tilde{P}_l \;\geq\; -P^{\max}_l - M(1 - z_l) \qquad \forall l \in \mathcal{L} \tag{A1−}$$

When `zₗ = 1`: enforces `−Pˡᵐᵃˣ ≤ P̃ₗ ≤ Pˡᵐᵃˣ` (infeasible if overloaded → optimizer must set `zₗ = 0`).  
When `zₗ = 0`: adds `±M` slack, fully relaxed.

#### 5.5.2 Backward: No gratuitous disconnection  
*(if* `zₗ = 0` *then* `|P̃ₗ| > Pˡᵐᵃˣ`*)*

Introduce direction indicator `βₗ ∈ {0,1}`:  `βₗ = 1` for positive overload, `βₗ = 0` for negative:

$$P^{\max}_l\,(1 - z_l) - M\,\beta_l \;\leq\; \tilde{P}_l
\qquad \forall l \in \mathcal{L} \tag{A2+}$$

$$\tilde{P}_l \;\leq\; -P^{\max}_l\,(1 - z_l) + M\,(1 - \beta_l)
\qquad \forall l \in \mathcal{L} \tag{A2−}$$

$$\beta_l \;\leq\; 1 - z_l \qquad \forall l \in \mathcal{L} \tag{A2 coupling}$$

When `zₗ = 0`: exactly one of `βₗ = 1` or `βₗ = 0` is active, enforcing either `P̃ₗ ≥ +Pˡᵐᵃˣ` or `P̃ₗ ≤ −Pˡᵐᵃˣ`.  
When `zₗ = 1`: `βₗ` is forced to 0 by (A2 coupling), both (A2) constraints are relaxed by `M`.

---

## 6. Complete MILP Summary

$$\min_{P_g,\, P_l,\, \tilde{P}_l,\, z_l,\, \beta_l} \quad \sum_{g} c_g P_g$$

subject to:

| # | Constraint | Scope |
|---|------------|-------|
| Natural flow | $\tilde{P}_l = \sum_g \text{PTDF}_{l,g}(P_g - P^d_{b(g)})$ | $\forall l$ |
| Balance | $\sum_{g \in \mathcal{G}_b} P_g - \sum P_l^{\text{out}} + \sum P_l^{\text{in}} = P^d_b$ | $\forall b$ |
| Gen limits | $P^{\min}_g \leq P_g \leq P^{\max}_g$ | $\forall g$ |
| §5.3 flow–status link | $P_l - M(1-z_l) \leq \tilde{P}_l \leq P_l + M(1-z_l)$ | $\forall l$ |
| §5.4 zero-when-tripped | $-P^{\max}_l z_l \leq P_l \leq P^{\max}_l z_l$ | $\forall l$ |
| (A1+) trip if + overload | $\tilde{P}_l \leq P^{\max}_l + M(1-z_l)$ | $\forall l$ |
| (A1−) trip if − overload | $\tilde{P}_l \geq -P^{\max}_l - M(1-z_l)$ | $\forall l$ |
| (A2+) only trip if + OL | $P^{\max}_l(1-z_l) - M\beta_l \leq \tilde{P}_l$ | $\forall l$ |
| (A2−) only trip if − OL | $\tilde{P}_l \leq -P^{\max}_l(1-z_l) + M(1-\beta_l)$ | $\forall l$ |
| (A2 coupling) | $\beta_l \leq 1 - z_l$ | $\forall l$ |
| Integrality | $z_l \in \{0,1\},\; \beta_l \in \{0,1\}$ | $\forall l$ |

---

## 7. Big-M Bound

A valid and tight Big-M is:

$$M \;=\; \sum_{g} P^{\max}_g$$

This is an upper bound on any feasible natural flow magnitude, ensuring
the relaxed constraints are never binding in the wrong state while keeping
the LP relaxation as tight as possible (large M weakens the relaxation).

*For the two-bus system:*  `M = 200 + 150 = 350 MW`

---

## 8. Variable Count for the Two-Bus System

| Variable | Count |
|----------|-------|
| `Pg` (G1, G2) | 2 continuous |
| `P₁₂` (actual flow) | 1 continuous |
| `P̃₁₂` (natural flow) | 1 continuous |
| `z_{L12}` | 1 binary |
| `β_{L12}` | 1 binary |
| **Total** | **4 continuous + 2 binary** |

vs. the LP (no automaton): 3 continuous, 0 binary.

---

## 9. Logical Truth Table (Two-Bus Verification)

| `zₗ` | `βₗ` | Active constraints | Meaning |
|------|------|--------------------|---------|
| 1 | 0 | §5.3, §5.4 bind `Pₗ = P̃ₗ ∈ [−Pᵐᵃˣ, +Pᵐᵃˣ]` | Line on, normal operation |
| 0 | 1 | §5.4 forces `Pₗ=0`; (A2+) requires `P̃ₗ ≥ +Pᵐᵃˣ` | Tripped — positive overload |
| 0 | 0 | §5.4 forces `Pₗ=0`; (A2−) requires `P̃ₗ ≤ −Pᵐᵃˣ` | Tripped — negative overload |
| 1 | 1 | Excluded by (A2 coupling) `βₗ ≤ 1 − zₗ = 0` | Infeasible combination |
