# backend/bots/budget_bot.py
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, Query
import math

app = FastAPI(title="Budget Optimization Bot (Advanced)")

# -------------------------
# Models
# -------------------------
class OptionIn(BaseModel):
    id: str
    name: Optional[str] = None
    cost: float = Field(..., ge=0.0)  # total cost amount (e.g., total hotel stay)
    score: Optional[float] = None     # higher = better, if None we'll compute heuristics
    meta: Optional[Dict[str, Any]] = None

class CategoryIn(BaseModel):
    category: str
    options: List[OptionIn]
    # optional constraints for this category
    max_picks: Optional[int] = 1        # for mckp default 1; for flexible you can put >1
    weight: Optional[float] = 1.0       # relative importance when normalizing scores

class OptimizeRequest(BaseModel):
    budget: float = Field(..., gt=0.0)
    categories: Optional[List[CategoryIn]] = None
    # mode: "mckp" (one-per-category) or "flexible" (0/1 knapsack over pooled items)
    mode: Optional[str] = "mckp"
    # optional: normalize scores to 0..1 across all options before optimizing
    normalize_scores: Optional[bool] = True

class SelectedOption(BaseModel):
    id: str
    name: Optional[str] = None
    cost: float
    score: float
    category: Optional[str] = None
    meta: Optional[Dict[str, Any]] = None

class OptimizeResponse(BaseModel):
    selected: List[SelectedOption]
    total_cost: float
    total_score: float
    within_budget: bool
    mode: str

# -------------------------
# Helpers
# -------------------------
def _safe_score(opt: OptionIn) -> float:
    # if score missing, derive a heuristic (higher cost -> assume better up to a point)
    if opt.score is not None:
        return float(opt.score)
    # heuristic: base score as inverse cost (cheap = low score) + small randomish function
    # but deterministic: use sqrt(cost)
    return float(math.sqrt(opt.cost) if opt.cost > 0 else 0.0)

def normalize_scores(categories: List[CategoryIn], category_weights: Optional[Dict[str,float]] = None):
    """
    Normalize scores across all options into 0..1 and apply per-category weights.
    Returns a parallel structure of scores (same shape as categories).
    """
    all_scores = []
    for c in categories:
        for o in c.options:
            all_scores.append(_safe_score(o))
    if not all_scores:
        return  # nothing to do
    lo = min(all_scores)
    hi = max(all_scores)
    span = hi - lo if hi != lo else 1.0
    # compute normalized score for each option
    for c in categories:
        w = (category_weights or {}).get(c.category, c.weight if hasattr(c, 'weight') else getattr(c, 'weight', 1.0))
        for o in c.options:
            raw = _safe_score(o)
            norm = (raw - lo) / span
            # apply category weight
            o._computed_score = norm * (w if w is not None else 1.0)

# -------------------------
# Solvers
# -------------------------
def solve_mckp(categories: List[CategoryIn], budget: float) -> Tuple[List[SelectedOption], float, float]:
    """
    Multi-Choice Knapsack: at most one pick from each category (we allow skip).
    DP over categories, cost scaled to integer cents.
    Returns selected options list, total_cost, total_score.
    """
    scale = 100  # cents
    B = int(round(budget * scale))

    # prepare options lists with computed scores and keep original references
    cats_opts = []
    for c in categories:
        opts = []
        for o in c.options:
            score = getattr(o, "_computed_score", _safe_score(o))
            opts.append({"obj": o, "cost_c": int(round(o.cost * scale)), "score": float(score)})
        cats_opts.append(opts)

    n = len(cats_opts)
    # dp[i][c] = best score using first i categories with cost c
    # We keep dp as list of size B+1 and store parent pointers per stage for backtracking
    dp = [-1e9] * (B + 1)
    dp[0] = 0.0
    parents = []  # list of parent arrays for each category

    for idx in range(n):
        opts = cats_opts[idx]
        dp2 = [-1e9] * (B + 1)
        parent = [None] * (B + 1)
        for cost_so_far in range(B + 1):
            if dp[cost_so_far] < -1e8:
                continue
            # Option: skip this category (carryover)
            if dp[cost_so_far] > dp2[cost_so_far]:
                dp2[cost_so_far] = dp[cost_so_far]
                parent[cost_so_far] = (cost_so_far, None)  # None indicates skip
            # Try each option in this category
            for opt_index, opt in enumerate(opts):
                oc = opt["cost_c"]
                nc = cost_so_far + oc
                if nc <= B:
                    sc = dp[cost_so_far] + opt["score"]
                    if sc > dp2[nc]:
                        dp2[nc] = sc
                        parent[nc] = (cost_so_far, opt_index)
        dp = dp2
        parents.append(parent)

    # find the best score and cost
    best_score = max(dp)
    best_cost = dp.index(best_score)

    # backtrack
    picks = [None] * n
    c = best_cost
    for i in range(n - 1, -1, -1):
        parent = parents[i]
        p = parent[c]
        if p is None:
            picks[i] = None
            # c unchanged
        else:
            prev_cost, opt_idx = p
            if opt_idx is not None:
                picks[i] = opt_idx
            else:
                picks[i] = None
            c = prev_cost

    # assemble results
    selected = []
    total_cost = 0.0
    total_score = 0.0
    for i, pick in enumerate(picks):
        if pick is None:
            continue
        chosen = cats_opts[i][pick]["obj"]
        sc = cats_opts[i][pick]["score"]
        selected.append(SelectedOption(
            id=chosen.id, name=chosen.name, cost=chosen.cost, score=sc, category=categories[i].category, meta=chosen.meta
        ))
        total_cost += chosen.cost
        total_score += sc

    return selected, round(total_cost, 2), round(total_score, 6)

def solve_flexible(categories: List[CategoryIn], budget: float, max_picks_by_category: Optional[Dict[str,int]] = None) -> Tuple[List[SelectedOption], float, float]:
    """
    Flexible solver: treat all options as items in a 0/1 knapsack; allows picking multiple items.
    Enforces per-category max picks if provided. Returns selected items list, total cost, total score.
    """
    scale = 100
    B = int(round(budget * scale))

    items = []  # (cost_c, score, category, OptionIn)
    for c in categories:
        for o in c.options:
            sc = getattr(o, "_computed_score", _safe_score(o))
            items.append({"cost_c": int(round(o.cost * scale)), "score": float(sc), "cat": c.category, "obj": o})

    n = len(items)
    # If no items, return empty
    if n == 0:
        return [], 0.0, 0.0

    # dp[c] = best score for cost c; track parent with (prev_cost, item_index)
    dp = [-1e9] * (B + 1)
    dp[0] = 0.0
    parent = [None] * (B + 1)

    # To enforce per-category max picks we need to track counts per category.
    # That makes DP state explode if we track counts in dp. Instead we use greedy + DP hybrid:
    # 1) If no per-category constraints, do classic 0/1 knapsack DP (works).
    # 2) If per-category max constraints exist, we apply a two-phase heuristic:
    #    - For each category, sort options by score/cost, take top `max_picks` as candidate pool.
    #    - Run 0/1 knapsack on pooled candidates.
    # This is a pragmatic compromise for moderate sizes.
    if max_picks_by_category:
        # filter items by top-K per category
        pool = []
        by_cat: Dict[str, List[Dict]] = {}
        for it in items:
            by_cat.setdefault(it["cat"], []).append(it)
        for cat, list_items in by_cat.items():
            topk = max_picks_by_category.get(cat, None)
            # sort by score desc (or score per cost)
            list_items_sorted = sorted(list_items, key=lambda x: (x["score"], -x["cost_c"]), reverse=True)
            if topk is not None and topk < len(list_items_sorted):
                list_items_sorted = list_items_sorted[:topk*3]  # allow some extra candidates (safety)
            pool.extend(list_items_sorted)
        items = pool
        n = len(items)

    # classic 0/1 knapsack DP with parent tracking
    # We iterate items and update dp in reverse order for 0/1 knapsack
    # to reconstruct choices, we maintain a parent mapping with (prev_cost, item_index)
    parents_per_stage: List[List[Optional[Tuple[int,int]]]] = []
    dp_stages = []
    dp = [-1e9] * (B + 1)
    dp[0] = 0.0
    dp_stages.append(dp.copy())
    parents_per_stage.append([None] * (B + 1))

    for idx, it in enumerate(items):
        cost_i = it["cost_c"]
        score_i = it["score"]
        dp2 = dp.copy()
        parent2 = [None] * (B + 1)
        for c in range(B + 1):
            parent2[c] = None
        for c in range(B, cost_i - 1, -1):
            if dp[c - cost_i] > -1e8:
                candidate = dp[c - cost_i] + score_i
                if candidate > dp2[c]:
                    dp2[c] = candidate
                    parent2[c] = (c - cost_i, idx)
        dp = dp2
        dp_stages.append(dp.copy())
        parents_per_stage.append(parent2)

    # pick best
    best_score = max(dp)
    best_cost = dp.index(best_score)

    # backtrack through stages
    selected_indices = []
    c = best_cost
    for stage in range(len(items), 0, -1):
        parent = parents_per_stage[stage]
        p = parent[c]
        if p is None:
            # item not picked at this stage
            # we need to find a previous 'c' with same dp value; however
            # easier: compare dp_stages[stage][c] vs dp_stages[stage-1][c] - if equal then not picked
            if dp_stages[stage][c] == dp_stages[stage-1][c]:
                # not picked
                continue
            # else if different, we must find prior cost where dp_stages[stage-1][prev_c] + score = dp_stages[stage][c]
            found = False
            for prev_c in range(0, c+1):
                if dp_stages[stage-1][prev_c] > -1e8 and abs(dp_stages[stage-1][prev_c] + items[stage-1]["score"] - dp_stages[stage][c]) < 1e-9 and prev_c + items[stage-1]["cost_c"] == c:
                    selected_indices.append(stage-1)
                    c = prev_c
                    found = True
                    break
            if not found:
                # treat as not selected
                continue
        else:
            prev_c, item_idx = p
            selected_indices.append(item_idx)
            c = prev_c

    # deduplicate and map to items
    selected_indices = sorted(set(selected_indices))
    selected = []
    total_cost = 0.0
    total_score = 0.0
    for idx in selected_indices:
        it = items[idx]
        o = it["obj"]
        sc = it["score"]
        selected.append(SelectedOption(id=o.id, name=o.name, cost=o.cost, score=sc, category=it["cat"], meta=o.meta))
        total_cost += o.cost
        total_score += sc

    return selected, round(total_cost, 2), round(total_score, 6)

# -------------------------
# API
# -------------------------
@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    if not req.categories:
        raise HTTPException(status_code=400, detail="Provide `categories` in request (no remote fetching in this endpoint).")

    # validate mode
    mode = (req.mode or "mckp").lower()
    if mode not in ("mckp", "flexible"):
        raise HTTPException(status_code=400, detail="mode must be 'mckp' or 'flexible'")

    # normalize scores if requested
    if req.normalize_scores:
        normalize_scores(req.categories)

    # build per-category max picks map
    max_picks = {}
    for c in req.categories:
        if c.max_picks is not None:
            max_picks[c.category] = c.max_picks

    # run solver
    if mode == "mckp":
        selected, total_cost, total_score = solve_mckp(req.categories, req.budget)
    else:
        selected, total_cost, total_score = solve_flexible(req.categories, req.budget, max_picks_by_category=max_picks or None)

    return OptimizeResponse(selected=selected, total_cost=round(total_cost,2), total_score=round(total_score,6), within_budget=(total_cost <= req.budget + 1e-6), mode=mode)

@app.get("/health")
def health():
    return {"status":"ok","bot":"budget_advanced"}
