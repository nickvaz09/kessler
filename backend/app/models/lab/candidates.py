"""The candidate pool.

Every candidate is fit on training events only and predicts the FINAL log10
risk of unseen events (regression) or an escalation score/probability for
events still below -6 at the cutoff (early warning). Anything tuned (early
stopping, decision threshold, blend, Platt calibration) is tuned inside the
TRAINING fold - by cross-fitting where the tuning set would otherwise be a
handful of high-risk events - so the outer cross-validation stays unbiased.

PRE-REGISTERED PRIMARY CANDIDATES (fixed before any real-data run; the ship
gate tests each as a single hypothesis, so there is no selection among many
candidates and no winner's curse. The one gated non-primary, serving_resid
- the only VIGIL the app's artifact format loads - is judged on its max-T
adjusted interval; everything else in the pool is reported as exploratory
with multiplicity-adjusted intervals):
    VIGIL          hurdle_lrp            (benchmark features)
    VIGIL serving  hurdle_lrp_serving    (what the live app can compute)
    VIGIL-E        e_hurdle
    VIGIL-E serving e_serving

Regression pool:
  lrp             organisers' latest-risk baseline (the bar to beat)
  sesc            the organisers' post-competition LightGBM control (Uriot
                  et al. 2022, §6.1; key kept for continuity - NOT team
                  sesc's winning entry, which was a hand-tuned rule cascade):
                  last admissible CDM, numerical columns only, delta target,
                  25 trees, trained on test-like events
  gbm_resid       residual boosted model on all benchmark features
  gbm_resid_hr    same, weighted toward the region the official score reads
  gbm_resid_aug   same + earlier-cutoff augmentation rows (grouped carve)
  gbm_quant       distributional (quantile) model, decision tuned for L
  ridge_phys      regularised linear model on compact physics/history
  hurdle          metric-aligned two-part model, free decision
  hurdle_lrp      PRIMARY: two-part model that keeps the baseline's high-risk
                  calls and adds only confident early catches
  hurdle_lrp_serving  PRIMARY (serving): the same on serving features
  serving_resid   serving residual model in the app's current artifact format
  hurdle_lrp_serving_bench  ROUND 2 PRIMARY (derived, no refit): the same
                  serving hurdle with its decision re-tuned for the official
                  test's published class mix (BENCH_HR_PREVALENCE)
  hurdle_eserving(_aux)  hurdle on VIGIL-E's production features (its
                  classifier gives the Round 2 VIGIL-E primary); _aux adds
                  near-miss labels (exploratory)

Early-warning pool (events with latest admissible risk < -6):
  e_logit         logistic regression on latest log-risk (the baseline)
  e_max           no-ML: highest risk seen so far
  e_physics       no-ML: shrink-projected risk from the encounter plane
  e_quant         P(final >= -6) from the distributional regressor
  e_hurdle        PRIMARY: P(final >= -6) from the hurdle classifier, which
                  learns from every high-risk-final event, not only the
                  events that escalate
  e_clf           class-weighted boosted classifier (Platt probability)
  e_rank_blend    rank average of e_hurdle, e_clf, e_physics (label-free)
  e_serving       PRIMARY (serving): classifier on VIGIL-E's production features
  e_hurdle_eserving       ROUND 2 PRIMARY: P(final >= -6) from hurdle_eserving -
                          e_hurdle's mechanism (learn from every high-risk-final
                          event) on the features the app computes
  e_hurdle_eserving_aux   the same with near-miss labels (exploratory)
(The "PRIMARY" tags above are Round 1's; Round 2's are in PRIMARY below.)
"""
from __future__ import annotations

import numpy as np

from backend.app.models.lab import metric
from backend.app.models.lab.data import HIGH_RISK, LOW_PREDICTION, RISK_FLOOR

# Round 1 (judged 2026-09-24; see docs/v2/vigil-lab.md, "Results on the real
# ESA data"). Kept for the record - these are no longer tested.
ROUND1_PRIMARY = {"vigil": "hurdle_lrp", "vigil_serving": "hurdle_lrp_serving",
                  "vigil_e": "e_hurdle", "vigil_e_serving": "e_serving"}
# Round 2 (pre-registered 2026-10-08, before any Round 2 real-data run; see
# the doc's "Round 2 pre-registration"). One primary per analysis:
#   vigil_serving    the sealed test's second and last look
#   vigil_e_serving  a second pooled-CV analysis of the same data
PRIMARY = {"vigil_serving": "hurdle_lrp_serving_bench",
           "vigil_e_serving": "e_hurdle_eserving"}

# The official test set's published class mix: 150 of its 2,167 events end
# high-risk (Uriot et al. 2022, §4.2 and Fig. 10). The organisers enriched it
# with high-risk events, so the L-optimal decision there differs from the
# training file's (0.8% high-risk among eligible events).
BENCH_HR_PREVALENCE = 150 / 2167

QUANTILES = np.array([0.01, 0.03, 0.06, 0.1, 0.2, 0.3, 0.4, 0.5,
                      0.6, 0.7, 0.8, 0.9, 0.94, 0.97, 0.99])
RESID_CLIP = 8.0          # decades - lab regression clip
SERVING_RESID_CLIP = 4.0  # = risk_model.RESID_CLIP_DECADES, so an exported
                          # serving model is evaluated exactly as the app serves it
MIN_HR_FOR_TUNING = 40    # fewer high-risk events than this -> fixed decision
FALLBACK_TAU, FALLBACK_ALPHA = 0.04, 0.5   # near the L-optimal operating point


def _xgb():
    import xgboost as xgb
    return xgb


def inner_split(n: int, strata: np.ndarray, seed: int, frac: float = 0.2,
                groups: np.ndarray | None = None):
    """Stratified (fit, valid) split of training rows. With ``groups``, all
    rows of a group land on the same side (augmentation rows travel with
    their parent event)."""
    from sklearn.model_selection import StratifiedShuffleSplit

    strata = np.asarray(strata)
    if groups is not None:
        groups = np.asarray(groups)
        ug, first = np.unique(groups, return_index=True)
        gfit, gval = inner_split(len(ug), strata[first], seed, frac)
        val_groups = set(ug[gval].tolist())
        is_val = np.array([g in val_groups for g in groups])
        return np.flatnonzero(~is_val), np.flatnonzero(is_val)
    _, inv = np.unique(strata, return_inverse=True)
    if len(np.unique(strata)) < 2 or np.min(np.bincount(inv)) < 2:
        rng = np.random.RandomState(seed)
        idx = rng.permutation(n)
        k = max(1, int(round(frac * n)))
        return idx[k:], idx[:k]
    sss = StratifiedShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    fit, val = next(sss.split(np.zeros(n), strata))
    return fit, val


def _gbm_params(seed: int, **over) -> dict:
    p = dict(n_estimators=2000, learning_rate=0.03, max_depth=5, min_child_weight=5,
             subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0, reg_alpha=0.5,
             gamma=0.0, tree_method="hist", random_state=seed, n_jobs=4,
             early_stopping_rounds=100)
    p.update(over)
    return p


class Context:
    """What a candidate sees for one fit."""

    def __init__(self, X, y, latest, strata, seed, eligible=None, extra=None):
        self.X = X
        self.y = np.asarray(y, float)
        self.latest = np.asarray(latest, float)
        self.strata = np.asarray(strata)
        self.seed = seed
        self.eligible = np.ones(len(self.y), bool) if eligible is None else np.asarray(eligible, bool)
        self.extra = extra or {}


def _platt(margin, y):
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y, int)
    if not (0 < y.sum() < len(y)):
        return None
    return LogisticRegression(C=1e3, max_iter=1000).fit(np.asarray(margin).reshape(-1, 1), y)


def _apply_platt(platt, margin):
    margin = np.asarray(margin, float).reshape(-1, 1)
    if platt is None:
        return 1.0 / (1.0 + np.exp(-margin[:, 0]))
    return platt.predict_proba(margin)[:, 1]


# --- regression candidates -----------------------------------------------------

class LRP:
    name = "lrp"
    track = "benchmark"

    def fit(self, ctx):
        return self

    def predict(self, X, latest):
        return metric.lrp(latest)


class GBMResid:
    """Boosted model on residual = final - latest.

    weight_mode   'flat' | 'hr' (focus where the official score reads)
    train_on      'all' | 'eligible' (only test-like training events)
    exclude       column-name prefixes to drop (sesc drops categoricals)
    """

    def __init__(self, name, track="benchmark", weight_mode="flat", augment=False,
                 params=None, early_stop=True, resid_clip=RESID_CLIP,
                 train_on="all", exclude=()):
        self.name, self.track = name, track
        self.resid_clip = resid_clip
        self.weight_mode, self.augment = weight_mode, augment
        self.params = params or {}
        self.early_stop = early_stop
        self.train_on, self.exclude = train_on, tuple(exclude)

    def _cols(self, ctx):
        names = ctx.extra.get("colnames")
        if not self.exclude or names is None:
            return None
        return np.array([not n.startswith(self.exclude) for n in names])

    def _weights(self, y, latest):
        if self.weight_mode == "hr":
            near = (y >= HIGH_RISK - 1.0) | (latest >= HIGH_RISK - 1.0)
            return np.where(near, 5.0, 1.0)
        return np.ones(len(y))

    def fit(self, ctx):
        xgb = _xgb()
        self.colmask = self._cols(ctx)
        rows = np.flatnonzero(ctx.eligible) if self.train_on == "eligible" else np.arange(len(ctx.y))
        X, y, latest = ctx.X[rows], ctx.y[rows], ctx.latest[rows]
        strata, groups = ctx.strata[rows], np.arange(len(rows))
        if self.augment and "aug" in ctx.extra:
            aX, ay, alat, aparent = ctx.extra["aug"]
            X = np.vstack([X, aX]); y = np.concatenate([y, ay]); latest = np.concatenate([latest, alat])
            strata = np.concatenate([strata, np.full(len(ay), -1)])
            # parent positions refer to ctx rows; remap to this fit's rows
            pos = {r: i for i, r in enumerate(rows)}
            groups = np.concatenate([groups, np.array([pos.get(p, -1 - i) for i, p in enumerate(aparent)])])
        if self.colmask is not None:
            X = X[:, self.colmask]
        resid = np.clip(y - latest, -self.resid_clip, self.resid_clip)
        w = self._weights(y, latest)
        params = _gbm_params(ctx.seed, **self.params)
        if self.early_stop:
            fit, val = inner_split(len(y), np.where(strata < 0, 0, strata), ctx.seed,
                                   groups=groups if self.augment else None)
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X[fit], resid[fit], sample_weight=w[fit],
                           eval_set=[(X[val], resid[val])], sample_weight_eval_set=[w[val]],
                           verbose=False)
        else:
            params.pop("early_stopping_rounds", None)
            self.model = xgb.XGBRegressor(**params)
            self.model.fit(X, resid, sample_weight=w, verbose=False)
        return self

    def predict_raw(self, X, latest):
        if self.colmask is not None:
            X = X[:, self.colmask]
        r = np.clip(self.model.predict(X), -self.resid_clip, self.resid_clip)
        return np.clip(latest + r, RISK_FLOOR, 0.0)

    def predict(self, X, latest):
        return metric.clip_low(self.predict_raw(X, latest))


def Sesc():
    # The organisers' post-competition LightGBM study (Uriot et al. 2022,
    # §6.1 - only 1 of its 5,000 models beat the baseline on both train and
    # test): 25 trees, feature_fraction 0.25, learning rate 0.05, default
    # num_leaves 31 and min_data_in_leaf 20 (= min_child_weight 20 for squared
    # error), the last CDM >= 2 days out, numerical columns only, target =
    # change in risk, trained on test-like events; §6.1's quantile-transformed
    # target is omitted. Emulated with xgboost's leaf-wise growth. This is not
    # team sesc's winning entry (§5.3.1: a hand-tuned threshold cascade, partly
    # fitted against the test leaderboard); the key is kept for continuity.
    return GBMResid("sesc", track="last", early_stop=False, train_on="eligible",
                    exclude=("l_ctype_",), params=dict(
                        n_estimators=25, learning_rate=0.05, max_depth=0, max_leaves=31,
                        grow_policy="lossguide", colsample_bytree=0.25, subsample=1.0,
                        min_child_weight=20, reg_lambda=0.0, reg_alpha=0.0))


def _cdf_normal_score(values: np.ndarray, levels: np.ndarray, x: float) -> float:
    """CDF at x from quantile knots, interpolating in normal-score space and
    extrapolating linearly beyond the outer knots (smooth, non-zero tails -
    flat tails made ~97% of events tie at probability 0)."""
    from scipy.stats import norm

    z = norm.ppf(levels)
    v = np.asarray(values, float)
    if x <= v[0] or x >= v[-1]:
        i, j = (0, 1) if x <= v[0] else (-2, -1)
        dv = v[j] - v[i]
        slope = (z[j] - z[i]) / dv if dv > 1e-9 else 0.0
        zx = z[i] + slope * (x - v[i]) if x <= v[0] else z[j] + slope * (x - v[j])
        if slope == 0.0:
            zx = -8.0 if x <= v[0] else 8.0
    else:
        zx = np.interp(x, v, z)
    return float(norm.cdf(zx))


class GBMQuantile:
    """Multi-quantile boosted model of the residual. Point = median;
    P(final >= -6) from the predicted CDF; decision threshold tuned for L on
    the inner carve (a control, not a primary candidate)."""

    name = "gbm_quant"
    track = "benchmark"

    def fit(self, ctx):
        xgb = _xgb()
        resid = np.clip(ctx.y - ctx.latest, -RESID_CLIP, RESID_CLIP)
        fit, val = inner_split(len(ctx.y), ctx.strata, ctx.seed)
        params = _gbm_params(ctx.seed, objective="reg:quantileerror",
                             quantile_alpha=QUANTILES, max_depth=4)
        self.model = xgb.XGBRegressor(**params)
        self.model.fit(ctx.X[fit], resid[fit], eval_set=[(ctx.X[val], resid[val])], verbose=False)
        v = val[ctx.eligible[val]] if ctx.eligible[val].any() else val
        p = self.prob_high(ctx.X[v], ctx.latest[v])
        med = self._median(ctx.X[v], ctx.latest[v])
        best = (np.inf, 0.5)
        for tau in np.linspace(0.01, 0.95, 48):
            L = metric.official(self._decide(med, p, tau), ctx.y[v])["L"]
            if L < best[0]:
                best = (L, tau)
        self.tau = best[1]
        return self

    def _quantiles(self, X):
        q = np.asarray(self.model.predict(X), float).reshape(len(X), -1)
        return np.sort(q, axis=1)

    def _median(self, X, latest):
        q = self._quantiles(X)
        return np.clip(latest + q[:, int(np.argmin(np.abs(QUANTILES - 0.5)))], RISK_FLOOR, 0.0)

    def prob_high(self, X, latest):
        q = self._quantiles(X)
        need = HIGH_RISK - np.asarray(latest, float)
        return np.array([1.0 - _cdf_normal_score(q[i], QUANTILES, need[i]) for i in range(len(q))])

    @staticmethod
    def _decide(point, p_high, tau):
        return np.where(p_high >= tau, np.maximum(point, HIGH_RISK), LOW_PREDICTION)

    def predict(self, X, latest):
        return self._decide(self._median(X, latest), self.prob_high(X, latest), self.tau)


class Hurdle:
    """Metric-aligned two-part model - the core design of the lab.

    The official score charges regression error only on events whose TRUE
    final risk is >= -6. Given a high-risk call, the Bayes-optimal point
    forecast for MSE_HR is therefore E[final | x, final >= -6] - not
    E[final | x], which ordinary regressors estimate and which the ~70% of
    events that fade to -30 drag down. F2 is a separate decision:

      p(x) = P(final >= -6 | x)       boosted classifier, Platt-calibrated
      m(x) = E[final | x, final>=-6]  boosted regressor on high-risk-final
                                      events, residual to max(latest, -6)
      pred = max(value, -6) if decision(x) else -6.001

      mode 'free': decision = p(x) >= tau
      mode 'lrp' : decision = latest >= -6 (the baseline's call, keeping its
                   F2) or p(x) >= tau; for baseline-high events value =
                   alpha*m(x) + (1-alpha)*latest, alpha in {0, 0.5, 1}
                   (alpha = 0 is exactly the baseline there).

    CROSS-FITTING: the training fold is split into K inner folds. Each inner
    model (classifier + regressor, early-stopped on a carve of ITS OWN
    training part) predicts its held-out inner fold, giving out-of-fold p and
    m for every training event. Platt calibration and the (tau, alpha)
    decision are then tuned on all eligible training events instead of a
    ~25% carve holding a dozen high-risk events. The final predictor is the
    average of the K inner models (cross-validation bagging).
    """

    INNER_FOLDS = 5

    def __init__(self, name, mode="free", track="benchmark", aux=False):
        """aux=True: for events still below -6 at the cutoff, the classifier
        learns "crossed -6 at any CDM inside the cutoff" (near-miss labels;
        late CDMs are LABELS only, never inputs). Platt calibration and the
        decision always use the true final label, so the output still means
        P(final >= -6)."""
        self.name, self.mode, self.track, self.aux = name, mode, track, aux

    def _fit_pair(self, X, y, hr, anchor, strata, seed, hr_cls=None):
        xgb = _xgb()
        hr_cls = hr if hr_cls is None else hr_cls
        fit, es = inner_split(len(y), strata, seed, frac=0.15)
        pos = max(1, int(hr_cls[fit].sum()))
        cp = _gbm_params(seed, objective="binary:logistic", eval_metric="logloss",
                         max_depth=4, min_child_weight=3, n_estimators=1500,
                         scale_pos_weight=float(np.sqrt(max(1, len(fit) - pos) / pos)))
        clf = xgb.XGBClassifier(**cp)
        clf.fit(X[fit], hr_cls[fit], eval_set=[(X[es], hr_cls[es])], verbose=False)
        hf, he = fit[hr[fit] == 1], es[hr[es] == 1]
        rp = _gbm_params(seed, max_depth=3, min_child_weight=5, n_estimators=800,
                         learning_rate=0.02, reg_lambda=10.0)
        if len(hf) >= 10 and len(he) >= 3:
            reg = xgb.XGBRegressor(**rp)
            reg.fit(X[hf], y[hf] - anchor[hf], eval_set=[(X[he], y[he] - anchor[he])], verbose=False)
        else:
            rp.pop("early_stopping_rounds", None)
            reg = xgb.XGBRegressor(**{**rp, "n_estimators": 100})
            allhr = np.flatnonzero(hr == 1)
            idx = allhr if len(allhr) else fit
            reg.fit(X[idx], y[idx] - anchor[idx], verbose=False)
        return clf, reg

    def fit(self, ctx):
        from sklearn.model_selection import StratifiedKFold

        X, y, latest = ctx.X, ctx.y, ctx.latest
        hr = (y >= HIGH_RISK).astype(int)
        hr_cls = hr
        if self.aux:
            y_aux = ctx.extra.get("y_aux")
            if y_aux is None:
                raise ValueError(f"{self.name}: aux labels need ctx.extra['y_aux']")
            hr_cls = np.where(latest < HIGH_RISK, np.asarray(y_aux) >= HIGH_RISK, hr).astype(int)
        anchor = np.maximum(latest, HIGH_RISK)
        n = len(y)
        k = self.INNER_FOLDS
        _, inv = np.unique(ctx.strata, return_inverse=True)
        strata = ctx.strata if np.min(np.bincount(inv)) >= k else hr
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=ctx.seed)
        oof_margin, oof_m = np.zeros(n), np.zeros(n)
        self.clfs, self.regs = [], []
        for j, (tr, he) in enumerate(skf.split(np.zeros(n), strata)):
            clf, reg = self._fit_pair(X[tr], y[tr], hr[tr], anchor[tr], strata[tr], ctx.seed + 17 * j,
                                      hr_cls=hr_cls[tr])
            oof_margin[he] = clf.predict(X[he], output_margin=True)
            oof_m[he] = np.clip(anchor[he] + reg.predict(X[he]), HIGH_RISK, 0.0)
            self.clfs.append(clf); self.regs.append(reg)
        self.platt = _platt(oof_margin, hr)               # calibrated on the TRUE label
        p = _apply_platt(self.platt, oof_margin)
        el = ctx.eligible
        # honest out-of-fold inputs to the decision, kept so the decision can
        # be re-tuned for another class mix without refitting (with_prior)
        self._tune = {"p": p[el], "m": oof_m[el], "latest": latest[el], "y": y[el]}
        self.tuned = bool(hr[el].sum() >= MIN_HR_FOR_TUNING)
        if not self.tuned:
            self.tau, self.alpha = FALLBACK_TAU, (FALLBACK_ALPHA if self.mode == "lrp" else 1.0)
            return self
        best = (np.inf, FALLBACK_TAU, FALLBACK_ALPHA)
        taus = np.concatenate([np.linspace(0.005, 0.1, 20), np.linspace(0.12, 0.9, 27)])
        alphas = (0.0, 0.5, 1.0) if self.mode == "lrp" else (1.0,)
        for tau in taus:
            for alpha in alphas:
                pred = self._decide(p[el], oof_m[el], latest[el], tau, alpha)
                L = metric.official(pred, y[el])["L"]
                if L < best[0] - 1e-12:
                    best = (L, tau, alpha)
        _, self.tau, self.alpha = best
        return self

    def with_prior(self, name: str, prevalence: float = BENCH_HR_PREVALENCE) -> "Hurdle":
        """The same fitted model (same boosters, same calibration) with its
        decision (tau, alpha) re-tuned for a population whose high-risk
        prevalence is ``prevalence``: the cross-fitted search is repeated with
        label-shift importance weights. Only the operating point changes."""
        import copy

        t = self._tune
        hr = t["y"] >= HIGH_RISK
        out = copy.copy(self)
        out.name = name
        w = metric.label_shift_weights(hr, prevalence)
        if not self.tuned:
            # few high-risk events: map the fixed fallback through the prior shift
            p_tr = float(hr.mean()) if len(hr) else 0.0
            k = (prevalence / (1 - prevalence)) / (p_tr / (1 - p_tr)) if 0 < p_tr < 1 else 1.0
            odds = (FALLBACK_TAU / (1 - FALLBACK_TAU)) / k
            out.tau = odds / (1 + odds)
            return out
        # a higher prior lowers the useful threshold on train-prior probabilities
        taus = np.concatenate([np.geomspace(5e-4, 5e-3, 8), np.linspace(0.005, 0.1, 20),
                               np.linspace(0.12, 0.9, 27)])
        alphas = (0.0, 0.5, 1.0) if self.mode == "lrp" else (1.0,)
        best = (np.inf, self.tau, self.alpha)
        for tau in taus:
            for alpha in alphas:
                pred = self._decide(t["p"], t["m"], t["latest"], tau, alpha)
                L = metric.official(pred, t["y"], weights=w)["L"]
                if L < best[0] - 1e-12:
                    best = (L, tau, alpha)
        _, out.tau, out.alpha = best
        return out

    def _margin(self, X):
        return np.mean([c.predict(X, output_margin=True) for c in self.clfs], axis=0)

    def prob_high(self, X, latest):
        return _apply_platt(self.platt, self._margin(X))

    def _cond(self, X, latest):
        anchor = np.maximum(np.asarray(latest, float), HIGH_RISK)
        r = np.mean([g.predict(X) for g in self.regs], axis=0)
        return np.clip(anchor + r, HIGH_RISK, 0.0)

    def _decide(self, p, m, latest, tau, alpha):
        latest = np.asarray(latest, float)
        if self.mode == "lrp":
            base_hr = latest >= HIGH_RISK
            val_hr = np.where(base_hr, alpha * m + (1 - alpha) * latest, m)
            hi = base_hr | (p >= tau)
        else:
            val_hr, hi = m, p >= tau
        return np.where(hi, np.maximum(val_hr, HIGH_RISK), LOW_PREDICTION)

    def predict(self, X, latest):
        return self._decide(self.prob_high(X, latest), self._cond(X, latest),
                            latest, self.tau, self.alpha)


class RidgePhys:
    name = "ridge_phys"
    track = "benchmark"
    COLS = ("h_risk_last", "h_risk_max", "h_risk_slope", "h_risk_quad_extrap",
            "h_risk_delta_last", "h_last_tta", "h_n_inputs", "h_log_sigma_rate",
            "h_sigma_shrink_to_eval", "h_headroom_mre", "p_dilution_index",
            "p_log10_pcmax_gap", "p_proj_risk_trend", "p_proj_risk_s50",
            "h_risk_last_is_floor", "h_miss_log_ratio")

    def fit(self, ctx):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import RidgeCV
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        self.idx = [ctx.extra["colindex"][c] for c in self.COLS if c in ctx.extra["colindex"]]
        resid = np.clip(ctx.y - ctx.latest, -RESID_CLIP, RESID_CLIP)
        self.model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                   RidgeCV(alphas=np.logspace(-2, 3, 12)))
        self.model.fit(ctx.X[:, self.idx], resid)
        return self

    def predict(self, X, latest):
        r = np.clip(self.model.predict(X[:, self.idx]), -RESID_CLIP, RESID_CLIP)
        return metric.clip_low(np.clip(latest + r, RISK_FLOOR, 0.0))


# --- early-warning candidates -----------------------------------------------------

class ELogit:
    name = "e_logit"
    track = "benchmark"

    def fit(self, ctx):
        from sklearn.linear_model import LogisticRegression
        m = ctx.latest < HIGH_RISK
        self.model = LogisticRegression(max_iter=1000).fit(
            ctx.latest[m].reshape(-1, 1), ctx.y[m] >= HIGH_RISK)
        return self

    def score(self, X, latest):
        return self.model.predict_proba(np.asarray(latest).reshape(-1, 1))[:, 1]

    prob = score


class EColumn:
    """No-ML score = one feature column (higher = more likely to escalate)."""

    def __init__(self, name, column):
        self.name, self.column, self.track = name, column, "benchmark"

    def fit(self, ctx):
        self.j = ctx.extra["colindex"].get(self.column)
        return self

    def score(self, X, latest):
        if self.j is None:
            return np.asarray(latest, float)
        s = X[:, self.j].astype(float)
        return np.where(np.isfinite(s), s, np.asarray(latest, float))

    prob = None


class EClassifier:
    """Class-weighted classifier on escalation candidates. Scores are Platt
    probabilities (comparable across the models of different folds, unlike
    raw margins)."""

    def __init__(self, name, track="benchmark"):
        self.name, self.track = name, track

    def fit(self, ctx):
        xgb = _xgb()
        m = ctx.latest < HIGH_RISK
        X, y = ctx.X[m], (ctx.y[m] >= HIGH_RISK).astype(int)
        fit, val = inner_split(len(y), y, ctx.seed, frac=0.25)
        pos = max(1, int(y[fit].sum()))
        params = _gbm_params(ctx.seed, objective="binary:logistic", eval_metric="aucpr",
                             max_depth=3, min_child_weight=3, learning_rate=0.03,
                             scale_pos_weight=float((len(fit) - pos) / pos),
                             colsample_bytree=0.5, n_estimators=1500)
        self.model = xgb.XGBClassifier(**params)
        self.model.fit(X[fit], y[fit], eval_set=[(X[val], y[val])], verbose=False)
        margin = self.model.predict(X[val], output_margin=True)
        self.platt = _platt(margin, y[val])
        self.val_prob = _apply_platt(self.platt, margin)   # held-out, for export thresholds
        return self

    def margin(self, X):
        return np.asarray(self.model.predict(X, output_margin=True), float)

    def prob(self, X, latest):
        return _apply_platt(self.platt, self.margin(X))

    def score(self, X, latest):
        return self.prob(X, latest)


def rank01(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata
    x = np.asarray(x, float)
    if len(x) <= 1:
        return np.zeros(len(x))
    return (rankdata(x) - 1) / (len(x) - 1)


def regression_pool() -> list:
    return [LRP(), Sesc(),
            GBMResid("gbm_resid"),
            GBMResid("gbm_resid_hr", weight_mode="hr"),
            GBMResid("gbm_resid_aug", augment=True),
            GBMQuantile(), RidgePhys(),
            Hurdle("hurdle"), Hurdle("hurdle_lrp", mode="lrp"),
            Hurdle("hurdle_lrp_serving", mode="lrp", track="serving"),
            GBMResid("serving_resid", track="serving", resid_clip=SERVING_RESID_CLIP),
            # Round 2: the hurdle classifier on VIGIL-E's production features
            # (its P(final >= -6) is the VIGIL-E primary e_hurdle_eserving),
            # and the same with near-miss labels (exploratory)
            Hurdle("hurdle_eserving", track="eserving"),
            Hurdle("hurdle_eserving_aux", track="eserving", aux=True)]


# Candidates derived from a fitted pool member without refitting:
# name -> (base candidate, prevalence its decision is tuned for)
DERIVED = {"hurdle_lrp_serving_bench": ("hurdle_lrp_serving", BENCH_HR_PREVALENCE)}


def early_warning_pool() -> list:
    return [ELogit(), EColumn("e_max", "h_risk_max"),
            EColumn("e_physics", "p_proj_risk_trend"),
            EClassifier("e_clf"), EClassifier("e_serving", track="eserving")]
