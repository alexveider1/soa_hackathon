import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

BASE_NAMES = ("lgbm", "catboost")


class GenreStackingClassifier:
    def __init__(
        self,
        features: Sequence[str],
        class_names: Sequence[str],
        seed: int = 42,
        gpu: bool = True,
        lgbm_device: str = "gpu",
        catboost_params: dict | None = None,
        lgbm_params: dict | None = None,
        meta_C: float = 1.0,
        verbose: bool = True,
    ) -> None:
        self.features = list(features)
        self.class_names = list(class_names)
        self.n_classes = len(self.class_names)
        self.seed = seed
        self.gpu = gpu
        self.lgbm_device = lgbm_device
        self.verbose = verbose
        self.meta_C = meta_C

        self.catboost_params = catboost_params or self._default_catboost()
        self.lgbm_params = lgbm_params or self._default_lgbm()

        self.bases_: dict[str, object] = {}
        self.meta_: LogisticRegression | None = None
        self.log_prior_shift_: np.ndarray = np.zeros(self.n_classes)
        self.val_metrics_: dict = {}
        self.fit_seconds_: dict = {}

    def _default_catboost(self) -> dict:
        return dict(
            iterations=800,
            depth=6,
            learning_rate=0.08,
            l2_leaf_reg=4.0,
            bagging_temperature=0.5,
            border_count=64,
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
            auto_class_weights="Balanced",
            task_type="GPU" if self.gpu else "CPU",
            devices="0" if self.gpu else None,
            random_seed=self.seed,
            early_stopping_rounds=60,
            verbose=200 if self.verbose else False,
        )

    def _default_lgbm(self) -> dict:
        return dict(
            objective="multiclass",
            num_class=self.n_classes,
            class_weight="balanced",
            n_estimators=600,
            learning_rate=0.05,
            num_leaves=255,
            min_data_in_leaf=20,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            reg_lambda=1.0,
            device=self.lgbm_device,
            seed=self.seed,
            n_jobs=-1,
            verbose=-1,
        )

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
    ) -> "GenreStackingClassifier":
        Xtr = X_train[self.features]
        Xvl = X_val[self.features]
        train_pool = Pool(Xtr, label=y_train)
        val_pool = Pool(Xvl, label=y_val)
        val_probs: dict[str, np.ndarray] = {}

        self._log("[lgbm] fitting")
        t0 = time.time()
        params = dict(self.lgbm_params)
        try:
            lgbm = LGBMClassifier(**params)
            lgbm.fit(
                Xtr,
                y_train,
                eval_set=[(Xvl, y_val)],
                eval_metric="multi_logloss",
                callbacks=[early_stopping(50, verbose=self.verbose), log_evaluation(0)],
            )
        except Exception as exc:
            self._log(f"[lgbm] gpu failed: {exc!r}; retrying on CPU")
            params["device"] = "cpu"
            lgbm = LGBMClassifier(**params)
            lgbm.fit(
                Xtr,
                y_train,
                eval_set=[(Xvl, y_val)],
                eval_metric="multi_logloss",
                callbacks=[early_stopping(50, verbose=self.verbose), log_evaluation(0)],
            )
        self.bases_["lgbm"] = lgbm
        val_probs["lgbm"] = lgbm.predict_proba(Xvl)
        self.fit_seconds_["lgbm"] = time.time() - t0
        self._log(f"[lgbm] done in {self.fit_seconds_['lgbm']:.1f}s")

        self._log("[catboost] fitting")
        t0 = time.time()
        cb = CatBoostClassifier(**self._strip_none(self.catboost_params))
        cb.fit(train_pool, eval_set=val_pool, use_best_model=True)
        self.bases_["catboost"] = cb
        val_probs["catboost"] = cb.predict_proba(Xvl)
        self.fit_seconds_["catboost"] = time.time() - t0
        self._log(f"[catboost] done in {self.fit_seconds_['catboost']:.1f}s")

        self._log("[meta] fitting logistic on val probabilities")
        t0 = time.time()
        Z_val = self._stack(val_probs)
        meta = LogisticRegression(
            C=self.meta_C,
            solver="lbfgs",
            max_iter=2000,
            class_weight="balanced",
            random_state=self.seed,
        )
        meta.fit(Z_val, y_val)
        self.meta_ = meta
        self.fit_seconds_["meta"] = time.time() - t0
        self._log(f"[meta] done in {self.fit_seconds_['meta']:.1f}s")

        self._log("[calibrate] per-class log-prior shift")
        t0 = time.time()
        proba_val = meta.predict_proba(Z_val)
        self.log_prior_shift_ = self._fit_prior_shift(proba_val, y_val)
        self.fit_seconds_["calibrate"] = time.time() - t0

        for name, p in val_probs.items():
            self.val_metrics_[name] = self._metrics(y_val, p)
        self.val_metrics_["meta_raw"] = self._metrics(y_val, proba_val)
        self.val_metrics_["meta_calibrated"] = self._metrics(
            y_val, self._apply_shift(proba_val)
        )
        return self

    def _stack(self, probs: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate([probs[name] for name in BASE_NAMES], axis=1)

    def base_probs(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        Xb = X[self.features]
        return {name: self.bases_[name].predict_proba(Xb) for name in BASE_NAMES}

    def predict_proba(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        if self.meta_ is None:
            raise RuntimeError("call fit() first")
        Z = self._stack(self.base_probs(X))
        proba = self.meta_.predict_proba(Z)
        return self._apply_shift(proba) if calibrated else proba

    def predict(self, X: pd.DataFrame, calibrated: bool = True) -> np.ndarray:
        return np.argmax(self.predict_proba(X, calibrated=calibrated), axis=1)

    def _apply_shift(self, proba: np.ndarray) -> np.ndarray:
        eps = 1e-12
        logits = np.log(np.clip(proba, eps, 1.0)) + self.log_prior_shift_
        logits -= logits.max(axis=1, keepdims=True)
        e = np.exp(logits)
        return e / e.sum(axis=1, keepdims=True)

    def _fit_prior_shift(
        self,
        proba: np.ndarray,
        y: np.ndarray,
        n_passes: int = 2,
        grid: np.ndarray | None = None,
    ) -> np.ndarray:
        if grid is None:
            grid = np.linspace(-2.0, 2.0, 21)
        eps = 1e-12
        logits = np.log(np.clip(proba, eps, 1.0))
        shift = np.zeros(self.n_classes)

        def macro_f1(shift_vec: np.ndarray) -> float:
            pred = np.argmax(logits + shift_vec, axis=1)
            return f1_score(y, pred, average="macro", zero_division=0)

        best = macro_f1(shift)
        order = np.argsort(np.bincount(y, minlength=self.n_classes))[::-1]
        for _ in range(n_passes):
            for k in order:
                base = shift[k]
                best_k, best_v = base, best
                for delta in grid:
                    shift[k] = base + delta
                    v = macro_f1(shift)
                    if v > best_v:
                        best_v = v
                        best_k = shift[k]
                shift[k] = best_k
                best = best_v
            self._log(f"[calibrate] macro_f1 after pass: {best:.4f}")
        return shift

    def shap_values_base(self, X: pd.DataFrame, base_name: str) -> np.ndarray:
        if base_name not in self.bases_:
            raise KeyError(f"unknown base {base_name!r}; have {list(self.bases_)}")
        Xb = X[self.features]
        model = self.bases_[base_name]
        if base_name == "catboost":
            raw = np.array(
                model.get_feature_importance(type="ShapValues", data=Pool(Xb))
            )
            return raw[:, :, :-1]
        booster = model.booster_ if hasattr(model, "booster_") else model
        raw = booster.predict(Xb, pred_contrib=True)
        sv = np.asarray(raw).reshape(len(Xb), self.n_classes, len(self.features) + 1)
        return sv[:, :, :-1]

    def meta_weights(self) -> pd.DataFrame:
        if self.meta_ is None:
            raise RuntimeError("call fit() first")
        coef = self.meta_.coef_.reshape(self.n_classes, len(BASE_NAMES), self.n_classes)
        influence = np.abs(coef).sum(axis=2)
        return pd.DataFrame(influence, columns=list(BASE_NAMES), index=self.class_names)

    def shap_values_ensemble(
        self, X: pd.DataFrame, weight_by_meta: bool = True
    ) -> np.ndarray:
        per_base = {n: self.shap_values_base(X, n) for n in BASE_NAMES}
        if not weight_by_meta:
            return np.mean(list(per_base.values()), axis=0)
        w = self.meta_weights().to_numpy()
        w = w / w.sum(axis=1, keepdims=True).clip(min=1e-9)
        out = np.zeros_like(per_base[BASE_NAMES[0]])
        for bi, name in enumerate(BASE_NAMES):
            out += per_base[name] * w[:, bi][None, :, None]
        return out

    def save(self, dir_path: str | Path) -> None:
        if self.meta_ is None:
            raise RuntimeError("nothing to save; call fit() first")
        import joblib

        out = Path(dir_path)
        out.mkdir(parents=True, exist_ok=True)
        self.bases_["catboost"].save_model(str(out / "catboost.cbm"))
        joblib.dump(self.bases_["lgbm"], out / "lgbm.joblib")
        with open(out / "meta.json", "w") as f:
            json.dump(
                {
                    "coef": self.meta_.coef_.tolist(),
                    "intercept": self.meta_.intercept_.tolist(),
                    "classes": self.meta_.classes_.tolist(),
                },
                f,
            )
        with open(out / "bundle.json", "w") as f:
            json.dump(
                {
                    "features": self.features,
                    "class_names": self.class_names,
                    "log_prior_shift": self.log_prior_shift_.tolist(),
                    "val_metrics": self.val_metrics_,
                    "fit_seconds": self.fit_seconds_,
                    "seed": self.seed,
                },
                f,
                indent=2,
                default=float,
            )

    @classmethod
    def load(cls, dir_path: str | Path) -> "GenreStackingClassifier":
        import joblib

        src = Path(dir_path)
        with open(src / "bundle.json") as f:
            bundle = json.load(f)
        with open(src / "meta.json") as f:
            meta_payload = json.load(f)
        obj = cls(
            features=bundle["features"],
            class_names=bundle["class_names"],
            seed=bundle["seed"],
            verbose=False,
        )
        cb = CatBoostClassifier()
        cb.load_model(str(src / "catboost.cbm"))
        lgbm = joblib.load(src / "lgbm.joblib")
        obj.bases_ = {"lgbm": lgbm, "catboost": cb}
        meta = LogisticRegression()
        meta.coef_ = np.array(meta_payload["coef"])
        meta.intercept_ = np.array(meta_payload["intercept"])
        meta.classes_ = np.array(meta_payload["classes"])
        obj.meta_ = meta
        obj.log_prior_shift_ = np.array(bundle["log_prior_shift"])
        obj.val_metrics_ = bundle["val_metrics"]
        obj.fit_seconds_ = bundle["fit_seconds"]
        return obj

    def _metrics(self, y_true: np.ndarray, proba: np.ndarray) -> dict:
        pred = np.argmax(proba, axis=1)
        return {
            "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
            "weighted_f1": float(
                f1_score(y_true, pred, average="weighted", zero_division=0)
            ),
            "top1_acc": float((pred == y_true).mean()),
        }

    def _strip_none(self, params: dict) -> dict:
        return {k: v for k, v in params.items() if v is not None}

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg, flush=True)
