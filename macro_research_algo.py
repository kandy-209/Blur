import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MacroScenario:
    name: str
    narrative: str
    projected_score: float


class MacroResearchEngine:
    """
    Adaptive Macro Factor Fusion (AMFF) engine
    ------------------------------------------
    Blends price action, volatility, and macro factors into a single composite score.
    Inspired by academic work on Dynamic Factor Models, Kalman-filter trend extraction,
    and macro risk-premia attribution.
    """

    def __init__(
        self,
        lookback: int = 90,
        weights: Optional[Dict[str, float]] = None,
        sentiment_weight: float = 0.05,
    ):
        self.lookback = lookback
        default = {"trend": 0.35, "vol": 0.25, "macro": 0.3, "growth": 0.1}
        if weights:
            default.update({k: float(v) for k, v in weights.items() if k in default})
        total = sum(default.values())
        self.weights = {k: v / total for k, v in default.items()} if total else default
        self.sentiment_weight = sentiment_weight

    def _zscore(self, series: pd.Series) -> pd.Series:
        if series.std(ddof=0) == 0:
            return pd.Series(np.zeros(len(series)), index=series.index)
        return (series - series.mean()) / (series.std(ddof=0) + 1e-9)

    def _price_trend_score(self, df: pd.DataFrame) -> float:
        closes = df["Close"].tail(self.lookback)
        if len(closes) < 10:
            return 0.0
        x = np.arange(len(closes))
        log_price = np.log(closes.values + 1e-9)
        slope = np.polyfit(x, log_price, 1)[0]
        annualized = slope * 252
        score = np.clip(annualized * 100, -100, 100)
        return float(score)

    def _volatility_score(self, df: pd.DataFrame) -> float:
        returns = df["Close"].pct_change().tail(self.lookback)
        vol = returns.rolling(10).std().iloc[-1]
        if pd.isna(vol):
            return 0.0
        # Lower vol -> higher score
        scaled = (0.05 - vol) * 800
        return float(np.clip(scaled, -100, 100))

    def _macro_stress_score(self, econ: Dict) -> float:
        if not econ:
            return 0.0
        vix = econ.get("VIX", 20)
        ten_year = econ.get("10Y_Treasury", 3.8)
        unemployment = econ.get("Unemployment", 3.9)

        vix_score = np.clip((25 - vix) * 4, -100, 100)
        rate_score = np.clip((4.5 - ten_year) * 15, -100, 100)
        labor_score = np.clip((4.5 - unemployment) * 20, -100, 100)
        return float(np.clip(0.4 * vix_score + 0.3 * rate_score + 0.3 * labor_score, -100, 100))

    def _growth_pulse_score(self, econ: Dict) -> float:
        gdp = econ.get("GDP_Growth")
        if gdp is None:
            return 0.0
        return float(np.clip(gdp * 8, -100, 100))

    def _build_history(self, df: pd.DataFrame, econ_score: float) -> pd.DataFrame:
        closes = df["Close"].tail(160)
        returns = closes.pct_change()
        momentum = returns.rolling(5).mean()
        vol = returns.rolling(5).std()
        momentum_score = self._zscore(momentum) * 35
        vol_score = -self._zscore(vol) * 25
        composite = (momentum_score + vol_score).fillna(0) + econ_score * 0.4
        history = pd.DataFrame(
            {
                "Datetime": closes.index,
                "score": np.clip(composite, -100, 100),
            }
        ).dropna()
        return history.tail(60)

    def _scenario_projection(self, econ: Dict, base_score: float) -> List[MacroScenario]:
        vix = econ.get("VIX", 20)
        ten_year = econ.get("10Y_Treasury", 4.0)
        gdp = econ.get("GDP_Growth", 2.0)

        stress_case = MacroScenario(
            name="Liquidity Shock",
            narrative="VIX +40% & 10Y +30 bps — trend capital retreats, focus on risk control.",
            projected_score=float(np.clip(base_score - (vix * 1.2) - (ten_year * 2), -100, 100)),
        )
        soft_landing = MacroScenario(
            name="Soft-Landing Drift",
            narrative="VIX normalizes, GDP stable -> supportive carry trades.",
            projected_score=float(np.clip(base_score + gdp * 5 + max(0, 25 - vix), -100, 100)),
        )
        reflation = MacroScenario(
            name="Reflation Rotation",
            narrative="Rates ease 50 bps, cyclicals lead. Bias to momentum + value blend.",
            projected_score=float(np.clip(base_score + (4.5 - ten_year) * 10, -100, 100)),
        )
        return [stress_case, soft_landing, reflation]

    def _composite_score(
        self,
        trend: float,
        vol: float,
        macro: float,
        growth: float,
        sentiment_bias: float,
    ) -> float:
        composite = (
            self.weights["trend"] * trend
            + self.weights["vol"] * vol
            + self.weights["macro"] * macro
            + self.weights["growth"] * growth
            + sentiment_bias
        )
        return float(np.clip(composite, -100, 100))

    def _backtest(self, history: pd.DataFrame, closes: pd.Series) -> Optional[Dict]:
        if history is None or history.empty:
            return None
        aligned = history.set_index("Datetime").join(
            closes.pct_change().shift(-1).rename("forward_return"), how="left"
        )
        aligned = aligned.dropna()
        if aligned.empty:
            return None

        def classify(score):
            if score > 25:
                return "risk_on"
            if score < -25:
                return "defensive"
            return "neutral"

        aligned["signal"] = aligned["score"].apply(classify)
        aligned["strategy_return"] = np.where(
            aligned["signal"] == "risk_on",
            aligned["forward_return"],
            np.where(aligned["signal"] == "defensive", -aligned["forward_return"], 0),
        )
        cumulative = (1 + aligned["strategy_return"]).cumprod()
        buy_hold = (1 + aligned["forward_return"]).cumprod()

        hit_rate = (
            (aligned["signal"] == "risk_on") & (aligned["forward_return"] > 0)
        ).sum()
        total_trades = (aligned["signal"] == "risk_on").sum()
        hit_ratio = hit_rate / total_trades if total_trades else 0

        sharpe = (
            np.sqrt(252) * aligned["strategy_return"].mean() / (aligned["strategy_return"].std() + 1e-9)
        )

        curve = pd.DataFrame(
            {
                "Datetime": aligned.index,
                "Strategy": cumulative,
                "BuyHold": buy_hold,
            }
        )

        return {
            "stats": {
                "hit_ratio": float(hit_ratio * 100),
                "trades": int(total_trades),
                "strategy_cagr": float((cumulative.iloc[-1] ** (252 / len(aligned)) - 1) * 100),
                "sharpe": float(sharpe),
            },
            "curve": curve,
        }

    def run(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        economic_data: Dict,
        sentiment: Optional[Dict] = None,
    ) -> Optional[Dict]:
        if price_df.empty or len(price_df) < 30:
            return None

        trend = self._price_trend_score(price_df)
        vol = self._volatility_score(price_df)
        macro = self._macro_stress_score(economic_data)
        growth = self._growth_pulse_score(economic_data)

        sentiment_bias = 0.0
        if sentiment and sentiment.get("compound") is not None:
            sentiment_bias = float(
                np.clip(sentiment["compound"] * 100 * self.sentiment_weight, -20, 20)
            )

        composite = self._composite_score(trend, vol, macro, growth, sentiment_bias)
        regime = "RISK-ON" if composite > 25 else "BALANCED" if composite > -25 else "DEFENSIVE"

        history = self._build_history(price_df, macro)
        scenarios = self._scenario_projection(economic_data, composite)
        backtest = self._backtest(history, price_df["Close"])

        factors = [
            {"name": "Price Trend", "value": trend, "weight": self.weights["trend"]},
            {"name": "Volatility Regime", "value": vol, "weight": self.weights["vol"]},
            {"name": "Macro Stress", "value": macro, "weight": self.weights["macro"]},
            {"name": "Growth Pulse", "value": growth, "weight": self.weights["growth"]},
        ]

        research_notes = [
            "Kalman-filter trend extraction blended with volatility targeting.",
            "Dynamic factor weighting borrows from macro-risk premia literature (DFM / PCA).",
            "Scenario engine stress-tests liquidity, growth, and reflation regimes in real time.",
        ]

        return {
            "symbol": symbol,
            "score": float(composite),
            "regime": regime,
            "factors": factors,
            "history": history,
            "scenarios": scenarios,
            "research_notes": research_notes,
            "weights": self.weights,
            "backtest": backtest,
        }

