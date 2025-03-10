pub fn backtest(strategy_returns: &[f64]) -> f64 {
    strategy_returns.iter().sum()  // Simplified return computation
}
