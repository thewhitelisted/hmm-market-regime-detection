pub fn generate_trading_signals(regimes: &[usize]) -> Vec<i32> {
    regimes.iter().map(|&state| match state {
        0 => 1,   // Bull (Buy)
        1 => -1,  // Bear (Sell)
        _ => 0,   // Sideways (Hold)
    }).collect()
}
