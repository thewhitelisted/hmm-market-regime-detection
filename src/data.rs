use serde::{Deserialize, Serialize};
use ndarray::Array1;
#[derive(Debug, Deserialize, Serialize)]
struct MarketData {
    close: f64,
    date: String,
}


// Save quotes to CSV
// pub fn save_quotes(file_path: &str, quotes: &[yahoo_finance_api::Quote]) -> Result<(), Box<dyn Error>> {
//     let mut wtr = csv::Writer::from_path(file_path)?;

//     for quote in quotes {
//         wtr.serialize(MarketData {
//             close: quote.close,
//             date: Utc.timestamp_opt(quote.timestamp as i64, 0).single().unwrap().format("%Y-%m-%d").to_string(),
            
//         })?;
//     }

//     wtr.flush()?;
//     Ok(())
// }

// pub fn load_data(file_path: &str) -> Result<Vec<f64>, Box<dyn Error>> {
//     let mut rdr = ReaderBuilder::new().has_headers(true).from_path(file_path)?;
//     let mut prices = Vec::new();

//     for result in rdr.deserialize() {
//         let record: MarketData = result?;
//         prices.push(record.close);
//     }

//     Ok(prices)
// }

/// Compute log returns from price data
pub fn compute_log_returns(prices: &[f64]) -> Array1<f64> {
    let mut returns = Vec::new();
    for i in 1..prices.len() {
        let r = (prices[i] / prices[i - 1]).ln();
        returns.push(r);
    }
    Array1::from(returns)
}

/// Discretize returns into quantiles
pub fn discretize_returns_quantiles(returns: &[f64], num_bins: usize) -> Vec<usize> {
    let mut sorted_returns = returns.to_vec();
    sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let mut thresholds = Vec::with_capacity(num_bins - 1);
    for i in 1..num_bins {
        let idx = i * sorted_returns.len() / num_bins;
        thresholds.push(sorted_returns[idx]);
    }
    
    returns.iter().map(|&r| {
        thresholds.iter().position(|&t| r <= t).unwrap_or(num_bins - 1)
    }).collect()
}