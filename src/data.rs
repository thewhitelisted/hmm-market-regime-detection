use std::error::Error;
use csv::ReaderBuilder;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct MarketData {
    date: String,
    close: f64,
}

pub fn load_data(file_path: &str) -> Result<Vec<f64>, Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(file_path)?;
    let mut prices = Vec::new();

    for result in rdr.deserialize() {
        let record: MarketData = result?;
        prices.push(record.close);
    }

    Ok(prices)
}

// Compute log returns
pub fn compute_log_returns(prices: &[f64]) -> Vec<f64> {
    prices.windows(2).map(|w| (w[1] / w[0]).ln()).collect()
}
