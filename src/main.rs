use plotters::prelude::*;
use yahoo_finance_api as yahoo;
mod hmm;
mod data;


#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load data
    let provider = yahoo::YahooConnector::new()?;
    use time::OffsetDateTime;
    let start_date = OffsetDateTime::parse("2015-01-01T00:00:00Z", &time::format_description::well_known::Rfc3339).unwrap();
    let end_date = OffsetDateTime::parse("2025-01-01T00:00:00Z", &time::format_description::well_known::Rfc3339).unwrap();
    let response = provider.get_quote_history("SPY", start_date, end_date).await?;
    let quotes = response.quotes().unwrap();

    // discretize returns
    let prices = quotes.iter().map(|q| q.close).collect::<Vec<f64>>();
    let returns = data::compute_log_returns(&prices);
    let observations = data::discretize_returns_quantiles(returns.as_slice().unwrap(), 3).to_vec();

    // Train HMM
    let mut hmm = hmm::HMM::new(3, 3);
    println!("Initial transition matrix: {:?}", hmm.transition_matrix);
    println!("Initial emission matrix: {:?}", hmm.emission_matrix);
    hmm.train(observations.as_slice(), 900);
    println!("Transition matrix: {:?}", hmm.transition_matrix);
    println!("Emission matrix: {:?}", hmm.emission_matrix);
    let regimes = hmm.predict(&observations.as_slice());
    println!("Regimes: {:?}", regimes);
    

    // Plot
    let root = BitMapBackend::new("regimes.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root)
        .caption("SPY Regimes", ("sans-serif", 40))
        .margin(10)
        .x_label_area_size(20)
        .y_label_area_size(40)
        .build_cartesian_2d(0..prices.len(), 150.0..625.0)
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    // Plot price line
    chart
        .draw_series(LineSeries::new(
            prices.iter().enumerate().map(|(i, &p)| (i, p)),
            &BLUE,
        ))
        .unwrap();

    // Plot regime points
    for (i, &regime) in regimes.iter().enumerate() {
        let color = match regime {
            0 => RED,
            1 => GREEN,
            2 => BLUE,
            _ => BLACK,
        };
        chart
            .draw_series(PointSeries::of_element(
                vec![(i, prices[i])],
                3,
                color.filled(),
                &|coord, size, style| {
                    Circle::new(coord, size, style)
                },
            ))
            .unwrap();
    }

    root.present().unwrap();
    Ok(())
}