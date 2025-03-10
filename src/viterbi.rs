use ndarray::{Array2, Array1};
use crate::hmm::HMM;

pub fn viterbi(observations: &[usize], hmm: &HMM) -> Vec<usize> {
    let num_states = hmm.num_states();
    let num_obs = observations.len();
    
    let mut dp = Array2::<f64>::zeros((num_obs, num_states));
    let mut backpointer = Array2::<usize>::zeros((num_obs, num_states));

    // Initialization
    for state in 0..num_states {
        dp[[0, state]] = hmm.initial()[state] * hmm.emission()[[state, observations[0]]];
    }

    // Recursion
    for t in 1..num_obs {
        for j in 0..num_states {
            let (max_prob, prev_state) = (0..num_states)
                .map(|i| (dp[[t-1, i]] * hmm.transition()[[i, j]] * hmm.emission()[[j, observations[t]]], i))
                .max_by(|x, y| x.0.partial_cmp(&y.0).unwrap())
                .unwrap();
            
            dp[[t, j]] = max_prob;
            backpointer[[t, j]] = prev_state;
        }
    }

    // Backtracking
    let mut best_sequence = vec![0; num_obs];
    best_sequence[num_obs - 1] = dp.row(num_obs - 1).indexed_iter().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
    for t in (0..num_obs - 1).rev() {
        best_sequence[t] = backpointer[[t + 1, best_sequence[t + 1]]];
    }

    best_sequence
}
