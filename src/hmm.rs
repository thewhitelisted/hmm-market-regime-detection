use ndarray::{Array1, Array2, Array3, s};

pub struct HMM {
    num_states: usize,
    num_observations: usize,
    pub transition_matrix: Array2<f64>,
    pub emission_matrix: Array2<f64>,
    pub initial_probabilities: Array1<f64>,
}

impl HMM {
    pub fn new(num_states: usize, num_observations: usize) -> Self {
        
        let transition_matrix = Array2::from_shape_vec(
            (3, 3),
            vec![
                0.8, 0.1, 0.1,  // State 0 (Bear) is persistent
                0.1, 0.8, 0.1,  // State 1 (Neutral) is persistent
                0.1, 0.1, 0.8,  // State 2 (Bull) is persistent
            ]
        ).unwrap();
        let emission_matrix = Array2::from_shape_vec(
            (3, num_observations),
            vec![
                0.6, 0.3, 0.1,  // Bear state: mostly negative returns
                0.2, 0.6, 0.2,  // Neutral state: mostly neutral returns
                0.1, 0.3, 0.6,  // Bull state: mostly positive returns
            ]
        ).unwrap();
        let initial_probabilities = Array1::from_vec(vec![1.0/3.0, 1.0/3.0, 1.0/3.0]);

        // Normalize to create valid probability distributions
        let mut hmm = HMM {
            num_states,
            num_observations,
            transition_matrix,
            emission_matrix,
            initial_probabilities,
        };
        hmm.normalize_parameters();
        hmm
    }

    pub fn train(&mut self, observations: &[usize], iterations: usize) {
        if observations.is_empty() {
            return;
        }

        for _ in 0..iterations {
            // E-step: Compute scaled forward and backward variables
            let (alpha, scales) = self.scaled_forward(observations);
            let beta = self.scaled_backward(observations, &scales);

            // Compute posterior probabilities
            let (gamma, xi) = self.compute_gamma_xi(&alpha, &beta, observations, &scales);

            // M-step: Update parameters
            self.update_initial_probabilities(&gamma);
            self.update_transition_matrix(&gamma, &xi);
            self.update_emission_matrix(&gamma, observations);
            
            // Ensure valid probability distributions
            self.normalize_parameters();
        }
    }

    // Implementation of scaled forward algorithm with normalization
    fn scaled_forward(&self, observations: &[usize]) -> (Array2<f64>, Vec<f64>) {
        let n = observations.len();
        let mut alpha = Array2::zeros((n, self.num_states));
        let mut scales = vec![0.0; n];

        // Initialize first time step
        for i in 0..self.num_states {
            alpha[[0, i]] = self.initial_probabilities[i] * 
                           self.emission_matrix[[i, observations[0]]];
        }
        scales[0] = alpha.row(0).sum();
        if scales[0] > 0.0 {
            alpha.row_mut(0).mapv_inplace(|x| x / scales[0]);
        }

        // Recursion for subsequent steps
        for t in 1..n {
            for j in 0..self.num_states {
                let sum: f64 = (0..self.num_states)
                    .map(|i| alpha[[t-1, i]] * self.transition_matrix[[i, j]])
                    .sum();
                alpha[[t, j]] = sum * self.emission_matrix[[j, observations[t]]];
            }
            scales[t] = alpha.row(t).sum();
            if scales[t] > 0.0 {
                alpha.row_mut(t).mapv_inplace(|x| x / scales[t]);
            }
        }

        (alpha, scales)
    }

    // Implementation of scaled backward algorithm
    fn scaled_backward(&self, observations: &[usize], scales: &[f64]) -> Array2<f64> {
        let n = observations.len();
        let mut beta = Array2::ones((n, self.num_states));

        // Start from the end of the sequence
        let last_scale = scales[n-1];
        if last_scale > 0.0 {
            beta.row_mut(n-1).mapv_inplace(|x| x / last_scale);
        }

        // Work backwards through the sequence
        for t in (0..n-1).rev() {
            for i in 0..self.num_states {
                let mut sum = 0.0;
                for j in 0..self.num_states {
                    sum += self.transition_matrix[[i, j]] *
                           self.emission_matrix[[j, observations[t+1]]] *
                           beta[[t+1, j]];
                }
                beta[[t, i]] = sum / scales[t];
            }
        }

        beta
    }

    // Compute gamma (state probabilities) and xi (transition probabilities)
    fn compute_gamma_xi(
        &self,
        alpha: &Array2<f64>,
        beta: &Array2<f64>,
        observations: &[usize],
        scales: &[f64],
    ) -> (Array2<f64>, Array3<f64>) {
        let n = observations.len();
        let mut gamma = Array2::zeros((n, self.num_states));
        let mut xi = Array3::zeros((n-1, self.num_states, self.num_states));

        // Compute gamma values
        for t in 0..n {
            for i in 0..self.num_states {
                gamma[[t, i]] = alpha[[t, i]] * beta[[t, i]];
            }
        }

        // Compute xi values
        for t in 0..n-1 {
            let obs = observations[t+1];
            for i in 0..self.num_states {
                for j in 0..self.num_states {
                    xi[[t, i, j]] = alpha[[t, i]] *
                                    self.transition_matrix[[i, j]] *
                                    self.emission_matrix[[j, obs]] *
                                    beta[[t+1, j]] /
                                    scales[t+1];
                }
            }
        }

        (gamma, xi)
    }

    // M-step: Update initial probabilities
    fn update_initial_probabilities(&mut self, gamma: &Array2<f64>) {
        self.initial_probabilities.assign(&gamma.row(0));
    }

    // M-step: Update transition matrix
    fn update_transition_matrix(&mut self, gamma: &Array2<f64>, xi: &Array3<f64>) {
        for i in 0..self.num_states {
            let denominator: f64 = gamma.slice(s![0..-1, i]).sum();
            if denominator == 0.0 {
                continue;
            }
            
            for j in 0..self.num_states {
                let numerator: f64 = xi.slice(s![.., i, j]).sum();
                self.transition_matrix[[i, j]] = numerator / denominator;
            }
        }
    }

    // M-step: Update emission matrix
    fn update_emission_matrix(&mut self, gamma: &Array2<f64>, observations: &[usize]) {
        for i in 0..self.num_states {
            let total = gamma.column(i).sum();
            if total == 0.0 {
                continue;
            }

            for k in 0..self.num_observations {
                let count: f64 = gamma.column(i)
                    .iter()
                    .zip(observations.iter())
                    .filter(|(_, &obs)| obs == k)
                    .map(|(&prob, _)| prob)
                    .sum();
                    
                self.emission_matrix[[i, k]] = count / total;
            }
        }
    }

    // Normalize all probability distributions
    fn normalize_parameters(&mut self) {
        // Normalize initial probabilities
        let init_sum = self.initial_probabilities.sum();
        if init_sum > 0.0 {
            self.initial_probabilities /= init_sum;
        }

        // Normalize transition matrix rows
        for mut row in self.transition_matrix.rows_mut() {
            let sum = row.sum();
            if sum > 0.0 {
                row /= sum;
            }
        }

        // Normalize emission matrix rows
        for mut row in self.emission_matrix.rows_mut() {
            let sum = row.sum();
            if sum > 0.0 {
                row /= sum;
            }
        }
    }

    pub fn predict(&self, observations: &[usize]) -> Vec<usize> {
        let n = observations.len();
        let mut viterbi = Array2::zeros((n, self.num_states));
        let mut backpointers = Array2::zeros((n, self.num_states));
    
        // Initialize first step with log probabilities
        for i in 0..self.num_states {
            let p = self.initial_probabilities[i] * self.emission_matrix[[i, observations[0]]];
            viterbi[[0, i]] = if p > 0.0 { p.ln() } else { std::f64::NEG_INFINITY };
        }
    
        // Recursion (using log probabilities)
        for t in 1..n {
            for j in 0..self.num_states {
                let (max_val, max_state) = (0..self.num_states)
                    .map(|i| {
                        let trans_prob = self.transition_matrix[[i, j]];
                        if viterbi[[t-1, i]] == std::f64::NEG_INFINITY || trans_prob == 0.0 {
                            (std::f64::NEG_INFINITY, i)
                        } else {
                            (viterbi[[t-1, i]] + trans_prob.ln(), i)
                        }
                    })
                    .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
                    .unwrap();
                
                let emit_prob = self.emission_matrix[[j, observations[t]]];
                viterbi[[t, j]] = if emit_prob > 0.0 && max_val != std::f64::NEG_INFINITY {
                    max_val + emit_prob.ln()
                } else {
                    std::f64::NEG_INFINITY
                };
                backpointers[[t, j]] = max_state as f64;
            }
        }
    
        // Backtrack to find most likely path
        let mut path = vec![0; n];
        path[n-1] = (0..self.num_states)
            .map(|i| (viterbi[[n-1, i]], i))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap().1;
    
        for t in (1..n).rev() {
            path[t-1] = backpointers[[t, path[t]]] as usize;
        }
    
        path
    }
}