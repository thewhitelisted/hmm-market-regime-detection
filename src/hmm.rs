use rand::Rng;
use ndarray::{Array2, Array1};

pub struct HMM {
    num_states: usize,
    transition: Array2<f64>,  // Transition probabilities
    emission: Array2<f64>,    // Emission probabilities
    initial: Array1<f64>,     // Initial state probabilities
}

impl HMM {
    pub fn new(num_states: usize) -> Self {
        let mut rng = rand::thread_rng();
        let transition = Array2::from_shape_fn((num_states, num_states), |_| rng.gen::<f64>());
        let emission = Array2::from_shape_fn((num_states, num_states), |_| rng.gen::<f64>());
        let initial = Array1::from_shape_fn(num_states, |_| rng.gen::<f64>());
        
        // Normalize
        let transition = transition.mapv(|x| x / transition.sum());
        let emission = emission.mapv(|x| x / emission.sum());
        let initial = initial.mapv(|x| x / initial.sum());

        HMM { num_states, transition, emission, initial }
    }

    fn forward(&self, observations: &[usize]) -> Array2<f64> {
        let num_observations = observations.len();
        let mut alpha = Array2::<f64>::zeros((num_observations, self.num_states));

        // Initialize alpha
        for state in 0..self.num_states {
            alpha[[0, state]] = self.initial[state] * self.emission[[state, observations[0]]];
        }

        // Compute alpha for each time step
        for t in 1..num_observations {
            for j in 0..self.num_states {
            let mut sum = 0.0;
            for i in 0..self.num_states {
                sum += alpha[[t - 1, i]] * self.transition[[i, j]];
            }
            alpha[[t, j]] = sum * self.emission[[j, observations[t]]];
            }
        }

        return alpha;
    }

    fn backward(&self, observations: &[usize]) -> Array2<f64> {
        let num_observations = observations.len();
        let mut beta = Array2::<f64>::zeros((num_observations, self.num_states));

        // Initialize beta
        for state in 0..self.num_states {
            beta[[num_observations - 1, state]] = 1.0;
        }

        // Compute beta for each time step
        for t in (0..num_observations - 1).rev() {
            for i in 0..self.num_states {
                let mut sum = 0.0;
                for j in 0..self.num_states {
                    sum += beta[[t + 1, j]] * self.transition[[i, j]] * self.emission[[j, observations[t + 1]]];
                }
                beta[[t, i]] = sum;
            }
        }

        return beta;
    }

    pub fn train(&mut self, observations: &[usize], iterations: usize) {
        
    }

    // getter methods
    pub fn num_states(&self) -> usize {
        self.num_states
    }

    pub fn transition(&self) -> &Array2<f64> {
        &self.transition
    }

    pub fn emission(&self) -> &Array2<f64> {
        &self.emission
    }

    pub fn initial(&self) -> &Array1<f64> {
        &self.initial
    }
}
