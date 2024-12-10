use crate::structure::Synapse;
use crate::optimizer::Optimizer;

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    first_momentum: Vec<f64>,
    second_momentum: Vec<f64>,
    beta1_timestep: f64,
    beta2_timestep: f64,
    timestep: usize,
}

impl Optimizer for Adam {
    fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            first_momentum: vec![0.0; 0],
            second_momentum: vec![0.0; 0],
            beta1_timestep: 1.0,
            beta2_timestep: 1.0,
            timestep: 0,
        }
    }

    fn post_initialize(&mut self) {
        let capacity: usize = 1000;
        self.first_momentum = vec![0.0; capacity];
        self.second_momentum = vec![0.0; capacity];
    }

    fn get_learning_rate(&self) -> f64 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }

    fn update(&mut self, synapse: &mut Synapse) {
        let gradient = synapse.output_neuron.delta * synapse.input_neuron.value;
        let synapse_id = synapse.id;

        // TODO: Remove this unnecessary piece
        if synapse_id >= self.first_momentum.len() {
            self.first_momentum.resize(synapse_id + 1, 0.0);
            self.second_momentum.resize(synapse_id + 1, 0.0);
        }

        let current_first_momentum: f64 = self.first_momentum[synapse_id];
        let current_second_momentum: f64 = self.second_momentum[synapse_id];

        let m: f64 = self.beta1 * current_first_momentum + (1.0 - self.beta1) * gradient;
        let v: f64 = self.beta2 * current_second_momentum + (1.0 - self.beta2) * gradient.powi(2);

        self.first_momentum[synapse_id] = m;
        self.second_momentum[synapse_id] = v;

        let m_hat = m / self.beta1_timestep;
        let v_hat = v / self.beta2_timestep;

        let delta_weight = (self.learning_rate * m_hat) / (v_hat.sqrt() + self.epsilon);

        synapse.weight += delta_weight;
    }

    fn post_iteration(&mut self, _layers: &[crate::layer::DenseLayer]) {
        self.timestep += 1;

        self.beta1_timestep = 1.0 - self.beta1.powi(self.timestep as i32 + 1);
        self.beta2_timestep = 1.0 - self.beta2.powi(self.timestep as i32 + 1);

        // TODO: Update all synapses
    }
}