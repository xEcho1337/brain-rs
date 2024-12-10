use rand::random;
use crate::structure::{Neuron, Synapse};

impl Synapse {
    pub fn new(input_neuron: Neuron, output_neuron: Neuron, bound: f64) -> Synapse {
        Synapse { input_neuron, output_neuron, weight: (random() * 2 * bound) - bound }
    }
}