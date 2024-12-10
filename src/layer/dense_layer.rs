use crate::activation::Activations;
use crate::layer::{DenseLayer, Layer};
use crate::structure::{Neuron, Synapse};

impl DenseLayer {
    pub fn new(input: i32, activation: Activations) {
        let mut neurons = Vec::with_capacity(input as usize);
        let synapses = vec![];

        for _ in 0..input {
            neurons.push(Neuron::new());
        }
        Self {
            neurons: Vec::with_capacity(input as usize),
            synapses: vec![],
            activation,
        };
    }
}

impl Layer for DenseLayer {
    fn apply_function(previous: impl Layer) {

    }
}