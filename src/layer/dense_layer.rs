use std::any::Any;

use crate::activation::Activation;
use crate::layer::{DenseLayer, Layer};
use crate::structure::Neuron;

impl DenseLayer {
    pub fn new(input: i32, activation: impl Activation) -> Self {
        let mut neurons = Vec::with_capacity(input as usize);

        for _ in 0..input {
            neurons.push(Neuron::new());
        }

        Self {
            neurons: Vec::with_capacity(input as usize),
            synapses: vec![],
            activation,
        }
    }
}

impl Layer for DenseLayer {
    /*
    fn get_neurons(&self) -> Vec<Neuron> {
        self.neurons
    }
    */

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn apply_function(&self, previous: Box<dyn Layer>) {

    }
}