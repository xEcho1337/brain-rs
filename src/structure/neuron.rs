use crate::structure::Neuron;

impl Neuron {
    pub fn new() -> Self {
        Self { synapses: vec![], value: 0.0, delta: 0.0, bias: 0.0 }
    }
}