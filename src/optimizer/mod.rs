pub mod optimizer {
    use crate::layer::Layer;
    use crate::structure::{Neuron, Synapse};

    const GRADIENT_CLIP: f64 = 5.;

    pub trait Optimizer {
        fn new(learning_rate: f64) -> Self where Self: Sized;

        fn post_initialize(&mut self) {}

        fn get_learning_rate(&self) -> f64;

        fn set_learning_rate(&mut self, learning_rate: f64);

        fn update(&mut self, synapse: &mut Synapse);

        fn post_iteration(&mut self, layers: &[Layer]);

        fn post_fit(&mut self, layers: &[Layer]) {}

        fn apply_gradient_step(&mut self, layer: &Layer, neuron: &mut Neuron, synapse: &mut Synapse) {
            let output = neuron.get_value();

            let error = Self::clip_gradient(synapse.get_weight() * synapse.get_output_neuron().get_delta());
            let delta = Self::clip_gradient(error * layer.get_activation().get_function().get_derivative(output));

            let weight_change = Self::clip_gradient(delta * synapse.get_input_neuron().get_value());

            neuron.set_delta(delta);
            synapse.set_weight(synapse.get_weight() + weight_change);
        }

        fn clip_gradient(gradient: f64) -> f64 {
            gradient.clamp(-GRADIENT_CLIP, GRADIENT_CLIP)
        }
    }
}
