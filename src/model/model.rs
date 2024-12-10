use crate::initialization::WeightInitialization;
use crate::layer::Layer;
use crate::loss::LossFunctions;
use crate::model::Model;

impl Model {
    pub fn new(layers: Vec<Box<dyn Layer>>) -> Self {
        Self {
            layers,
            init_technique: None,
            loss_functions: None
        }
    }

    pub fn compile(&mut self, init_technique: WeightInitialization, loss_functions: LossFunctions) {
        self.init_technique = Some(init_technique);
        self.loss_functions = Some(loss_functions);
    }
}