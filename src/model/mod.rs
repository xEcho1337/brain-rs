use crate::initialization::WeightInitialization;
use crate::layer::Layer;
use crate::loss::LossFunctions;
use crate::optimizer::Optimizer;

mod model;

pub struct Model<'a> {
    pub layers: Vec<&'a dyn Layer>,
    pub init_technique: Option<WeightInitialization>,
    pub loss_functions: Option<LossFunctions>,
    pub optimizer: Option<&'a dyn Optimizer>
}