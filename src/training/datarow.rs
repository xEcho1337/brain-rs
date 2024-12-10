use crate::training::DataRow;
use crate::utils::Vector;

impl DataRow {
    pub fn new(input: Vector, output: Vector) -> Self {
        Self { input, output }
    }
}
