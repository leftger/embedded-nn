use embedded_nn_macros::embedded_nn_model;

#[embedded_nn_model("models/gesture_mlp.json")]
pub struct GestureMlp;
