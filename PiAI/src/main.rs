mod neural_network;

const WINDOW_SIZE: usize = 10;

const NETWORK: [u32; 5] = [WINDOW_SIZE as u32, 64, 64, 32, 1];
const ACTIVATION_FUNC_HIDDEN : usize = 3;
const ACTIVATION_FUNC_OUTPUT : usize = 1;
const INITIAL_WEIGHT_RANGE : (f32, f32) = (-1.0, 1.0);
const INITIAL_BIAS_RANGE : (f32, f32) = (-0.1, 0.1);


fn main() {
    let ascii_digits = b"3141592653589793238462643383279502884197169";
    let binary_digits: Vec<u8> = ascii_digits.iter().map(|c| c - b'0').collect();

}


fn run(neural_network:&neural_network::NeuralNetwork, binary_digits:&Vec<u8>) -> usize {
    let mut next_digit_index:usize =  WINDOW_SIZE;
    loop {
        let window:Vec<u8> = Vec::from(&binary_digits[next_digit_index-WINDOW_SIZE..next_digit_index]);
        let normalized_window:Vec<f32> = window.iter().map(|x| *x as f32/10.0 as f32).collect();
        if !((neural_network.feed_forward(normalized_window)[0] * 10.0) as u8 == binary_digits[next_digit_index]) {
            break;
        }
        next_digit_index += 1;
    }
    return next_digit_index-WINDOW_SIZE;
}