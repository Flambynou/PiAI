mod neural_network;

fn main() {
    let ascii_digits = b"3141592653589793238462643383279502884197169";
    let binary_digits: Vec<u8> = ascii_digits.iter().map(|c| c - b'0').collect();
    println!("{}",binary_digits[0]);
}
