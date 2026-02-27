fn main() { 
    let s: ort::session::Session = unsafe { std::mem::zeroed() }; // Заглушка, чтобы код после нее был достижим
    println!("{:?}", s.inputs()[0].name()); 
}
