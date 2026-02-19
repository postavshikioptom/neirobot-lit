use std::collections::VecDeque;

#[derive(Debug, Clone, Default)]
pub struct PnlStats {
    pub pnls: VecDeque<f64>,
    pub sum: f64,
    pub sum_sq: f64,
    pub n: usize,
}

impl PnlStats {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn update(&mut self, x: f64, window: usize) {
        if window == 0 { return; }
        
        while self.pnls.len() >= window {
            if let Some(old) = self.pnls.pop_front() {
                self.sum -= old;
                self.sum_sq -= old * old;
                self.n -= 1;
            }
        }
        
        self.pnls.push_back(x);
        self.sum += x;
        self.sum_sq += x * x;
        self.n += 1;
        
        // Корректировка для минимизации накопления ошибки плавающей точки
        if self.n == 0 {
            self.sum = 0.0;
            self.sum_sq = 0.0;
        }
    }

    pub fn std_dev(&self) -> f64 {
        if self.n < 2 {
            return 0.0;
        }
        let var = (self.sum_sq - (self.sum * self.sum / self.n as f64)) / (self.n - 1) as f64;
        var.max(0.0).sqrt()
    }

    pub fn is_outlier(&self, x: f64, threshold: f64) -> bool {
        if self.n < 2 {
            return false;
        }
        let std = self.std_dev();
        if std <= 0.0000000001 { // Почти ноль
            return false;
        }
        let mean = self.sum / self.n as f64;
        let z_score = (x - mean).abs() / std;
        z_score > threshold
    }
}
