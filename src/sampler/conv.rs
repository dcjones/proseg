
use std::ops::MulAssign;

use ndarray::{Array1, Array2, Zip, s};
use realfft;
use rustfft;
use rustfft::num_complex::Complex;


// FFT based convolution, using pre-allocated buffers, since we are
// going to have to repeatedly recompute this.
pub struct Conv2D {
    kernel: Array2<f32>,
    input: Array2<f32>,
    fwd_output: Array2<Complex<f32>>,
    fwd_data_output_t: Array2<Complex<f32>>,
    fwd_kernel_output_t: Array2<Complex<f32>>,

    fft_fwd_row_buffer: Array1<Complex<f32>>,
    fft_fwd_col_buffer: Array1<Complex<f32>>,

    kernel_ext: Array2<f32>,
    rp: realfft::RealFftPlanner<f32>,
    cp: rustfft::FftPlanner<f32>,
}


impl Conv2D {
    pub fn new(shape: (usize, usize), kernel: Array2<f32>) -> Self {
        let (m_kernel, n_kernel) = kernel.dim();
        if m_kernel % 2 != 1 || n_kernel % 2 != 1 {
            panic!("kernel must have odd dimensions");
        }

        let (m_padded, n_padded) = (shape.0 + m_kernel - 1, shape.1 + n_kernel - 1);

        let kernel_ext = Array2::zeros((m_padded, n_padded));
        let input = Array2::zeros((m_padded, n_padded));
        let fwd_output = Array2::zeros((m_padded, n_padded / 2 + 1));
        let fwd_data_output_t = Array2::zeros((n_padded / 2 + 1, m_padded));
        let fwd_kernel_output_t = Array2::zeros((n_padded / 2 + 1, m_padded));

        let mut rp = realfft::RealFftPlanner::new();
        let cp = rustfft::FftPlanner::new();

        let fft_row = rp.plan_fft_forward(n_padded);
        let fft_fwd_row_buffer = Array1::zeros(fft_row.get_scratch_len());
        let fft_fwd_col_buffer = Array1::zeros((n_padded / 2 + 1) * m_padded);

        return Conv2D {
            kernel, input, fwd_output,
            fwd_data_output_t, fwd_kernel_output_t,
            fft_fwd_row_buffer, fft_fwd_col_buffer,
            kernel_ext,
            rp, cp };
    }

    pub fn shape(&self) -> (usize, usize) {
        let (m_kernel, n_kernel) = self.kernel.dim();
        return (self.input.shape()[0] - m_kernel + 1, self.input.shape()[1] - n_kernel + 1);
    }

    pub fn compute(&mut self, data: &mut Array2<f32>) {
        let (m, n) = data.dim();
        let (m_kernel, n_kernel) = self.kernel.dim();
        let (m_pad, n_pad) = ((m_kernel - 1) / 2, (n_kernel - 1) / 2);
        let (m_padded, n_padded) = self.input.dim();

        // dbg!(self.shape());
        // dbg!((m, n, m_pad, n_pad, m_padded, n_padded));
        assert!(m + 2*m_pad <= m_padded && n + 2*n_pad <= n_padded);

        self.kernel_ext.fill(0.0);
        self.kernel_ext.slice_mut(s![0..m_kernel, 0..n_kernel]).assign(&self.kernel);

        self.input.fill(0.0);
        self.input.slice_mut(s![m_pad..m_pad+m, n_pad..n_pad+n]).assign(&data);

        let fft_row = self.rp.plan_fft_forward(n_padded);
        let fft_col = self.cp.plan_fft_forward(m_padded);

        // data forward fft
        Zip::from(self.input.rows_mut())
            .and(self.fwd_output.rows_mut())
            .for_each(|mut row, mut output| {
                fft_row
                    .process_with_scratch(
                        row.as_slice_mut().unwrap(),
                        output.as_slice_mut().unwrap(),
                        self.fft_fwd_row_buffer.as_slice_mut().unwrap(),
                    )
                    .unwrap();
            });

        self.fwd_data_output_t.assign(&self.fwd_output.t());
        fft_col.process_with_scratch(
            self.fwd_data_output_t.as_slice_mut().unwrap(),
            self.fft_fwd_col_buffer.as_slice_mut().unwrap(),
        );

        // kernel forward fft
        Zip::from(self.kernel_ext.rows_mut())
            .and(self.fwd_output.rows_mut())
            .for_each(|mut row, mut output| {
                fft_row
                    .process_with_scratch(
                        row.as_slice_mut().unwrap(),
                        output.as_slice_mut().unwrap(),
                        self.fft_fwd_row_buffer.as_slice_mut().unwrap(),
                    )
                    .unwrap();
            });

        self.fwd_kernel_output_t.assign(&self.fwd_output.t());

        fft_col.process_with_scratch(
            self.fwd_kernel_output_t.as_slice_mut().unwrap(),
            self.fft_fwd_col_buffer.as_slice_mut().unwrap(),
        );

        // multiply
        self.fwd_data_output_t.mul_assign(&self.fwd_kernel_output_t);

        // inverse fft on product
        let ifft_row = self.rp.plan_fft_inverse(n_padded);
        let ifft_col = self.cp.plan_fft_inverse(m_padded);

        ifft_col.process_with_scratch(
            self.fwd_data_output_t.as_slice_mut().unwrap(),
            self.fft_fwd_col_buffer.as_slice_mut().unwrap()
        );

        self.fwd_output.assign(&self.fwd_data_output_t.t());

        Zip::from(self.fwd_output.rows_mut())
            .and(self.input.rows_mut())
            .for_each(|mut row, mut output| {
                if ifft_row.len() % 2 == 1 {
                    unsafe { row.uget_mut(0).im = 0.0 };
                } else {
                    unsafe { row.uget_mut(0).im = 0.0 };
                    unsafe { row.uget_mut(row.len() - 1).im = 0.0 }
                };

                ifft_row
                    .process_with_scratch(
                        row.as_slice_mut().unwrap(),
                        output.as_slice_mut().unwrap(),
                        self.fft_fwd_row_buffer.as_slice_mut().unwrap())
                    .unwrap();
            });

        let len_f32 = self.input.len() as f32;
        self.input.map_mut(|x| *x /= len_f32);

        // assign un-padded result back into data (computing the convolution in place)
        data.assign(&self.input.slice(s![m_kernel-1.., n_kernel-1..]));
    }
}


