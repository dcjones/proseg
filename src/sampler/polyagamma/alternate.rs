use super::common::{pgm_lgamma, random_left_bounded_gamma, upper_incomplete_gamma};
use super::float::Float;
use rand::Rng;
use rand_distr::StandardNormal;

const PGM_MAXH: f64 = 4.0;

pub fn sample_polyagamma_alternate<R: Rng>(rng: &mut R, mut h: f64, z: f64) -> f64 {
    let mut pr = Parameters::new(z);
    let mut out = 0.0;

    if h > PGM_MAXH {
        let chunk = if h >= PGM_MAXH + 1.0 {
            PGM_MAXH
        } else {
            PGM_MAXH - 1.0
        };
        pr.set(chunk, false);

        while h > PGM_MAXH {
            out += 0.25 * pr.random_jacobi_star(rng);
            h -= chunk;
        }

        pr.set(h, true);
        out += 0.25 * pr.random_jacobi_star(rng);
        out
    } else {
        pr.set(h, false);
        out += 0.25 * pr.random_jacobi_star(rng);
        out
    }
}

const PGM_LOG2: f64 = std::f64::consts::LN_2; // log(2)
const PGM_LOGPI_2: f64 = 0.451_582_705_289_454_9; // log(pi / 2)
const PGM_PI2_8: f64 = 1.233_700_550_136_169_7; // pi^2 / 8
const SQRT2_INV: f64 = 0.7071067811865475;
const PGM_LS2PI: f64 = 0.918_938_533_204_672_8; // log(sqrt(2 * pi))

struct Parameters {
    proposal_probability: f32,
    log_lambda_z: f64,
    lambda_z: f64,
    half_h2: f64,
    lgammah: f64,
    hlog2: f64,
    t_inv: f64,
    logx: f64,
    h_z2: f64,
    h_z: f64,
    z2: f64,
    h: f64,
    z: f64,
    x: f64,
    t: f64,
}

impl Parameters {
    fn new(z: f64) -> Self {
        Self {
            proposal_probability: 0.0,
            log_lambda_z: 0.0,
            lambda_z: 0.0,
            half_h2: 0.0,
            lgammah: 0.0,
            hlog2: 0.0,
            t_inv: 0.0,
            logx: 0.0,
            h_z2: 0.0,
            h_z: 0.0,
            z2: 0.0,
            h: 0.0,
            z: 0.5 * z.abs(),
            x: 0.0,
            t: 0.0,
        }
    }

    fn set(&mut self, h: f64, update: bool) {
        self.h = h;
        self.t = get_truncation_point(h);
        self.t_inv = self.t.recip();
        self.half_h2 = 0.5 * h * h;
        self.lgammah = pgm_lgamma(h);
        self.hlog2 = h * PGM_LOG2;
        let p;

        if !update && self.z > 0.0 {
            self.h_z = h / self.z;
            self.z2 = self.z * self.z;
            self.h_z2 = self.h_z * self.h_z;
            self.lambda_z = PGM_PI2_8 + 0.5 * self.z2;
            self.log_lambda_z = self.lambda_z.ln();
            p = (self.hlog2 - h * self.z).exp() * self.invgauss_cdf() as f64;
        } else if self.z > 0.0 {
            self.h_z = h / self.z;
            self.h_z2 = self.h_z * self.h_z;
            p = (self.hlog2 - h * self.z).exp() * self.invgauss_cdf() as f64;
        } else if !update {
            self.lambda_z = PGM_PI2_8;
            self.log_lambda_z = self.lambda_z.ln();
            p = self.hlog2.exp() * (h / (2.0 * self.t).sqrt()).erfc();
        } else {
            p = self.hlog2.exp() * (h / (2.0 * self.t).sqrt()).erfc();
        }

        let q = (h * (PGM_LOGPI_2 - self.log_lambda_z)).exp()
            * upper_incomplete_gamma(h, self.lambda_z * self.t, true);

        self.proposal_probability = (q / (p + q)) as f32;
    }

    fn invgauss_cdf(&self) -> f32 {
        let st = self.t.sqrt();
        let a = SQRT2_INV * self.h / st;
        let b = self.z * st * SQRT2_INV;
        let ez = (self.h * self.z).exp() as f32;

        0.5f32 * (((a - b) as f32).erfc() + ez * ((b + a) as f32).erfc() * ez)
    }

    fn random_jacobi_star<R: Rng>(&mut self, rng: &mut R) -> f64 {
        loop {
            let u = rng.random::<f32>();
            if u <= self.proposal_probability {
                self.x = random_left_bounded_gamma(rng, self.h, self.lambda_z, self.t);
            } else if self.z > 0.0 {
                self.random_right_bounded_invgauss(rng);
            } else {
                self.x = random_left_bounded_gamma(rng, 0.5, self.half_h2, self.t_inv).recip();
            }

            // we get stuck in an infinite loop below if x is 0
            assert!(self.x > 0.0);

            self.logx = (self.x as f32).ln() as f64;
            let u = rng.random::<f32>() * self.bounding_kernel();
            let mut s = self.piecewise_coef(0);

            let mut n = 1;
            loop {
                let old_s = s;
                if n & 1 != 0 {
                    s -= self.piecewise_coef(n);
                    if old_s >= s && u <= s {
                        return self.x;
                    }
                } else {
                    s += self.piecewise_coef(n);
                    if old_s >= s && u > s {
                        break;
                    }
                }
                n += 1;
            }
        }
    }

    fn piecewise_coef(&self, n: u32) -> f32 {
        let a = 2.0 * (n as f64) + self.h;
        let b = if n != 0 {
            pgm_lgamma(n as f64 + self.h) - self.lgammah
        } else {
            0.0
        };

        ((self.hlog2 + b
            - pgm_lgamma((n + 1) as f64)
            - PGM_LS2PI
            - 1.5 * self.logx
            - 0.5 * a * a / self.x) as f32)
            .exp()
            * a as f32
    }

    fn bounding_kernel(&self) -> f32 {
        if self.x > self.t {
            let a = 0.22579135264472733;
            ((self.h * a + (self.h - 1.0) * self.logx - PGM_PI2_8 * self.x - self.lgammah) as f32)
                .exp()
        } else if self.x > 0.0 {
            ((self.hlog2 - self.half_h2 / self.x - 1.5 * self.logx - PGM_LS2PI) as f32).exp()
                * self.h as f32
        } else {
            0.0
        }
    }

    fn random_right_bounded_invgauss<R: Rng>(&mut self, rng: &mut R) {
        if self.t < self.h_z {
            loop {
                self.x = random_left_bounded_gamma(rng, 0.5, self.half_h2, self.t_inv).recip();

                let u = rng.random::<f32>();
                if (-u).ln_1p() < (-0.5 * self.z2 * self.x) as f32 {
                    return;
                }
            }
        }
        loop {
            let y = rng.sample::<f64, StandardNormal>(StandardNormal);
            let w = self.h_z + 0.5 * y * y / self.z2;
            self.x = w - (w * w - self.h_z2).abs().sqrt();

            // If `y` happens to be very large, we can end up with x=0,
            // causing things to break downstream
            self.x = self.x.max(1e-10);

            let u = rng.random::<f64>();
            if u * (self.h_z + self.x) > self.h_z {
                self.x = self.h_z2 / self.x;
            }

            assert!(self.x > 0.0);

            if self.x < self.t {
                break;
            }
        }
    }
}

#[allow(clippy::excessive_precision)]
const PGM_H: [f32; 25] = [
    1.000000000,
    1.125000000,
    1.250000000,
    1.375000000,
    1.500000000,
    1.625000000,
    1.750000000,
    1.875000000,
    2.000000000,
    2.125000000,
    2.250000000,
    2.375000000,
    2.500000000,
    2.625000000,
    2.750000000,
    2.875000000,
    3.000000000,
    3.125000000,
    3.250000000,
    3.375000000,
    3.500000000,
    3.625000000,
    3.750000000,
    3.875000000,
    4.000000000,
];

#[allow(clippy::excessive_precision)]
const PGM_F: [f32; 25] = [
    1.273239366,
    1.901515423,
    2.281992126,
    2.607829551,
    2.910421526,
    3.200449543,
    3.482766779,
    3.759955106,
    4.033540671,
    4.304486011,
    4.573437633,
    4.840840644,
    5.107017272,
    5.372204821,
    5.636581947,
    5.900288573,
    6.163432428,
    6.426094330,
    6.688351603,
    6.950254767,
    7.211854235,
    7.473186206,
    7.734284136,
    7.995175158,
    8.255882407,
];

fn get_truncation_point(h: f64) -> f64 {
    if h < 1.0 {
        PGM_F[0] as f64
    } else if h == PGM_MAXH {
        *PGM_F.last().unwrap() as f64
    } else {
        // start binary search
        let mut offset = 0;
        let mut len = PGM_H.len() - 1;
        let mut index = offset + len / 2;

        while len > 0 {
            index = offset + len / 2;
            if (PGM_H[index] as f64) < h {
                len -= index + 1 - offset;
                offset = index + 1;
            } else if offset < index && (PGM_H[index] as f64) > h {
                len = index - offset;
            } else {
                break;
            }
        }

        let x0 = PGM_H[index];
        let f0 = PGM_F[index];
        let x1 = PGM_H[index + 1];
        let f1 = PGM_F[index + 1];

        (f0 + (f1 - f0) * (h as f32 - x0) / (x1 - x0)) as f64
    }
}
