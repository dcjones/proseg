
use numeric_literals::replace_float_literals;
use rand::Rng;
use rand_distr::{Distribution, StandardNormal, Standard, Exp1};
use super::float::Float;
use super::common::{upper_incomplete_gamma, pgm_lgamma, random_left_bounded_gamma};


#[replace_float_literals(T::from(literal).unwrap())]
pub fn sample_polyagamma_alternate<T: Float, R: Rng>(rng: &mut R, mut h: T, z: T) -> T
where Exp1: Distribution<T>, StandardNormal: Distribution<T>
{
    let mut pr = Parameters::new(h, z);
    let pgm_maxh = T::from(PGM_MAXH).unwrap();
    let mut out = T::zero();

    if h > pgm_maxh {
        let chunk = if h >= pgm_maxh + 1.0 {
            pgm_maxh
        } else {
            pgm_maxh - 1.0
        };
        pr.set(chunk, false);

        while h > pgm_maxh {
            out += 0.25 * pr.random_jacobi_star(rng);
            h -= chunk;
        }

        pr.set(h, true);
        out += 0.25 * pr.random_jacobi_star(rng);
        return out;
    }

    pr.set(h, false);
    out += 0.25 * pr.random_jacobi_star(rng);
    return out;
}


const PGM_LOG2: f64 = 0.6931471805599453094172321214581766; // log(2)
const PGM_LOGPI_2: f64 = 0.4515827052894548647261952298948821;  // log(pi / 2)
const PGM_PI2_8: f64 = 1.233700550136169827354311374984519;  // pi^2 / 8
const SQRT2_INV: f64 = 0.7071067811865475;
const PGM_LS2PI: f64 = 0.9189385332046727417803297364056177;  // log(sqrt(2 * pi))

struct Parameters<T: Float> {
    proposal_probability: T,
    log_lambda_z: T,
    lambda_z: T,
    half_h2: T,
    // loggamma(h)
    lgammah: T,
    hlog2: T,
    // 1 / t: T,
    t_inv: T,
    logx: T,
    // (h / z) ** 2
    h_z2: T,
    h_z: T,
    z2: T,
    h: T,
    z: T,
    x: T,
    t: T,
}


impl<T: Float> Parameters<T> {
    #[replace_float_literals(T::from(literal).unwrap())]
    fn new(h: T, z: T) -> Self {
        let params =  Self {
            proposal_probability: T::zero(),
            log_lambda_z: T::zero(),
            lambda_z: T::zero(),
            half_h2: T::zero(),
            lgammah: T::zero(),
            hlog2: T::zero(),
            t_inv: T::zero(),
            logx: T::zero(),
            h_z2: T::zero(),
            h_z: T::zero(),
            z2: T::zero(),
            h: T::zero(),
            z: 0.5 * z.abs(),
            x: T::zero(),
            t: T::zero(),
        };

        return params;
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    fn set(&mut self, h: T, update: bool) {
        self.h = h;
        self.t = get_truncation_point(h);
        self.t_inv = self.t.recip();
        self.half_h2 = 0.5 * h * h;
        self.lgammah = pgm_lgamma(h);
        self.hlog2 = h * T::from(PGM_LOG2).unwrap();
        let p;

        if !update && self.z > 0.0 {
            self.h_z = h / self.z;
            self.z2 = self.z * self.z;
            self.h_z2 = self.h_z * self.h_z;
            self.lambda_z = T::from(PGM_PI2_8).unwrap() + 0.5 * self.h_z2;
            self.log_lambda_z = self.lambda_z.ln();
            p = (self.hlog2 - h * self.z).exp() * self.invgauss_cdf();
        } else if self.z > 0.0 {
            self.h_z = h / self.z;
            self.h_z2 = self.h_z * self.h_z;
            p = (self.hlog2 - h * self.z).exp() * self.invgauss_cdf();
        } else if !update {
            self.lambda_z = T::from(PGM_PI2_8).unwrap();
            self.log_lambda_z = self.lambda_z.ln();
            p = self.hlog2.exp() * (h / (2.0 * self.t).sqrt()).erfc();
        } else {
            p = self.hlog2.exp() * (h / (2.0 * self.t).sqrt()).erfc();
        }

        let q = (h * (T::from(PGM_LOGPI_2).unwrap() - self.log_lambda_z)).exp() *
            upper_incomplete_gamma(h, self.lambda_z * self.t, true);

        self.proposal_probability = q / (p + q);
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    fn invgauss_cdf(&self) -> T {

        let st = self.t.sqrt();
        let a = T::from(SQRT2_INV).unwrap() * self.h / st;
        let b = self.z * st * T::from(SQRT2_INV).unwrap();
        let ez = (self.h * self.z).exp();

        return 0.5 * ((a - b).erfc() + ez * (b + a).erfc() * ez);
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    fn random_jacobi_star<R: Rng>(&mut self, rng: &mut R) -> T
    where Exp1: Distribution<T>, StandardNormal: Distribution<T>
    {
        loop {
            let u = T::from(rng.gen::<f32>()).unwrap();
            if u <= self.proposal_probability {
                self.x = random_left_bounded_gamma(rng, self.h, self.lambda_z, self.t);
            } else if self.z > 0.0 {
                self.random_right_bounded_invgauss(rng);
            } else {
                self.x = 1.0 / random_left_bounded_gamma(rng, 0.5, self.half_h2, self.t_inv);
            }

            self.logx = self.x.ln();
            let u = T::from(rng.gen::<f32>()).unwrap() * self.bounding_kernel();
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

    #[replace_float_literals(T::from(literal).unwrap())]
    fn piecewise_coef(&self, n: u32) -> T {
        let a = 2.0 * T::from(n).unwrap() + self.h;
        let b = if n != 0 {
            pgm_lgamma(T::from(n).unwrap() + self.h) - self.lgammah
        } else {
            0.0
        };

        return (self.hlog2 + b -
            pgm_lgamma(T::from(n + 1).unwrap()) -
            T::from(PGM_LS2PI).unwrap() -
            1.5 * self.logx -
            0.5 * a * a / self.x).exp() * a;
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    fn bounding_kernel(&self) -> T {
        if self.x > self.t {
            let a = 0.22579135264472733;
            return (self.h * a + (self.h - 1.0) * self.logx -
                T::from(PGM_PI2_8).unwrap() * self.x -
                self.lgammah).exp();
        } else if self.x > 0.0 {
            return (self.hlog2 - self.half_h2 / self.x -
                1.5 * self.logx -
                T::from(PGM_LS2PI).unwrap()).exp() * self.h;
        }
        return 0.0;
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    fn random_right_bounded_invgauss<R: Rng>(&mut self, rng: &mut R)
    where Exp1: Distribution<T>, StandardNormal: Distribution<T>
    {
        if self.t < self.h_z {
            loop {
                self.x = random_left_bounded_gamma(rng, 0.5, self.half_h2, self.t_inv).recip();

                let u = T::from(rng.gen::<f32>()).unwrap();
                if (-u).ln_1p() < -0.5 * self.z2 * self.x {
                    return;
                }
            }
        }
        loop {
            let y = rng.sample::<T, StandardNormal>(StandardNormal);
            let w = self.h_z + 0.5 * y * y / self.z2;
            self.x = w - (w * w - self.h_z2).abs().sqrt();

            let u = T::from(rng.gen::<f32>()).unwrap();
            if u * (self.h_z + self.x) > self.h_z {
                self.x = self.h_z2 / self.x;
            }

            if self.x < self.t {
                break;
            }
        }
    }

}



const PGM_H: [f32; 25] = [
    1.000000000, 1.125000000, 1.250000000, 1.375000000, 1.500000000,
    1.625000000, 1.750000000, 1.875000000, 2.000000000, 2.125000000,
    2.250000000, 2.375000000, 2.500000000, 2.625000000, 2.750000000,
    2.875000000, 3.000000000, 3.125000000, 3.250000000, 3.375000000,
    3.500000000, 3.625000000, 3.750000000, 3.875000000, 4.000000000
];

const PGM_F: [f32; 25] = [
    1.273239366, 1.901515423, 2.281992126, 2.607829551, 2.910421526,
    3.200449543, 3.482766779, 3.759955106, 4.033540671, 4.304486011,
    4.573437633, 4.840840644, 5.107017272, 5.372204821, 5.636581947,
    5.900288573, 6.163432428, 6.426094330, 6.688351603, 6.950254767,
    7.211854235, 7.473186206, 7.734284136, 7.995175158, 8.255882407
];

const PGM_MAXH: f32 = 4.0;


#[replace_float_literals(T::from(literal).unwrap())]
fn get_truncation_point<T: Float>(h: T) -> T {
    if h < 1.0 {
        return T::from(PGM_F[0]).unwrap();
    } else if h == T::from(PGM_MAXH).unwrap() {
        return T::from(*PGM_F.last().unwrap()).unwrap();
    } else {
        // start binary search
        let mut offset = 0;
        let mut len = PGM_H.len() - 1;
        let mut index = offset + len / 2;

        while len > 0 {
            index = offset + len / 2;
            if T::from(PGM_H[index]).unwrap() < h {
                len = len - (index + 1 - offset);
                offset = index + 1;
                continue;
            } else if offset < index && T::from(PGM_H[index]).unwrap() > h {
                len = index - offset;
                continue;
            } else {
                break;
            }
        }

        let x0 = PGM_H[index - 1];
        let f0 = PGM_F[index - 1];
        let x1 = PGM_H[index + 1];
        let f1 = PGM_F[index + 1];

        return T::from(f0 + (f1 - f0) * (PGM_H[index] - x0) / (x1 - x0)).unwrap();
    }
}