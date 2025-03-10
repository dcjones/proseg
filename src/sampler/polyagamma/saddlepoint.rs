use super::common::{random_left_bounded_gamma, upper_incomplete_gamma};
use super::float::Float;
use numeric_literals::replace_float_literals;
use rand::Rng;
use rand_distr::{Distribution, Exp1, StandardNormal, StandardUniform};

#[replace_float_literals(T::from(literal).unwrap())]
pub fn sample_polyagamma_saddlepoint<T: Float, R: Rng>(rng: &mut R, h: T, z: T) -> T
where
    StandardNormal: Distribution<T>,
    StandardUniform: Distribution<T>,
    Exp1: Distribution<T>,
{
    let mut pr = Parameters::new(h, 0.5 * z.abs());
    let sqrt_rho = (-2.0 * pr.left_tangent_slope).sqrt();
    let sqrt_rho_inv = sqrt_rho.recip();

    let p = (h * (0.5 / pr.xc + pr.left_tangent_intercept - sqrt_rho)
        + invgauss_logcdf(pr.xc, sqrt_rho_inv, h))
    .exp()
        + pr.sqrt_alpha;
    assert!(p.is_finite());

    let hrho = -h * pr.right_tangent_slope;

    // let q = upper_incomplete_gamma(h, hrho * pr.xc, false) * pr.right_kernel_coef *
    //     (h * (pr.right_tangent_intercept - hrho.ln())).exp();
    let q = ((upper_incomplete_gamma(h, hrho * pr.xc, false) * pr.right_kernel_coef).ln()
        + (h * (pr.right_tangent_intercept - hrho.ln())))
    .exp();
    assert!(q.is_finite());

    let proposal_probability = p / (p + q);

    let mu2 = sqrt_rho_inv * sqrt_rho_inv;

    loop {
        let u = T::from(rng.random::<f32>()).unwrap();
        if u < proposal_probability {
            loop {
                let y = rng.sample::<T, StandardNormal>(StandardNormal);
                let w = sqrt_rho_inv + 0.5 * mu2 * y * y / h;
                pr.x = w - (w * w - mu2).abs().sqrt();

                if rng.sample::<T, StandardUniform>(StandardUniform) * (1.0 + pr.x * sqrt_rho) > 1.0
                {
                    pr.x = mu2 / pr.x;
                }

                if pr.x < pr.xc {
                    break;
                }
            }
        } else {
            pr.x = random_left_bounded_gamma(rng, h, hrho, pr.xc);
        }

        if T::from(rng.random::<f32>()).unwrap() * bounding_kernel(&pr) <= saddle_point(&pr) {
            break;
        }
    }

    0.25 * h * pr.x
}

struct Parameters<T: Float> {
    // y intercept of tangent line to xr.
    right_tangent_intercept: T,
    // y intercept of tangent line to xl.
    left_tangent_intercept: T,
    // derivative of the line to xr
    right_tangent_slope: T,
    // derivative of the line to xl
    left_tangent_slope: T,
    // config->sqrt_h2pi * config->sqrt_alpha_r
    right_kernel_coef: T,
    // config->sqrt_h2pi * config->sqrt_alpha
    left_kernel_coef: T,
    // sqrt(1 / alpha_l) constant
    sqrt_alpha: T,
    // log(cosh(z))
    log_cosh_z: T,
    // the constant sqrt(h / (2 * pi))
    sqrt_h2pi: T,
    // 0.5 * z * z
    half_z2: T,
    // log of center point
    logxc: T,
    xc: T,
    h: T,
    // z: T,
    x: T,
}

impl<T: Float> Parameters<T> {
    /*
     * Configure some constants to be used during sampling.
     *
     * NOTE
     * ----
     * Note that unlike the recommendations of Windle et al (2014) to use
     * xc = 1.1 * xl and xr = 1.2 * xl, we found that using xc = 2.75 * xl and
     * xr = 3 * xl provides the best envelope for the target density function and
     * thus gives the best performance in terms of runtime due to the algorithm
     * rejecting much fewer proposals; while the former produces an envelope that
     * exceeds the target by too much in height and has a narrower variance, thus
     * leading to many rejected proposal samples. Tests show that the latter
     * selection results in the saddle approximation being over twice as fast as
     * when using the former. Also note that log(xr) and log(xc) need not be
     * computed directly. Since both xr and xc depend on xl, then their logs can
     * be written as log(xr) = log(3) + log(xl) and log(xc) = log(2.75) + log(xl).
     * Thus the log() function can be called once on xl and then the constants
     * be precalculated at compile time, making the calculation of the logs a
     * little more efficient.
     */
    #[replace_float_literals(T::from(literal).unwrap())]
    fn new(h: T, z: T) -> Self {
        let log275 = 1.0116009116784799;
        let log3 = 1.0986122886681098;

        let xl;
        let logxl;
        let half_z2;
        let log_cosh_z;

        if z > 0. {
            xl = tanh_x(z);
            logxl = xl.ln();
            half_z2 = 0.5 * (z * z);
            log_cosh_z = z.cosh().ln();
        } else {
            xl = 1.;
            logxl = 0.;
            half_z2 = 0.;
            log_cosh_z = 0.;
        }

        let xc = 2.75 * xl;
        let xr = 3. * xl;

        let xc_inv = xc.recip();
        let xl_inv = xl.recip();
        let ul = -half_z2;

        let (ur, _) = newton_raphson(xr, select_starting_guess(xr));
        let (_, rv) = newton_raphson(xc, select_starting_guess(xc));
        let tr = ur + half_z2;

        // t = 0 at x = m, since K'(0) = m when t(x) = 0
        let left_tangent_slope = -0.5 * (xl_inv * xl_inv);
        let left_tangent_intercept = cumulant(ul, log_cosh_z) - 0.5 * xc_inv + xl_inv;
        assert!(left_tangent_intercept.is_finite());
        let logxc = log275 + logxl;
        let right_tangent_slope = -tr - 1. / xr;
        let right_tangent_intercept = cumulant(ur, log_cosh_z) + 1.0 - log3 - logxl + logxc;
        assert!(right_tangent_intercept.is_finite());

        let alpha_r = rv.fprime * (xc_inv * xc_inv); // K''(t(xc)) / xc^2
        let alpha_l = xc_inv * alpha_r; // K''(t(xc)) / xc^3

        let sqrt_alpha = 1.0 / alpha_l.sqrt();

        // let sqrt_h2pi = (h / 6.283185307179586).sqrt();
        let sqrt_h2pi = (h / T::TAU).sqrt();
        let left_kernel_coef = sqrt_h2pi * sqrt_alpha;
        let right_kernel_coef = sqrt_h2pi / alpha_r.sqrt();

        Parameters {
            right_tangent_intercept,
            left_tangent_intercept,
            right_tangent_slope,
            left_tangent_slope,
            right_kernel_coef,
            left_kernel_coef,
            sqrt_alpha,
            log_cosh_z,
            sqrt_h2pi,
            half_z2,
            logxc,
            xc,
            h,
            // z,
            x: 0.0,
        }
    }
}

/*
 * Compute f(x) = tanh(x) / x in the range [0, infinity).
 *
 * This implementation is based off the analysis presented by Beebe [1].
 * For x <= 5, we use a rational polynomial approximation of Cody and Waite [2].
 * For x > 5, we use g(x) = 1 / x to approximate the function.
 *
 * Tests show that the absolute maximum relative error compared to output
 * produced by the standard library tanh(x) / x is 9.080398e-05.
 *
 * References
 * ---------
 * [1] Beebe, Nelson H. F. (1993). Accurate hyperbolic tangent computation.
 *     Technical report, Center for Scientific Computing, Department of
 *     Mathematics, University ofUtah, Salt Lake City, UT 84112, USA, April 20
 *    1993.
 * [2] William J. Cody, Jr. and William Waite. Software Manual for the Elementary
 *     Functions. Prentice-Hall, Upper Saddle River, NJ 07458, USA, 1980.
 *     ISBN 0-13-822064-6. x + 269 pp. LCCN QA331 .C635 1980.
 */
#[replace_float_literals(T::from(literal).unwrap())]
fn tanh_x<T: Float>(x: T) -> T {
    if x > 4.95 {
        return 1. / x;
    }
    let p0 = -1.613_411_902_399_622_7e3;
    let p1 = -9.922_592_967_223_608e1;
    let p2 = -9.643_749_277_722_548e-1;
    /* denominator coefficients */
    let q0 = 4.840_235_707_198_869e3;
    let q1 = 2.233_772_071_896_231_4e3;
    let q2 = 1.127_447_438_053_494_9e2;
    let x2 = x * x;

    1. + x2 * ((p2 * x2 + p1) * x2 + p0) / (((x2 + q2) * x2 + q1) * x2 + q0)
}

fn tan_x<T: Float>(x: T) -> T {
    x.tan() / x
}

/*
 * A struct to store a function's value and derivative at a point.
 */
struct FuncReturnValue<T: Float> {
    f: T,
    fprime: T,
}

/*
 * compute K(t), the cumulant generating function of X
 */
// #define cumulant(u, v)                                              \
//     ((u) < 0. ? (v)->log_cosh_z - logf(coshf(sqrtf(-2. * (u)))) :   \
//      (u) > 0. ? (v)->log_cosh_z - logf(cosf(sqrtf(2. * (u)))) :     \
//      (v)->log_cosh_z)                                               \
#[replace_float_literals(T::from(literal).unwrap())]
fn cumulant<T: Float>(u: T, log_cosh_z: T) -> T {
    if u < 0.0 {
        log_cosh_z - (-2.0 * u).sqrt().cosh().ln()
    } else if u > 0.0 {
        log_cosh_z - (2.0 * u).sqrt().cos().ln()
    } else {
        log_cosh_z
    }
}

/*
 * Compute K'(t), the derivative of the Cumulant Generating Function (CGF) of X.
 */
#[replace_float_literals(T::from(literal).unwrap())]
fn cumulant_prime<T: Float>(u: T) -> FuncReturnValue<T> {
    let s = 2.0 * u;
    let f = if s < 0.0 {
        tanh_x((-s).sqrt())
    } else if s > 0.0 {
        tan_x(s.sqrt())
    } else {
        1.0
    };
    let fprime = f * f + (1.0 - f) / s;

    FuncReturnValue { f, fprime }
}

fn isclose<T: Float>(a: T, b: T, atol: T, rtol: T) -> bool {
    (a - b).abs() <= atol.max(rtol * a.abs().max(b.abs()))
}

/*
 * Select the starting guess for the solution `u` of Newton's method given a
 * value of x.
 *
 * - When x = 1, then u = 0.
 * - When x < 1, then u < 0.
 * - When x > 1, then u > 0.
 *
 * Page 16 of Windle et al. (2014) shows that the upper bound of `u` is pi^2/8.
 */
#[replace_float_literals(T::from(literal).unwrap())]
fn select_starting_guess<T: Float>(x: T) -> T {
    if x <= 0.25 {
        -9.0
    } else if x <= 0.5 {
        -1.78
    } else if x <= 1.0 {
        -0.147
    } else if x <= 1.5 {
        0.345
    } else if x <= 2.5 {
        0.72
    } else if x <= 4.0 {
        0.95
    } else {
        1.15
    }
}

const PGM_MAX_ITER: usize = 25;

#[replace_float_literals(T::from(literal).unwrap())]
fn newton_raphson<T: Float>(arg: T, mut x0: T) -> (T, FuncReturnValue<T>) {
    let atol = 1e-05;
    let rtol = 1e-05;

    let mut n = 0;
    let mut x = x0;
    let mut value;
    loop {
        x0 = x;
        value = cumulant_prime(x0);
        let fval = value.f - arg;
        if fval.abs() <= atol || value.fprime <= atol {
            return (x0, value);
        }
        x = x0 - fval / value.fprime;

        n += 1;

        if isclose(x, x0, atol, rtol) || n >= PGM_MAX_ITER {
            break;
        }
    }

    (x, value)
}

/*
 * Compute the saddle point estimate at x.
 */
fn saddle_point<T: Float>(pr: &Parameters<T>) -> T {
    let (u, rv) = newton_raphson(pr.x, select_starting_guess(pr.x));
    let t = u + pr.half_z2;

    (pr.h * (cumulant(u, pr.log_cosh_z) - t * pr.x)).exp() * pr.sqrt_h2pi / rv.fprime.sqrt()
}

/*
 * k(x|h,z): The bounding kernel of the saddle point approximation. See
 * Proposition 17 of Windle et al (2014).
 */
#[replace_float_literals(T::from(literal).unwrap())]
fn bounding_kernel<T: Float>(pr: &Parameters<T>) -> T {
    if pr.x > pr.xc {
        let point = pr.right_tangent_slope * pr.x + pr.right_tangent_intercept;
        (pr.h * (pr.logxc + point) + (pr.h - 1.0) * pr.x.ln()).exp() * pr.right_kernel_coef
    } else {
        let point = pr.left_tangent_slope * pr.x + pr.left_tangent_intercept;
        (0.5 * pr.h * (1.0 / pr.xc - 1.0 / pr.x) + pr.h * point - 1.5 * pr.x.ln()).exp()
            * pr.left_kernel_coef
    }
}

/*
 * Compute the logarithm of the standard normal distribution function (cdf).
 */
#[replace_float_literals(T::from(literal).unwrap())]
fn log_norm_cdf<T: Float>(x: T) -> T {
    (-0.5 * (x / T::SQRT_2).erfc()).ln_1p()
}

/*
 * Calculate the logarithm of the cumulative distribution function of an
 * Inverse-Gaussian.
 *
 * We use the computation method presented in [1] to avoid numerical issues
 * when the inputs have very large/small values.
 *
 * References
 * ----------
 *  [1] Giner, Goknur and G. Smyth. “statmod: Probability Calculations for the
 *      Inverse Gaussian Distribution.” R J. 8 (2016): 339.
 */
#[replace_float_literals(T::from(literal).unwrap())]
fn invgauss_logcdf<T: Float>(x: T, mu: T, lambda: T) -> T {
    let qm = x / mu;
    let tm = mu / lambda;
    let r = (x / lambda).sqrt();
    let a = log_norm_cdf((qm - 1.) / r);
    let b = 2. / tm + log_norm_cdf(-(qm + 1.) / r);

    a + (b - a).exp().ln_1p()
}
