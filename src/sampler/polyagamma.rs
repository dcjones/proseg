// use num_traits::{cast::FromPrimitive, float::Float};
use numeric_literals::replace_float_literals;
use rand::Rng;
use rand_distr::{Distribution, Exp1, Normal, StandardNormal, StandardUniform};

mod float;
use float::Float;

mod common;
mod saddlepoint;
use saddlepoint::sample_polyagamma_saddlepoint;

mod alternate;
use alternate::sample_polyagamma_alternate;

pub struct PolyaGamma<T: Float> {
    h: T,
    z: T,
}

#[replace_float_literals(T::from(literal).unwrap())]
fn sech<T: Float>(x: T) -> T {
    1.0 / x.cosh()
}

impl<T: Float> PolyaGamma<T>
where
    StandardNormal: Distribution<T>,
    StandardUniform: Distribution<T>,
    Exp1: Distribution<T>,
{
    pub fn new(h: T, z: T) -> Self {
        let eps = T::from(1e-4).unwrap();
        // if h.is_sign_negative() {
        if h < eps {
            panic!("h must be positive (and not too small)")
        }

        Self { h, z }
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn mean(&self) -> T {
        if self.z == T::zero() {
            self.h / 4.0
        } else {
            self.h * 0.5 * self.z.recip() * (0.5 * self.z).tanh()
        }
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn var(&self) -> T {
        if self.z == T::zero() {
            self.h / 24.0
        } else if self.z.sinh().is_infinite() {
            self.h
                * 0.25
                * (self.z.powi(3).recip() * 2.0 * self.z.signum()
                    - self.z.recip().powi(2) * sech(0.5 * self.z).powi(2))
        } else {
            self.h
                * 0.25
                * self.z.powi(3).recip()
                * (self.z.sinh() - self.z)
                * sech(0.5 * self.z).powi(2)
        }
    }

    #[replace_float_literals(T::from(literal).unwrap())]
    pub fn sample<R: Rng>(&self, rng: &mut R) -> T {
        assert!(self.h > T::zero(), "h must be non-negative");

        // if self.h >= 50.0 {
        //     return self.sample_normal(rng);
        // } else {
        //     return self.sample_saddlepoint(rng);
        // };

        if self.h >= 50.0 {
            self.sample_normal(rng)
        } else if self.h >= 8.0 || (self.h > 4.0 && self.z <= 4.0) {
            self.sample_saddlepoint(rng)
        } else {
            self.sample_alternate(rng)
        }

        // } else if self.h >= 8.0 || (self.h > 4.0 && self.z <= 4.0) {
        //     return self.sample_saddlepoint(rng);
        // } else if self.h == 1.0 || (self.h == self.h.floor() && self.z <= 1.0) {
        //     return self.sample_devroye(rng);
        // } else {
        //     return self.sample_alternate(rng);
        // }
    }

    fn sample_normal<R: Rng>(&self, rng: &mut R) -> T {
        Normal::new(self.mean(), self.var().sqrt())
            .unwrap()
            .sample(rng)
    }

    fn sample_saddlepoint<R: Rng>(&self, rng: &mut R) -> T {
        sample_polyagamma_saddlepoint(rng, self.h, self.z)
    }

    fn sample_alternate<R: Rng>(&self, rng: &mut R) -> T {
        T::from(sample_polyagamma_alternate(
            rng,
            self.h.as_f64(),
            self.z.as_f64(),
        ))
        .unwrap()
    }
}

#[test]
fn try_pg_sampler() {
    let mut rng = rand::rng();
    let pg = PolyaGamma::new(2.0, 2.0);
    let mut rs = Vec::new();
    for _ in 0..1000 {
        rs.push(pg.sample(&mut rng));
    }
    dbg!(rs);
}
