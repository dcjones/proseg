
use std::cmp::Ordering;

fn clockwise_cmp(c: (f32, f32), a: (f32, f32), b: (f32, f32)) -> Ordering {
    // From: https://stackoverflow.com/a/6989383
    if a == b {
        return Ordering::Equal;
    } else if a.0 >= c.0 && b.0 < c.0 {
        return Ordering::Less;
    } else if a.0 < c.0 && b.0 >= c.0 {
        return Ordering::Greater;
    } else if a.0 == c.0 && b.0 == c.0 {
        if a.1 < c.1 && b.1 > c.1 {
            return Ordering::Greater;
        } else if a.1 > c.1 && b.1 < c.1 {
            return Ordering::Less;
        }
    }

    // compute the cross product of vectors (c -> a) x (c -> b)
    let det = (a.0 - c.0) * (b.1 - c.1) - (b.0 - c.0) * (a.1 - c.1);

    if det < 0.0 {
        Ordering::Less
    } else if det > 0.0 {
        Ordering::Greater
    } else {
        // points a and b are on the same line from the c
        // break these ties using distance from c
        let d1 = (a.0 - c.0) * (a.0 - c.0) + (a.1 - c.1) * (a.1 - c.1);
        let d2 = (b.0 - c.0) * (b.0 - c.0) + (b.1 - c.1) * (b.1 - c.1);
        d2.partial_cmp(&d1).unwrap()
    }
}

fn center(vertices: &[(f32, f32)]) -> (f32, f32) {
    let mut x = 0.0;
    let mut y = 0.0;
    for v in vertices {
        x += v.0;
        y += v.1;
    }
    x /= vertices.len() as f32;
    y /= vertices.len() as f32;
    (x, y)
}

pub fn polygon_area(vertices: &mut [(f32, f32)]) -> f32 {
    let c = center(vertices);
    vertices.sort_unstable_by(|a, b| clockwise_cmp(c, *a, *b));

    let mut area = 0.0;

    for (i, u) in vertices.iter().enumerate() {
        let j = (i + 1) % vertices.len();
        let v = vertices[j];

        // triangle formula.
        // area += u.0 * v.1 - v.0 * u.1;

        // trapezoid formula (this is more numerically stable with large coordinates)
        area += (v.0 + u.0) * (v.1 - u.1);
    }
    area = area.abs() / 2.0;

    area
}
